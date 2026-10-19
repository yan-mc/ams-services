"""voices 扫描测试：命名校验/重复/缺文件/pth 解析器注入（只读）。"""

from __future__ import annotations

from pathlib import Path

from app.voices.pth_parser import PthMetadata
from app.voices.scanner import scan_voices, validate_folder_name

from conftest import write_file


def _make_voice(root: Path, name: str, pth_size: int = 1024, index: bool = False) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    write_file(directory / f"{name}.pth", size=pth_size)
    if index:
        write_file(directory / f"{name}.index", size=128, fill=b"\x01")
    return directory


class FakeParser:
    name = "fake"

    def parse(self, path: Path) -> PthMetadata:
        return PthMetadata(
            status="ok",
            meta={"info": "fake-model", "sr": 48000, "f0": 1, "version": "v2"},
            parser=self.name,
        )


class FailingParser:
    name = "failing"

    def parse(self, path: Path) -> PthMetadata:
        return PthMetadata(status="error", reason="模拟解析失败", parser=self.name)


def test_validate_folder_name_rules():
    assert validate_folder_name("爱莉希雅") == []
    assert any(issue.code == "NAME_WHITESPACE" for issue in validate_folder_name(" 曼波 "))
    assert any(issue.code == "NAME_CHARS" for issue in validate_folder_name("bad/name"))
    assert any(issue.code == "NAME_RESERVED" for issue in validate_folder_name("CON"))
    assert any(issue.code == "NAME_EMPTY" for issue in validate_folder_name(""))


def test_scan_fake_voice_tree(tmp_path: Path):
    root = tmp_path / "voices"
    good = _make_voice(root, "爱莉希雅", pth_size=2 * 1024 * 1024, index=True)
    _make_voice(root, "缺文件")
    (root / "缺文件" / "缺文件.pth").unlink()
    bad = root / "名字不符"
    bad.mkdir()
    write_file(bad / "另一个名字.pth", size=2 * 1024 * 1024)
    _make_voice(root, "小体积", pth_size=1024)

    result = scan_voices(root, FakeParser())
    assert result["readonly"] is True
    assert result["count"] == 4
    names = {voice["name"]: voice for voice in result["voices"]}

    assert names["爱莉希雅"]["status"] == "ok"
    assert names["爱莉希雅"]["meta"]["sr"] == 48000
    assert names["爱莉希雅"]["index"]["exists"] is True

    assert names["缺文件"]["status"] == "error"
    assert any(issue["code"] == "PTH_MISSING" for issue in names["缺文件"]["issues"])

    assert names["名字不符"]["status"] == "error"
    assert any(issue["code"] == "PTH_NAME_MISMATCH" for issue in names["名字不符"]["issues"])

    assert names["小体积"]["status"] == "warning"
    assert any(issue["code"] == "PTH_SMALL" for issue in names["小体积"]["issues"])

    # 只读：扫描不修改文件
    assert (good / "爱莉希雅.pth").is_file()
    assert (good / "爱莉希雅.index").is_file()


def test_scan_without_parser_returns_pending(tmp_path: Path):
    root = tmp_path / "voices"
    _make_voice(root, "待解析音色", pth_size=2 * 1024 * 1024)
    result = scan_voices(root)
    voice = result["voices"][0]
    assert voice["status"] == "pending_parse"
    assert voice["meta_parser"] == "null"
    assert any(issue["code"] == "META_PENDING" for issue in voice["issues"])


def test_scan_meta_parse_failure_is_warning(tmp_path: Path):
    root = tmp_path / "voices"
    _make_voice(root, "坏权重", pth_size=2 * 1024 * 1024)
    result = scan_voices(root, FailingParser())
    voice = result["voices"][0]
    assert voice["status"] == "warning"
    assert any(issue["code"] == "META_PARSE_FAILED" for issue in voice["issues"])


def test_scan_missing_root(tmp_path: Path):
    result = scan_voices(tmp_path / "not-exists")
    assert result["count"] == 0
    assert any(issue["code"] == "VOICES_DIR_MISSING" for issue in result["issues"])


def test_api_voices_scan(client, auth, ams_home: Path):
    voices_root = ams_home / "voices"
    _make_voice(voices_root, "测试音色", pth_size=2 * 1024 * 1024)
    response = client.post("/api/v1/voices/scan", headers=auth, json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["count"] >= 1
    voice = next(item for item in data["voices"] if item["name"] == "测试音色")
    assert voice["status"] == "pending_parse"
    assert voice["pth"]["size_bytes"] > 0
    assert data["parser"] == "null"
    # GET 便捷接口
    listed = client.get("/api/v1/voices", headers=auth)
    assert listed.status_code == 200
    assert listed.json()["data"]["count"] >= 1
