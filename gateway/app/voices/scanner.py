"""voices/ 扫描器（只读）。

规则（00 第 6 节）：
- 一个音源 = voices/<名称>/，文件夹名即显示名/音源 ID；
- 必须存在同名 <名称>.pth；<名称>.index 可选，有则必须同名；
- 校验 Windows 非法字符、首尾空格、保留名；报告重复/损坏/缺失。
本模块只读取目录与文件元数据，不修改任何文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .pth_parser import MetadataParser, NullParser

INVALID_CHARS = set('<>:"/\\|?*')


def normalize_sample_rate(value) -> int | None:
    """把 pth 里的 sr 统一成整数 Hz（兼容 40k/48k/44.1k/40000 等写法）。"""
    if value is None:
        return None
    text = str(value).strip().lower().replace("hz", "").replace(" ", "")
    aliases = {
        "32k": 32000, "32khz": 32000, "32000": 32000,
        "40k": 40000, "40khz": 40000, "40000": 40000,
        "44k": 44100, "44.1k": 44100, "44100": 44100,
        "48k": 48000, "48khz": 48000, "48000": 48000,
    }
    if text in aliases:
        return aliases[text]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None

RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
SMALL_PTH_BYTES = 1 * 1024 * 1024

PTH_EXTENSIONS = {".pth"}
INDEX_EXTENSIONS = {".index"}


@dataclass(slots=True)
class Issue:
    level: str  # info | warning | error
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass(slots=True)
class FileInfo:
    file: str
    exists: bool = True
    size_bytes: int = 0
    mtime: str | None = None

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
        }


@dataclass(slots=True)
class VoiceEntry:
    name: str
    path: str
    status: str = "ok"  # ok | warning | error | pending_parse
    pth: FileInfo | None = None
    index: FileInfo | None = None
    meta: dict | None = None
    meta_parser: str | None = None
    issues: list[Issue] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "pth": self.pth.as_dict() if self.pth else None,
            "index": self.index.as_dict() if self.index else None,
            "meta": self.meta,
            "meta_parser": self.meta_parser,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_info(path: Path) -> FileInfo:
    stat = path.stat()
    return FileInfo(
        file=str(path),
        exists=True,
        size_bytes=stat.st_size,
        mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    )


def validate_folder_name(name: str) -> list[Issue]:
    issues: list[Issue] = []
    if not name or not name.strip():
        issues.append(Issue("error", "NAME_EMPTY", "音源文件夹名为空"))
        return issues
    if name != name.strip():
        issues.append(Issue("error", "NAME_WHITESPACE", "音源文件夹名首尾不能有空格"))
    if name in {".", ".."}:
        issues.append(Issue("error", "NAME_INVALID", "音源文件夹名非法（. 或 ..）"))
    if any(char in INVALID_CHARS for char in name):
        issues.append(
            Issue("error", "NAME_CHARS", '音源文件夹名含 Windows 非法字符（<>:"/\\|?*）')
        )
    if any(ord(char) < 32 for char in name):
        issues.append(Issue("error", "NAME_CONTROL", "音源文件夹名含控制字符"))
    if name.split(".")[0].upper() in RESERVED_NAMES:
        issues.append(Issue("error", "NAME_RESERVED", "音源文件夹名是 Windows 保留名"))
    if len(name) > 120:
        issues.append(Issue("warning", "NAME_LONG", "音源文件夹名过长（>120 字符）"))
    return issues


def _scan_one(directory: Path, parser: MetadataParser) -> VoiceEntry:
    entry = VoiceEntry(name=directory.name, path=str(directory))
    entry.issues.extend(validate_folder_name(directory.name))

    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        entry.status = "error"
        entry.issues.append(Issue("error", "DIR_READ_FAILED", f"目录读取失败：{exc}"))
        return entry

    pth_files = [item for item in children if item.is_file() and item.suffix.lower() in PTH_EXTENSIONS]
    index_files = [
        item for item in children if item.is_file() and item.suffix.lower() in INDEX_EXTENSIONS
    ]
    expected_pth = [item for item in pth_files if item.stem.casefold() == directory.name.casefold()]
    expected_index = [
        item for item in index_files if item.stem.casefold() == directory.name.casefold()
    ]

    if not pth_files:
        entry.status = "error"
        entry.issues.append(Issue("error", "PTH_MISSING", "缺少 .pth 文件"))
    elif not expected_pth:
        entry.status = "error"
        entry.issues.append(
            Issue(
                "error",
                "PTH_NAME_MISMATCH",
                f".pth 文件名必须与文件夹同名：期望 {directory.name}.pth，实际 "
                + ", ".join(item.name for item in pth_files),
            )
        )
    elif len(expected_pth) > 1:
        entry.status = "error"
        entry.issues.append(
            Issue("error", "PTH_DUPLICATE", "存在多个同名 .pth（大小写/扩展名重复）")
        )
    else:
        entry.pth = _file_info(expected_pth[0])
        if entry.pth.size_bytes == 0:
            entry.status = "error"
            entry.issues.append(Issue("error", "PTH_EMPTY", ".pth 文件为空"))
        elif entry.pth.size_bytes < SMALL_PTH_BYTES:
            if entry.status == "ok":
                entry.status = "warning"
            entry.issues.append(
                Issue("warning", "PTH_SMALL", f".pth 体积偏小（{entry.pth.size_bytes} 字节），可能不是 RVC v2 权重")
            )

    extra_pth = [item for item in pth_files if item not in expected_pth]
    if expected_pth and extra_pth:
        if entry.status == "ok":
            entry.status = "warning"
        entry.issues.append(
            Issue("warning", "PTH_EXTRA", "存在多余 .pth 文件：" + ", ".join(item.name for item in extra_pth))
        )

    if expected_index:
        entry.index = _file_info(expected_index[0])
        if len(expected_index) > 1:
            if entry.status == "ok":
                entry.status = "warning"
            entry.issues.append(Issue("warning", "INDEX_DUPLICATE", "存在多个同名 .index 文件"))
    elif index_files:
        if entry.status == "ok":
            entry.status = "warning"
        entry.issues.append(
            Issue(
                "warning",
                "INDEX_NAME_MISMATCH",
                ".index 与文件夹不同名，将被忽略：" + ", ".join(item.name for item in index_files),
            )
        )
    else:
        entry.issues.append(
            Issue("info", "INDEX_OPTIONAL_MISSING", "无 .index（仍可用，index_rate=0）")
        )

    # pth 元数据解析（仅结构正确且文件存在时）
    if entry.pth is not None and entry.status != "error":
        parsed = parser.parse(Path(entry.pth.file))
        entry.meta_parser = parsed.parser
        if parsed.status == "ok":
            entry.meta = parsed.meta
            sr = (parsed.meta or {}).get("sr")
            sr_hz = normalize_sample_rate(sr)
            if sr is not None and sr_hz not in {32000, 40000, 44100, 48000}:
                if entry.status == "ok":
                    entry.status = "warning"
                entry.issues.append(
                    Issue("warning", "SR_UNEXPECTED", f"pth 采样率 {sr} 不是常见 RVC 值（40k/44.1k/48k）")
                )
        elif parsed.status == "pending_parse":
            entry.status = "pending_parse" if entry.status == "ok" else entry.status
            entry.issues.append(Issue("info", "META_PENDING", parsed.reason or "元数据待解析"))
        else:
            if entry.status == "ok":
                entry.status = "warning"
            entry.issues.append(Issue("warning", "META_PARSE_FAILED", parsed.reason or "pth 解析失败"))

    return entry


def scan_voices(root: Path | str, parser: MetadataParser | None = None) -> dict:
    """扫描 voices 根目录，返回结构化结果（只读）。"""
    root_path = Path(root).expanduser()
    parser = parser or NullParser()
    voices: list[VoiceEntry] = []
    root_issues: list[Issue] = []

    if not root_path.exists():
        root_issues.append(Issue("info", "VOICES_DIR_MISSING", f"voices 目录不存在：{root_path}"))
    elif not root_path.is_dir():
        root_issues.append(Issue("error", "VOICES_NOT_DIR", f"voices 路径不是目录：{root_path}"))
    else:
        try:
            entries = sorted(root_path.iterdir(), key=lambda item: item.name.lower())
        except OSError as exc:
            root_issues.append(Issue("error", "VOICES_READ_FAILED", f"无法读取 voices 目录：{exc}"))
            entries = []
        for item in entries:
            if not item.is_dir():
                if item.suffix.lower() in PTH_EXTENSIONS | INDEX_EXTENSIONS:
                    root_issues.append(
                        Issue(
                            "warning",
                            "ROOT_LOOSE_FILE",
                            f"voices 根目录下的散装文件不会被识别（应为 voices/<名称>/<名称>.pth）：{item.name}",
                        )
                    )
                continue
            if item.name.startswith("."):
                continue
            voices.append(_scan_one(item, parser))

    summary = {"ok": 0, "warning": 0, "error": 0, "pending_parse": 0}
    for voice in voices:
        summary[voice.status] = summary.get(voice.status, 0) + 1

    return {
        "root": str(root_path),
        "scanned_at": _utc_now(),
        "parser": parser.name,
        "count": len(voices),
        "summary": summary,
        "issues": [issue.as_dict() for issue in root_issues],
        "voices": [voice.as_dict() for voice in voices],
        "readonly": True,
        "note": "本接口只读扫描，不修改任何文件；pth 元数据解析需配置 RVC runtime（AMS_RVC_PYTHON）",
    }
