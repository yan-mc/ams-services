"""pth 元数据解析器（可替换）。

策略（M0.3）：
1. 配置了 RVC Python（AMS_RVC_PYTHON，且文件存在）时，用子进程 torch.load 只读解析；
2. 未配置/不可用时返回 pending_parse，等 M0.4 或启动器注入 RVC runtime 后重扫。

解析器通过 MetadataParser 协议注入，测试可替换为 FakeParser；绝不写 pth 文件。
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

RVC_PARSE_SCRIPT = """
import json
import sys

path = sys.argv[1]
try:
    import torch
except Exception as exc:
    print(json.dumps({"status": "error", "reason": "torch_import_failed: %s" % exc}))
    raise SystemExit(0)
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": "load_failed: %s" % exc}))
        raise SystemExit(0)
except Exception as exc:
    print(json.dumps({"status": "error", "reason": "load_failed: %s" % exc}))
    raise SystemExit(0)


def pick(*keys):
    if not isinstance(ckpt, dict):
        return None
    for key in keys:
        if key in ckpt:
            return ckpt[key]
    return None


meta = {
    "info": pick("info"),
    "sr": pick("sr"),
    "f0": pick("f0"),
    "version": pick("version"),
}
weight = pick("weight")
if hasattr(weight, "shape"):
    meta["weight_shape"] = [int(value) for value in weight.shape]
print(json.dumps({"status": "ok", "meta": meta}, ensure_ascii=False, default=str))
"""


@dataclass(slots=True)
class PthMetadata:
    """pth 解析结果。"""

    status: str  # ok | pending_parse | error
    meta: dict | None = None
    reason: str | None = None
    parser: str = "unknown"


class MetadataParser(Protocol):
    name: str

    def parse(self, path: Path) -> PthMetadata:
        ...


class NullParser:
    """未配置 RVC runtime：不解析，返回待解析状态。"""

    name = "null"

    def parse(self, path: Path) -> PthMetadata:
        return PthMetadata(
            status="pending_parse",
            reason="未配置 RVC runtime（AMS_RVC_PYTHON）；可先展示文件信息，M0.4 注入后重扫",
            parser=self.name,
        )


class RvcSubprocessParser:
    """通过 RVC 的 Python 子进程（含 torch）只读解析 pth。"""

    def __init__(self, python_path: str | Path, timeout_sec: float = 60.0) -> None:
        self.python_path = str(python_path)
        self.timeout_sec = timeout_sec
        self.name = "rvc-subprocess"

    def parse(self, path: Path) -> PthMetadata:
        command = [self.python_path, "-B", "-c", RVC_PARSE_SCRIPT, str(path)]
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PthMetadata(status="error", reason="解析超时", parser=self.name)
        except OSError as exc:
            return PthMetadata(status="error", reason=f"无法启动解析子进程：{exc}", parser=self.name)

        stdout = (completed.stdout or "").strip()
        last_line = stdout.splitlines()[-1] if stdout else ""
        if not last_line:
            reason = (completed.stderr or "").strip()[:300] or f"子进程退出码 {completed.returncode}"
            return PthMetadata(status="error", reason=reason, parser=self.name)
        try:
            payload = json.loads(last_line)
        except ValueError:
            return PthMetadata(
                status="error",
                reason=f"解析输出不是 JSON：{last_line[:200]}",
                parser=self.name,
            )
        if payload.get("status") == "ok":
            return PthMetadata(status="ok", meta=payload.get("meta") or {}, parser=self.name)
        return PthMetadata(
            status="error",
            reason=str(payload.get("reason") or "未知解析错误"),
            parser=self.name,
        )


def build_parser(rvc_python: str | None) -> MetadataParser:
    if rvc_python:
        candidate = Path(rvc_python).expanduser()
        if candidate.is_file():
            return RvcSubprocessParser(candidate)
    return NullParser()
