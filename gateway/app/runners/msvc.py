"""convert.msvc：RVC runtime python + ams-tools/msvc2pth.py。

msvc2pth 输出约定：
- 进度/诊断信息以普通文本打印到 stdout（如 "[calibrate] ..."）；
- 最终以 --json 打印完整 conversion report（多行 JSON）；
- 产物：<out_dir>/<name>/<name>.pth 与 <out_dir>/<name>/conversion.json。

Runner 行为：实时把 stdout/stderr 转 Job 日志，按标记推进阶段进度，
结束后从 stdout 提取 JSON（失败则回退读取 conversion.json），产出两个 ArtifactSpec。
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any

from ..jobs.runner import ArtifactSpec, JobCancelled, JobContext
from .base import BaseProcessRunner, STDERR_TAIL
from .config import BackendConfig, RunnerError

PROGRESS_MARKERS = (("[cache]", 30.0), ("[calibrate]", 55.0))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从混合文本中提取第一个 JSON 对象（兼容 msvc2pth 的多行报告）。"""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class MsvcConvertRunner(BaseProcessRunner):
    job_type = "convert.msvc"
    gpu = False
    timeout_sec = 1800.0
    description = "MSVC 容器转 RVC v2 pth（msvc2pth）"

    def default_script_path(self) -> Path:  # msvc 使用外部工具，不使用包内脚本
        raise NotImplementedError

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        input_value = params.get("input_path")
        if not input_value:
            raise RunnerError("INVALID_PARAMS", "缺少参数 input_path（.msvc 文件路径）")
        out_dir = params.get("out_dir")
        if not out_dir:
            raise RunnerError("INVALID_PARAMS", "缺少参数 out_dir（转换输出根目录）")
        input_path = Path(str(input_value)).expanduser()
        if not input_path.is_file():
            raise RunnerError(
                "INPUT_NOT_FOUND",
                f"输入 .msvc 文件不存在：{input_path}",
                hint="检查 input_path；MSVC 文件由用户本地提供",
            )
        output_dir = Path(str(out_dir)).expanduser()
        name = str(params.get("name") or "").strip()
        if name and (os_sep := ("/" in name or "\\" in name)):
            raise RunnerError("INVALID_PARAMS", f"name 不能包含路径分隔符：{name!r}")
        reference = params.get("reference_pth")
        if reference:
            reference_path = Path(str(reference)).expanduser()
            if not reference_path.is_file():
                raise RunnerError(
                    "INPUT_NOT_FOUND",
                    f"reference_pth 不存在：{reference_path}",
                    hint="reference_pth 必须是同架构 RVC pth（可选）；留空则跳过参考校验",
                )
            reference = str(reference_path)
        return {
            "input_path": str(input_path),
            "out_dir": str(output_dir),
            "name": name or None,
            "reference_pth": str(reference) if reference else None,
            "no_cache": bool(params.get("no_cache", False)),
        }

    def build_command(
        self,
        cfg: BackendConfig,
        params: dict[str, Any],
        ctx: JobContext,
    ) -> tuple[list[str], Path | None, dict[str, str] | None]:
        python, tool = cfg.require_msvc()
        command = [str(python), "-B", str(tool), params["input_path"], "--out", params["out_dir"]]
        if params.get("name"):
            command.extend(["--name", params["name"]])
        if params.get("reference_pth"):
            command.extend(["--reference-pth", params["reference_pth"]])
        if params.get("no_cache"):
            command.append("--no-cache")
        command.append("--json")
        return command, ctx.work_dir, None

    async def _consume(self, ctx: JobContext, proc: asyncio.subprocess.Process) -> list[ArtifactSpec]:
        stdout_lines: list[str] = []
        stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL)
        percent = 5.0
        await ctx.progress(percent, "convert.starting", "启动 msvc2pth")

        async def read_stream(stream: asyncio.StreamReader, is_stderr: bool) -> None:
            nonlocal percent
            async for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                if is_stderr:
                    stderr_tail.append(line)
                    ctx.log("info", f"[stderr] {line}")
                    continue
                stdout_lines.append(line)
                marker = 0.0
                for token, value in PROGRESS_MARKERS:
                    if token in line:
                        marker = value
                        break
                if marker > percent:
                    percent = marker
                    await ctx.progress(percent, "convert.calibrating", line[:200])
                else:
                    ctx.log("info", f"[msvc] {line}")

        assert proc.stdout is not None and proc.stderr is not None
        await asyncio.gather(
            read_stream(proc.stdout, False),
            read_stream(proc.stderr, True),
        )
        return_code = await proc.wait()
        if ctx.cancelled:
            raise JobCancelled(ctx.job_id)
        if return_code != 0:
            raise RunnerError(
                "BACKEND_FAILED",
                f"msvc2pth 退出码 {return_code}",
                detail={"stderr_tail": list(stderr_tail)[-10:], "stdout_tail": stdout_lines[-10:]},
                hint="查看任务日志 [msvc]/[stderr]；确认 .msvc 未损坏且 reference_pth 架构匹配",
            )

        report = _extract_json_object("\n".join(stdout_lines))
        out_dir = Path(self._params.get("out_dir", "")) if getattr(self, "_params", None) else None
        if report is None and out_dir is not None:
            candidates = sorted(
                out_dir.glob("*/conversion.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                try:
                    report = json.loads(candidates[0].read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    report = None
        if report is None:
            raise RunnerError(
                "BACKEND_PROTOCOL_ERROR",
                "无法从 msvc2pth 输出解析 JSON 报告",
                detail={"stdout_tail": stdout_lines[-10:]},
            )

        output_pth_value = report.get("output_pth")
        if not output_pth_value or not Path(str(output_pth_value)).is_file():
            raise RunnerError(
                "BACKEND_FAILED",
                "msvc2pth 报告未给出有效的 output_pth",
                detail={"output_pth": output_pth_value, "warnings": report.get("warnings")},
            )
        output_pth = Path(str(output_pth_value))
        artifacts = [
            ArtifactSpec(
                name=output_pth.name,
                path=output_pth,
                kind="model",
                meta={
                    "voice_name": report.get("voice_name"),
                    "tensor_count": report.get("tensor_count"),
                    "warnings": report.get("warnings") or [],
                    "tool_version": report.get("tool_version"),
                },
            )
        ]
        report_path = output_pth.parent / "conversion.json"
        if report_path.is_file():
            artifacts.append(
                ArtifactSpec(
                    name="conversion.json",
                    path=report_path,
                    kind="report",
                    meta={"tool": report.get("tool"), "tool_version": report.get("tool_version")},
                )
            )
        await ctx.progress(100, "convert.done", "转换完成")
        return artifacts
