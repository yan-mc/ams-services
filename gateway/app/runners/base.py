"""外部进程 Runner 基类（异步子进程 + JSONL 协议 + 取消/超时）。

协议（脚本 stdout 每行一个 JSON）：
    {"type":"progress","percent":0-100,"stage":"...","message":"..."?}
    {"type":"log","level":"info|warning|error","message":"..."}
    {"type":"artifact","name":"...","path":"绝对路径","kind":"file|model|report","meta":{...}?}
    {"type":"result","ok":true,"artifacts":[...]?,"meta":{...}?}
    {"type":"result","ok":false,"error":{"code":"...","message":"...","hint":"..."?}}
stderr 逐行转为 Job 日志（[stderr] 前缀）。

取消：Gateway 调 ctx.request_cancel() -> 执行已注册 kill 回调（杀进程树）。
超时：默认见各类 timeout_sec；超时同样杀进程树并抛 BACKEND_TIMEOUT。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..jobs.runner import ArtifactSpec, JobCancelled, JobContext, kill_process_tree
from .config import BackendConfig, RunnerError

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
STDERR_TAIL = 40


def backend_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """构造子进程环境：继承 + UTF-8 + 不写 __pycache__。"""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.update(extra or {})
    return env


class BaseProcessRunner:
    """外部进程 Runner 基类。"""

    job_type = ""
    gpu = False
    timeout_sec = 3600.0
    description = ""

    def __init__(self, *, script_path: str | Path | None = None) -> None:
        self._script_path = Path(script_path).expanduser() if script_path else None

    # ------------------------------------------------------------------ 子类实现
    def default_script_path(self) -> Path:
        raise NotImplementedError

    def resolved_script_path(self) -> Path:
        return self._script_path or self.default_script_path()

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def build_command(
        self,
        cfg: BackendConfig,
        params: dict[str, Any],
        ctx: JobContext,
    ) -> tuple[list[str], Path | None, dict[str, str] | None]:
        """返回 (命令, cwd, 追加环境变量)。"""
        raise NotImplementedError

    async def _consume(self, ctx: JobContext, proc: asyncio.subprocess.Process) -> list[ArtifactSpec]:
        raise NotImplementedError

    # ------------------------------------------------------------------ 主流程
    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        params = self.validate_params(dict(ctx.params or {}))
        self._params = params
        cfg = BackendConfig.from_env()
        command, cwd, extra_env = self.build_command(cfg, params, ctx)
        ctx.log(
            "info",
            f"[{self.job_type}] 启动后端：{Path(command[0]).name} "
            f"{Path(command[1]).name if len(command) > 1 else ''}",
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd) if cwd else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=backend_env(extra_env),
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise RunnerError(
                "BACKEND_START_FAILED",
                f"无法启动后端进程：{exc}",
                hint="检查 AMS_RVC_PYTHON / AMS_PYMSS_PYTHON / AMS_MSVC_TOOL 路径与文件权限",
            ) from exc

        ctx.add_kill_callback(lambda: self._kill(proc))
        started = time.monotonic()
        try:
            artifacts = await asyncio.wait_for(self._consume(ctx, proc), timeout=self.timeout_sec)
        except asyncio.TimeoutError as exc:
            self._kill(proc)
            await self._reap(proc)
            raise RunnerError(
                "BACKEND_TIMEOUT",
                f"{self.job_type} 后端超时（{self.timeout_sec:.0f}s）",
                hint="检查输入大小/显存是否不足；可拆分输入、降低 chunk 或换更快模型",
            ) from exc
        except JobCancelled:
            self._kill(proc)
            await self._reap(proc)
            raise
        finally:
            if proc.returncode is None:
                self._kill(proc)
                await self._reap(proc)
        if ctx.cancelled:
            raise JobCancelled(ctx.job_id)
        ctx.log(
            "info",
            f"[{self.job_type}] 后端完成：{time.monotonic() - started:.1f}s，产物 {len(artifacts)} 个",
        )
        return artifacts

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            try:
                kill_process_tree(proc.pid)
            except Exception:  # noqa: BLE001 - 取消失败不影响状态收口
                pass

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass


class ProtocolProcessRunner(BaseProcessRunner):
    """脚本型后端：stdout 按行 JSON 协议，stderr 转日志。"""

    async def _consume(self, ctx: JobContext, proc: asyncio.subprocess.Process) -> list[ArtifactSpec]:
        artifacts: list[ArtifactSpec] = []
        stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL)
        result: dict[str, Any] | None = None

        async def read_stderr() -> None:
            assert proc.stderr is not None
            async for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                stderr_tail.append(line)
                ctx.log("info", f"[stderr] {line}")

        stderr_task = asyncio.create_task(read_stderr())
        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    ctx.log("info", f"[stdout] {line}")
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    ctx.log("warning", f"[stdout] 无法解析的 JSON 行：{line[:200]}")
                    continue
                kind = payload.get("type")
                if kind == "progress":
                    await ctx.progress(
                        float(payload.get("percent") or 0),
                        str(payload.get("stage") or "running"),
                        str(payload["message"]) if payload.get("message") else None,
                    )
                elif kind == "log":
                    ctx.log(str(payload.get("level") or "info"), str(payload.get("message") or ""))
                elif kind == "artifact":
                    artifacts.append(self._to_spec(payload))
                elif kind == "result":
                    result = payload
                    break
                else:
                    ctx.log("debug", f"[stdout] 未知消息类型 {kind!r}")
            rc = await proc.wait()
        finally:
            await asyncio.gather(stderr_task, return_exceptions=True)

        if ctx.cancelled:
            raise JobCancelled(ctx.job_id)
        if result is None:
            if rc != 0:
                raise RunnerError(
                    "BACKEND_FAILED",
                    f"{self.job_type} 后端退出码 {rc}",
                    detail={"stderr_tail": list(stderr_tail)[-10:]},
                    hint="查看任务日志中的 [stderr] 行；检查模型/输入是否有效",
                )
            raise RunnerError(
                "BACKEND_PROTOCOL_ERROR",
                f"{self.job_type} 后端未返回 result 消息",
                detail={"stderr_tail": list(stderr_tail)[-10:]},
            )
        if not result.get("ok", False):
            error = result.get("error") or {}
            raise RunnerError(
                str(error.get("code") or "BACKEND_FAILED"),
                str(error.get("message") or f"{self.job_type} 后端失败"),
                detail=error.get("detail") or {"stderr_tail": list(stderr_tail)[-10:]},
                hint=error.get("hint"),
            )
        if rc != 0:
            raise RunnerError(
                "BACKEND_FAILED",
                f"{self.job_type} 返回成功但进程退出码为 {rc}",
                detail={"stderr_tail": list(stderr_tail)[-10:]},
            )
        for item in result.get("artifacts") or []:
            if isinstance(item, dict):
                artifacts.append(self._to_spec(item))
        return self._dedupe(artifacts)

    @staticmethod
    def _to_spec(payload: dict[str, Any]) -> ArtifactSpec:
        path_value = payload.get("path")
        if not path_value:
            raise RunnerError("BACKEND_PROTOCOL_ERROR", "artifact 消息缺少 path")
        path = Path(str(path_value)).expanduser()
        name = str(payload.get("name") or path.name)
        return ArtifactSpec(
            name=name,
            path=path,
            kind=str(payload.get("kind") or "file"),
            meta=dict(payload.get("meta") or {}),
        )

    @staticmethod
    def _dedupe(artifacts: list[ArtifactSpec]) -> list[ArtifactSpec]:
        seen: dict[str, ArtifactSpec] = {}
        for item in artifacts:
            seen[str(item.path)] = item
        return list(seen.values())
