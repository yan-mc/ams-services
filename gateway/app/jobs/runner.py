"""任务 Runner 接口、注册表与内置 mock 任务。

为训练/分离等真实后端预留注册点（见 build_default_registry）：
    registry.register("separate.pymss", lambda: PymssRunner(...), gpu=True,
                      description="pymss 分离（M0.7 后接入）")
Runner.run(ctx) 返回产物列表；进度/日志通过 ctx 上报；取消通过 ctx.cancel_event
与 ctx.request_cancel() 触发的 kill 回调实现（mock 用子进程 sleep 模拟长任务）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger("ams.gateway.runner")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class JobCancelled(Exception):
    """任务被取消（Runner 抛出，由 JobService 统一收口为 cancelled 状态）。"""


@dataclass(slots=True)
class ArtifactSpec:
    """Runner 产出的产物声明（落盘文件由 JobService 计算大小与 sha256）。"""

    name: str
    path: Path
    kind: str = "file"
    meta: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[float, str, str | None], Awaitable[None]]
LogCallback = Callable[[str, str], None]


class JobContext:
    """单次任务执行上下文：进度、日志、取消、产物目录。"""

    def __init__(
        self,
        *,
        job_id: str,
        job_type: str,
        params: dict[str, Any],
        work_dir: Path,
        progress_cb: ProgressCallback,
        log_cb: LogCallback,
    ) -> None:
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._progress_cb = progress_cb
        self._log_cb = log_cb
        self._cancel_event = asyncio.Event()
        self._kill_callbacks: list[Callable[[], None]] = []
        self._cancelled = False

    # ---- 上报 ----
    async def progress(self, percent: float, stage: str, message: str | None = None) -> None:
        value = max(0.0, min(100.0, float(percent)))
        await self._progress_cb(value, stage, message)

    def log(self, level: str, message: str) -> None:
        self._log_cb(level, message)

    # ---- 取消 ----
    @property
    def cancel_event(self) -> asyncio.Event:
        return self._cancel_event

    @property
    def cancelled(self) -> bool:
        return self._cancelled or self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled(self.job_id)

    def add_kill_callback(self, callback: Callable[[], None]) -> None:
        self._kill_callbacks.append(callback)

    def request_cancel(self) -> None:
        """由 JobService.cancel 调用：置位取消事件并执行 kill 回调。"""
        self._cancelled = True
        self._cancel_event.set()
        for callback in list(self._kill_callbacks):
            try:
                callback()
            except Exception:  # noqa: BLE001 - 取消失败不应影响其他回调
                logger.exception("取消回调执行失败 job=%s", self.job_id)


class Runner(Protocol):
    """任务执行器协议。"""

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        ...


RunnerFactory = Callable[[], Runner]


@dataclass(slots=True)
class RunnerEntry:
    job_type: str
    factory: RunnerFactory
    gpu: bool = False
    description: str = ""


class RunnerRegistry:
    """任务类型注册表（训练/分离等真实后端的接入点）。"""

    def __init__(self) -> None:
        self._entries: dict[str, RunnerEntry] = {}

    def register(
        self,
        job_type: str,
        factory: RunnerFactory,
        *,
        gpu: bool = False,
        description: str = "",
        replace: bool = False,
    ) -> None:
        if not job_type or not job_type.strip():
            raise ValueError("job_type 不能为空")
        if job_type in self._entries and not replace:
            raise ValueError(f"任务类型已注册：{job_type}")
        self._entries[job_type] = RunnerEntry(job_type, factory, gpu, description)
        logger.info("注册任务类型 type=%s gpu=%s desc=%s", job_type, gpu, description)

    def has(self, job_type: str) -> bool:
        return job_type in self._entries

    def is_gpu(self, job_type: str) -> bool:
        entry = self._entries.get(job_type)
        return bool(entry and entry.gpu)

    def create(self, job_type: str) -> Runner:
        entry = self._entries.get(job_type)
        if entry is None:
            raise KeyError(job_type)
        return entry.factory()

    def describe(self) -> list[dict]:
        return [
            {
                "type": entry.job_type,
                "gpu": entry.gpu,
                "description": entry.description,
            }
            for entry in sorted(self._entries.values(), key=lambda item: item.job_type)
        ]


# --------------------------------------------------------------------- 工具函数
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def kill_process_tree(pid: int) -> None:
    """终止进程树（Windows: taskkill /T /F；POSIX: 进程组 SIGKILL）。"""
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                timeout=10,
                check=False,
            )
        except Exception:  # noqa: BLE001
            logger.exception("taskkill 失败 pid=%s", pid)
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("killpg 失败 pid=%s", pid)


def _spawn_python(code: str, *args: str) -> subprocess.Popen:
    python = os.environ.get("AMS_MOCK_PYTHON") or os.environ.get("AMS_RVC_PYTHON")
    if not python or not Path(python).is_file():
        python = sys.executable if not getattr(sys, "frozen", False) else (shutil.which("python") or "python")
    command = [python, "-c", code, *args]
    kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "creationflags": CREATE_NO_WINDOW if os.name == "nt" else 0,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


# --------------------------------------------------------------------- 内置 mock
class PingRunner:
    """最小任务：验证任务创建 -> 运行 -> 完成 -> 产物 全链路。"""

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        started = time.monotonic()
        ctx.log("info", "ping: 开始")
        await ctx.progress(10, "ping.start")
        ctx.check_cancelled()
        await asyncio.sleep(0.05)
        await ctx.progress(60, "ping.running")
        ctx.check_cancelled()
        await asyncio.sleep(0.05)
        payload = {
            "message": "pong",
            "job_id": ctx.job_id,
            "job_type": ctx.job_type,
            "elapsed_sec": round(time.monotonic() - started, 4),
            "params": ctx.params,
        }
        artifact_path = ctx.work_dir / "ping-result.json"
        write_json_artifact(artifact_path, payload)
        await ctx.progress(100, "ping.done")
        ctx.log("info", "ping: 完成")
        return [ArtifactSpec(name="ping-result.json", path=artifact_path, kind="json", meta=payload)]


class MockLongRunner:
    """长任务 mock：启动一个 sleep 子进程，用于验证进度/取消/杀进程树/GPU 串行。"""

    DEFAULT_SECONDS = 8.0
    MIN_SECONDS = 0.2
    MAX_SECONDS = 300.0

    def __init__(self, sleep_code: str | None = None) -> None:
        self._sleep_code = sleep_code or "import sys, time; time.sleep(float(sys.argv[1]))"

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        seconds = float(ctx.params.get("seconds", self.DEFAULT_SECONDS))
        seconds = max(self.MIN_SECONDS, min(self.MAX_SECONDS, seconds))
        started = time.monotonic()
        ctx.log("info", f"mock.long: 启动子进程 sleep {seconds}s")
        await ctx.progress(2, "mock.starting")

        process = _spawn_python(self._sleep_code, str(seconds))
        ctx.add_kill_callback(lambda: kill_process_tree(process.pid))
        try:
            while True:
                if ctx.cancelled:
                    raise JobCancelled(ctx.job_id)
                return_code = process.poll()
                elapsed = time.monotonic() - started
                percent = min(95.0, 5.0 + (elapsed / seconds) * 90.0)
                await ctx.progress(percent, "mock.sleep", f"已运行 {elapsed:.1f}s/{seconds:.1f}s")
                if return_code is not None:
                    break
                await asyncio.sleep(0.1)
        finally:
            if process.poll() is None:
                kill_process_tree(process.pid)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("mock.long 子进程未按时退出 pid=%s", process.pid)

        if ctx.cancelled:
            raise JobCancelled(ctx.job_id)

        elapsed = time.monotonic() - started
        payload = {
            "message": "mock.long done",
            "job_id": ctx.job_id,
            "requested_seconds": seconds,
            "elapsed_sec": round(elapsed, 3),
            "child_return_code": process.returncode,
        }
        artifact_path = ctx.work_dir / "mock-result.json"
        write_json_artifact(artifact_path, payload)
        await ctx.progress(100, "mock.done")
        ctx.log("info", f"mock.long: 完成，用时 {elapsed:.1f}s")
        return [ArtifactSpec(name="mock-result.json", path=artifact_path, kind="json", meta=payload)]


def build_default_registry() -> RunnerRegistry:
    """内置任务注册（M0 只需 ping/mock.long）。"""
    registry = RunnerRegistry()
    registry.register("ping", lambda: PingRunner(), gpu=False, description="连通性测试任务（CPU）")
    registry.register(
        "mock.long",
        lambda: MockLongRunner(),
        gpu=True,
        description="长任务 mock（子进程 sleep；验证进度/取消/GPU 串行）",
    )
    return registry
