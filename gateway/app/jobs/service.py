"""Job 服务：队列、状态机、执行、取消与事件。

工作线程（asyncio task）：
- gpu 队列：串行执行 GPU/显存型任务（00 第 8 节 GPU 锁），同一时刻最多 1 个；
- cpu 队列：按 cpu_concurrency 并发执行 CPU 任务。

状态机（00 第 8 节）：
    queued -> running -> succeeded / failed / cancelled
    queued -> cancelled（未开始即取消）
    running -> stopping -> cancelled（取消请求，Runner 抛 JobCancelled 后收口）
服务重启：running/stopping 视为 failed（code=GATEWAY_RESTARTED），queued 重新入队。
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..db import Database, TERMINAL_STATUSES, utc_now
from ..errors import ApiError
from .events import EventBus
from .runner import ArtifactSpec, JobCancelled, JobContext, RunnerRegistry, sha256_file

logger = logging.getLogger("ams.gateway.service")


class JobService:
    """任务引擎（Gateway 单进程内）。"""

    def __init__(
        self,
        *,
        db: Database,
        bus: EventBus,
        registry: RunnerRegistry,
        settings: Settings,
    ) -> None:
        self.db = db
        self.bus = bus
        self.registry = registry
        self.settings = settings
        self._gpu_queue: asyncio.Queue[str] = asyncio.Queue()
        self._cpu_queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._active: dict[str, JobContext] = {}
        self._last_progress: dict[str, tuple[float, str]] = {}
        self._started_at: str | None = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> dict:
        """启动工作线程并做重启恢复（在 FastAPI lifespan 中调用）。"""
        recovery = self.db.recover_on_startup()
        self._started_at = utc_now()
        gpu_task = asyncio.create_task(self._worker_loop(self._gpu_queue, "gpu"), name="ams-gpu-worker")
        self._tasks.append(gpu_task)
        for index in range(max(1, self.settings.cpu_concurrency)):
            task = asyncio.create_task(
                self._worker_loop(self._cpu_queue, f"cpu-{index + 1}"),
                name=f"ams-cpu-worker-{index + 1}",
            )
            self._tasks.append(task)

        for job_id in recovery["queued_ids"]:
            job = self.db.get_job(job_id)
            if not job:
                continue
            queue = self._gpu_queue if self.registry.is_gpu(job["type"]) else self._cpu_queue
            queue.put_nowait(job_id)

        logger.info(
            "JobService 启动：workers=%s 恢复中断=%s 重新入队=%s",
            len(self._tasks),
            recovery["interrupted"],
            recovery["requeued"],
        )
        return recovery

    async def stop(self) -> None:
        """停止工作线程；未完成任务标记失败（M0 不续跑）。"""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        running: list[dict] = []
        for status in ("running", "stopping"):
            jobs, _ = self.db.list_jobs(status=status, limit=100)
            running.extend(jobs)
        error = {"code": "GATEWAY_SHUTDOWN", "message": "Gateway 关闭导致任务中断，请重试"}
        for job in running:
            updated = self.db.update_job(
                job["id"],
                status="failed",
                finished_at=utc_now(),
                error_json=error,
                gpu_lock_json=None,
            )
            if updated:
                self._publish(job["id"], "status", self._status_payload(updated))
        self._active.clear()
        logger.info("JobService 已停止")

    # ------------------------------------------------------------------ 查询
    def get_job(self, job_id: str) -> dict:
        job = self.db.get_job(job_id)
        if job is None:
            raise ApiError(404, "JOB_NOT_FOUND", f"任务不存在：{job_id}")
        return job

    def list_jobs(self, **filters: Any) -> tuple[list[dict], int]:
        return self.db.list_jobs(**filters)

    def list_artifacts(self, job_id: str) -> list[dict]:
        self.get_job(job_id)
        return self.db.list_artifacts(job_id)

    def stats(self) -> dict:
        counts = self.db.job_counts()
        return {
            "jobs": counts,
            "active_jobs": sorted(self._active.keys()),
            "workers": {
                "gpu": 1,
                "cpu": max(1, self.settings.cpu_concurrency),
                "gpu_busy": any(job_id in self._active for job_id in self._gpu_active_ids()),
            },
            "started_at": self._started_at,
        }

    def _gpu_active_ids(self) -> list[str]:
        return [
            job_id
            for job_id, job in self._iter_active_meta()
            if self.registry.is_gpu(job["type"])
        ]

    def _iter_active_meta(self) -> list[tuple[str, dict]]:
        result = []
        for job_id in list(self._active.keys()):
            job = self.db.get_job(job_id)
            if job:
                result.append((job_id, job))
        return result

    # ------------------------------------------------------------------ 提交/取消
    def submit(
        self,
        *,
        job_type: str,
        params: dict,
        project_id: str | None = None,
        agent_session_id: str | None = None,
    ) -> dict:
        if not self.registry.has(job_type):
            known = ", ".join(item["type"] for item in self.registry.describe())
            raise ApiError(
                400,
                "UNKNOWN_JOB_TYPE",
                f"未知任务类型：{job_type}",
                {"known_types": known.split(", ") if known else []},
            )
        gpu = self.registry.is_gpu(job_type)
        engine = {"name": "ams-gateway", "version": self.settings.version, "runner": job_type}
        job = self.db.create_job(
            job_type=job_type,
            params=params,
            project_id=project_id,
            agent_session_id=agent_session_id,
            engine=engine,
            gpu_waiting=gpu,
        )
        self._publish(
            job["id"],
            "status",
            self._status_payload(job),
        )
        self._publish(
            job["id"],
            "log",
            {"level": "info", "message": f"任务已入队（{'GPU 队列' if gpu else 'CPU 队列'}）：{job_type}"},
        )
        queue = self._gpu_queue if gpu else self._cpu_queue
        queue.put_nowait(job["id"])
        logger.info("任务入队 id=%s type=%s gpu=%s", job["id"], job_type, gpu)
        return job

    async def cancel(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] in TERMINAL_STATUSES:
            # 幂等：已完成任务再次取消直接返回当前状态
            return job
        if job["status"] == "queued":
            updated = self.db.try_cancel_queued(job_id)
            if updated is None:
                return self.get_job(job_id)
            self._publish(job_id, "status", self._status_payload(updated))
            self._publish(job_id, "log", {"level": "warning", "message": "任务在排队阶段被取消"})
            logger.info("任务取消（queued）：%s", job_id)
            return updated

        updated = self.db.update_job(
            job_id,
            status="stopping",
            cancel_requested=1,
            stage="stopping",
        )
        if updated:
            self._publish(job_id, "status", self._status_payload(updated))
            self._publish(job_id, "log", {"level": "warning", "message": "收到取消请求，正在终止任务"})
        ctx = self._active.get(job_id)
        if ctx is not None:
            ctx.request_cancel()
        else:
            # 竞态兜底：任务不在活跃表（尚未开始/刚结束），直接标记 cancelled
            updated = self.db.update_job(
                job_id,
                status="cancelled",
                finished_at=utc_now(),
                gpu_lock_json=None,
            )
            if updated:
                self._publish(job_id, "status", self._status_payload(updated))
        logger.info("任务取消请求：%s", job_id)
        return self.get_job(job_id)

    # ------------------------------------------------------------------ 工作线程
    async def _worker_loop(self, queue: asyncio.Queue[str], worker_name: str) -> None:
        while True:
            job_id = await queue.get()
            try:
                await self._run_job(job_id, worker_name)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 工作线程必须活到进程结束
                logger.exception("工作线程异常 worker=%s job=%s", worker_name, job_id)
            finally:
                queue.task_done()

    async def _run_job(self, job_id: str, worker_name: str) -> None:
        job = self.db.get_job(job_id)
        if job is None or job["status"] != "queued":
            return
        gpu = self.registry.is_gpu(job["type"])
        gpu_lock = (
            {"state": "held", "worker": worker_name, "since": utc_now()} if gpu else None
        )
        started = self.db.try_start_job(job_id, gpu_lock=gpu_lock)
        if started is None:
            # 竞争失败（例如排队阶段已被取消）
            return
        job = started
        self._publish(job_id, "status", self._status_payload(job))
        self._publish(job_id, "log", {"level": "info", "message": f"任务开始执行（{worker_name}）"})

        work_dir = self.settings.artifacts_dir / job_id
        ctx = JobContext(
            job_id=job_id,
            job_type=job["type"],
            params=dict(job.get("params") or {}),
            work_dir=work_dir,
            progress_cb=lambda percent, stage, message: self._on_progress(
                job_id, percent, stage, message
            ),
            log_cb=lambda level, message: self._on_log(job_id, level, message),
        )
        self._active[job_id] = ctx
        try:
            runner = self.registry.create(job["type"])
            specs: list[ArtifactSpec] = await runner.run(ctx)
            if ctx.cancelled:
                raise JobCancelled(job_id)
            artifacts = [self._register_artifact(job_id, spec) for spec in specs]
            finished = self.db.update_job(
                job_id,
                status="succeeded",
                progress=100.0,
                stage="done",
                finished_at=utc_now(),
                gpu_lock_json=None,
            )
            for artifact in artifacts:
                self._publish(job_id, "artifact", artifact)
            if finished:
                self._publish(job_id, "status", self._status_payload(finished))
            self._publish(job_id, "log", {"level": "info", "message": "任务执行成功"})
        except JobCancelled:
            finished = self.db.update_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                finished_at=utc_now(),
                gpu_lock_json=None,
            )
            if finished:
                self._publish(job_id, "status", self._status_payload(finished))
            self._publish(job_id, "log", {"level": "warning", "message": "任务已取消"})
            logger.info("任务已取消：%s", job_id)
        except Exception as exc:  # noqa: BLE001 - 任务级错误统一收口
            detail = traceback.format_exc(limit=8)
            # 后端 Runner 可抛 RunnerError(code/message/detail/hint)，原样写入 Job.error 便于前端/Agent 排障
            error = {
                "code": getattr(exc, "code", "RUNNER_ERROR"),
                "message": getattr(exc, "message", str(exc)),
                "detail": getattr(exc, "detail", None)
                if getattr(exc, "detail", None) is not None
                else detail,
            }
            hint = getattr(exc, "hint", None)
            if hint:
                error["hint"] = hint
            finished = self.db.update_job(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error_json=error,
                gpu_lock_json=None,
            )
            if finished:
                self._publish(job_id, "status", self._status_payload(finished))
            self._publish(job_id, "log", {"level": "error", "message": f"任务失败：{exc}"})
            logger.exception("任务失败 id=%s type=%s", job_id, job.get("type"))
        finally:
            self._active.pop(job_id, None)
            self._last_progress.pop(job_id, None)

    # ------------------------------------------------------------------ 回调/事件
    async def _on_progress(
        self, job_id: str, percent: float, stage: str, message: str | None
    ) -> None:
        previous = self._last_progress.get(job_id)
        if previous is not None:
            last_percent, last_stage = previous
            if last_stage == stage and abs(percent - last_percent) < 1.0 and percent < 100.0:
                return
        self._last_progress[job_id] = (percent, stage)
        self.db.update_job(job_id, progress=percent, stage=stage)
        payload: dict[str, Any] = {"progress": round(percent, 1), "stage": stage}
        if message:
            payload["message"] = message
        self._publish(job_id, "progress", payload)

    def _on_log(self, job_id: str, level: str, message: str) -> None:
        self._publish(job_id, "log", {"level": level, "message": message})

    def _publish(self, job_id: str, kind: str, data: dict) -> dict:
        event = self.db.add_event(
            job_id,
            kind,
            data,
            max_events_per_job=max(50, self.settings.event_ring),
        )
        self.bus.publish(job_id, event)
        return event

    @staticmethod
    def _status_payload(job: dict) -> dict:
        return {
            "status": job["status"],
            "progress": job["progress"],
            "stage": job.get("stage"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "error": job.get("error"),
            "gpu_lock": job.get("gpu_lock"),
        }

    # ------------------------------------------------------------------ 产物
    def _register_artifact(self, job_id: str, spec: ArtifactSpec) -> dict:
        path = Path(spec.path).resolve()
        if not path.is_file():
            raise RuntimeError(f"Runner 声明的产物不存在：{path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        return self.db.add_artifact(
            job_id=job_id,
            name=spec.name,
            path=str(path),
            kind=spec.kind,
            size=size,
            sha256=digest,
            meta=spec.meta,
        )
