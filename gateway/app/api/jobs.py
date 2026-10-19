"""任务 API：创建/查询/取消/SSE/产物下载。"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..db import TERMINAL_STATUSES
from ..errors import ApiError
from ..models import (
    ArtifactListResponse,
    ArtifactOut,
    JobCreateRequest,
    JobListResponse,
    JobOut,
)
from ..security import require_token

logger = logging.getLogger("ams.gateway.api.jobs")

router = APIRouter(tags=["jobs"], dependencies=[Depends(require_token)])

SSE_KEEPALIVE_SECONDS = 15.0


def _sse_frame(event: dict) -> str:
    payload = json.dumps(event.get("data") or {}, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.get('id')}\nevent: {event.get('kind')}\ndata: {payload}\n\n"


@router.post("/jobs", status_code=201, response_model=JobOut)
def create_job(body: JobCreateRequest, request: Request) -> dict:
    service = request.app.state.service
    return service.submit(
        job_type=body.type,
        params=body.params,
        project_id=body.project_id,
        agent_session_id=body.agent_session_id,
    )


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    request: Request,
    status: str | None = Query(default=None, description="按状态过滤"),
    type: str | None = Query(default=None, description="按任务类型过滤"),
    project_id: str | None = Query(default=None, description="按工程过滤"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    if status is not None and status not in {
        "queued",
        "running",
        "stopping",
        "succeeded",
        "failed",
        "cancelled",
    }:
        raise ApiError(422, "INVALID_PARAMS", f"非法 status：{status}")
    service = request.app.state.service
    jobs, total = service.list_jobs(
        status=status, job_type=type, project_id=project_id, limit=limit, offset=offset
    )
    return {"ok": True, "data": jobs, "total": total, "limit": limit, "offset": offset}


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, request: Request) -> dict:
    return request.app.state.service.get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel_job(job_id: str, request: Request) -> dict:
    return await request.app.state.service.cancel(job_id)


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def list_artifacts(job_id: str, request: Request) -> dict:
    artifacts = request.app.state.service.list_artifacts(job_id)
    return {"ok": True, "data": artifacts}


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, request: Request) -> FileResponse:
    state = request.app.state
    artifact = state.db.get_artifact(artifact_id)
    if artifact is None:
        raise ApiError(404, "ARTIFACT_NOT_FOUND", f"产物不存在：{artifact_id}")
    path = Path(artifact["path"]).resolve()
    artifacts_root = state.settings.artifacts_dir.resolve()
    try:
        path.relative_to(artifacts_root)
    except ValueError as exc:
        raise ApiError(403, "ARTIFACT_FORBIDDEN", "产物路径不在 Gateway 产物目录内") from exc
    if not path.is_file():
        raise ApiError(404, "ARTIFACT_FILE_MISSING", "产物文件已不存在")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=artifact["name"],
    )


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int = Query(default=0, ge=0)):
    """SSE：先回放历史事件，再实时推送；任务进入终态后关闭连接。"""
    state = request.app.state
    service = state.service
    db = state.db
    bus = state.bus
    service.get_job(job_id)  # 不存在时先 404

    last_event_header = request.headers.get("last-event-id")
    if last_event_header and last_event_header.isdigit():
        after = max(after, int(last_event_header))

    async def generator() -> AsyncIterator[str]:
        queue = bus.subscribe(job_id)
        last_seen = after
        try:
            for event in db.get_events(job_id, after_id=last_seen, limit=2000):
                last_seen = int(event["id"])
                yield _sse_frame(event)
            job = db.get_job(job_id)
            if job is not None and job["status"] in TERMINAL_STATUSES:
                # 终态任务：把队列中可能新增的事件补发完再关闭
                while not queue.empty():
                    event = queue.get_nowait()
                    if int(event["id"]) > last_seen:
                        last_seen = int(event["id"])
                        yield _sse_frame(event)
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if int(event["id"]) <= last_seen:
                    continue
                last_seen = int(event["id"])
                yield _sse_frame(event)
                data = event.get("data") or {}
                if event.get("kind") == "status" and data.get("status") in TERMINAL_STATUSES:
                    return
        finally:
            bus.unsubscribe(job_id, queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generator(), media_type="text/event-stream", headers=headers)
