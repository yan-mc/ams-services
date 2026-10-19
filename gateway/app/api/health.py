"""GET /api/v1/health：Gateway 健康与运行信息。"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from ..security import require_token

router = APIRouter(tags=["health"], dependencies=[Depends(require_token)])


@router.get("/health")
def health(request: Request) -> dict:
    state = request.app.state
    settings = state.settings
    stats = state.service.stats()
    return {
        "ok": True,
        "status": "ok",
        "version": settings.version,
        "started_at": state.started_at,
        "uptime_sec": round(time.monotonic() - state.started_monotonic, 3),
        "http": {
            "host": settings.host,
            "port": getattr(state, "port", None),
            "api_prefix": "/api/v1",
        },
        "auth": {
            "scheme": "bearer",
            "token_file": str(settings.token_file),
        },
        "db": {
            "path": str(settings.db_path),
            "exists": Path(settings.db_path).exists(),
            "jobs": stats["jobs"],
        },
        "workers": stats["workers"],
        "active_jobs": stats["active_jobs"],
        "job_types": state.registry.describe(),
        "voices": {
            "path": str(settings.voices_dir),
            "exists": settings.voices_dir.exists(),
        },
        "limits": {
            "cpu_concurrency": settings.cpu_concurrency,
            "event_ring": settings.event_ring,
        },
    }
