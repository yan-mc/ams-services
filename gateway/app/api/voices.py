"""音色库扫描 API（只读，不修改文件）。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from ..models import VoiceScanRequest, VoiceScanResponse
from ..security import require_token
from ..voices.scanner import scan_voices

router = APIRouter(tags=["voices"], dependencies=[Depends(require_token)])


@router.post("/voices/scan", response_model=VoiceScanResponse)
async def scan(request: Request, body: VoiceScanRequest | None = None) -> dict:
    state = request.app.state
    settings = state.settings
    requested = (body.path if body else None) or str(settings.voices_dir)
    root = Path(requested).expanduser()
    result = await run_in_threadpool(scan_voices, root, state.voice_parser)
    return {"ok": True, "data": result}


@router.get("/voices", response_model=VoiceScanResponse)
async def list_voices(request: Request) -> dict:
    state = request.app.state
    result = await run_in_threadpool(scan_voices, state.settings.voices_dir, state.voice_parser)
    return {"ok": True, "data": result}
