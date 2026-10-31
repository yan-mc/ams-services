"""GPT-SoVITS TTS 只读 API（状态/音色列表）。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ..security import require_token

router = APIRouter(tags=["tts-gsv"], dependencies=[Depends(require_token)])
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


def _gsv_home(request: Request) -> Path:
    settings = request.app.state.settings
    value = os.environ.get("AMS_TTS_GSV_HOME", "").strip()
    return Path(value).expanduser() if value else settings.tts_gsv_home


def _read_voice(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "id": str(data.get("id") or path.parent.name),
        "name": str(data.get("name") or data.get("display_name") or path.parent.name),
        "version": str(data.get("version") or ""),
        "language": str(data.get("language") or "zh"),
        "created_at": data.get("created_at"),
        "tags": data.get("tags") or [],
        "dir": str(path.parent),
    }


@router.get("/tts/gsv/status")
def gsv_status(request: Request) -> dict:
    root_value = os.environ.get("AMS_TTS_GSV_ROOT", "").strip()
    settings = request.app.state.settings
    root = Path(root_value).expanduser() if root_value else settings.ams_home / "runtime" / "tts-gsv" / "0.1.0"
    home = _gsv_home(request)
    python = root / "runtime" / "python.exe"
    service = root / "tts_service" / "tts_service.py"
    voices_dir = home / "voices"
    voices = sorted(p.parent.name for p in voices_dir.glob("*/voice.json")) if voices_dir.is_dir() else []
    return {
        "ok": True,
        "data": {
            "root": str(root),
            "root_ready": root.is_dir() and python.is_file() and service.is_file(),
            "python": str(python),
            "service": str(service),
            "home": str(home),
            "voices_dir": str(voices_dir),
            "voice_count": len(voices),
            "default_engine": "v5turbo",
            "engines": ["v5turbo", "v2ProPlus"],
        },
    }


@router.get("/tts/gsv/voices")
def gsv_voices(request: Request) -> dict:
    home = _gsv_home(request)
    voices_dir = home / "voices"
    items = []
    if voices_dir.is_dir():
        for path in sorted(voices_dir.glob("*/voice.json")):
            try:
                items.append(_read_voice(path))
            except Exception:  # noqa: BLE001 - 坏音色包不应让列表 500
                items.append({"id": path.parent.name, "name": path.parent.name, "error": "invalid voice.json"})
    return {"ok": True, "data": {"items": items, "voices_dir": str(voices_dir)}}


@router.get("/tts/gsv/voices/{voice_id}")
def gsv_voice(voice_id: str, request: Request) -> dict:
    if not VOICE_ID_RE.match(voice_id):
        raise HTTPException(status_code=400, detail="voice_id 非法")
    path = _gsv_home(request) / "voices" / voice_id / "voice.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="音色不存在")
    return {"ok": True, "data": _read_voice(path)}
