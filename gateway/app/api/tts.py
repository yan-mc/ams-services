"""TTS 运行时模型列表（只读）。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from ..security import require_token

router = APIRouter(tags=["tts"], dependencies=[Depends(require_token)])


@router.get("/tts/models")
def list_tts_models(request: Request) -> dict:
    root_value = os.environ.get("AMS_TTS_ROOT", "").strip()
    if not root_value:
        return {"ok": True, "data": {"root": None, "models": []}}
    root = Path(root_value).expanduser()
    models_dir = root / "models"
    models = []
    if models_dir.is_dir():
        for path in sorted(models_dir.glob("*.onnx")):
            models.append({"name": path.stem, "path": str(path)})
    return {"ok": True, "data": {"root": str(root), "models": models}}