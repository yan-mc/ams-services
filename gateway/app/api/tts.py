"""TTS 运行时模型列表（只读）。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from ..security import require_token

router = APIRouter(tags=["tts"], dependencies=[Depends(require_token)])


@router.get("/tts/models")
def list_tts_models(request: Request) -> dict:
    settings = request.app.state.settings
    root_value = os.environ.get("AMS_TTS_ROOT", "").strip()
    roots: list[Path] = []
    if root_value:
        roots.append(Path(root_value).expanduser() / "models")
    # User-editable TTS voices live outside the versioned runtime so component
    # updates cannot overwrite them. List them first.
    roots.insert(0, settings.tts_voices_dir)

    models: list[dict] = []
    # De-duplicate by model name, not by absolute path. The user directory is
    # scanned first, so a user copy intentionally overrides the bundled runtime
    # model with the same stem instead of showing two identical options.
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.onnx")):
            key = path.stem.casefold()
            if key in seen:
                continue
            seen.add(key)
            models.append({
                "name": path.stem,
                "path": str(path),
                "source": "user" if root == settings.tts_voices_dir else "runtime",
            })
    return {"ok": True, "data": {"root": root_value or None, "user_root": str(settings.tts_voices_dir), "models": models}}