"""素材导入与路径边界。

用户选择的音频、视频、pth 等素材默认复制到 AMS_HOME/projects；
Gateway 任务参数在入队前统一经过这里，保证 Runner 看到的输入都在 data 内。
"""

from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import ApiError

INPUT_PATH_KEYS = {
    "input_path",
    "reference_pth",
    "voice_path",
    "index_path",
    "model_path",
    "audio_path",
    "video_path",
}
OPTIONAL_PATH_KEYS = {"voice_name"}
OUTPUT_PATH_KEYS = {"out_dir", "output_dir", "output_path"}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _resolve_source(value: str) -> Path:
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    else:
        source = source.resolve()
    if not source.exists():
        raise ApiError(
            422,
            "INPUT_PATH_MISSING",
            f"输入路径不存在：{source}",
            {"path": str(source), "hint": "请选择存在的音频、视频或模型文件"},
        )
    return source


def _copy_to(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}-{uuid.uuid4().hex[:8]}{destination.suffix}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def materialize_file(
    value: str,
    *,
    settings: Settings,
    project_id: str | None = None,
    bucket: str | None = None,
) -> str:
    """把外部素材复制进 projects，返回 data 内路径；data 内路径原样返回。"""
    source = _resolve_source(value)
    if _inside(source, settings.ams_home):
        return str(source)
    import_id = bucket or uuid.uuid4().hex
    root = settings.imports_dir / (project_id or "unscoped") / import_id
    target = _copy_to(source, root / (source.name or "input"))
    return str(target)


def sanitize_job_params(params: dict[str, Any], *, settings: Settings, project_id: str | None) -> dict[str, Any]:
    """复制所有外部输入到 projects，并把外部输出目录改成 data 内目录。"""
    result = copy.deepcopy(params)
    import_id = uuid.uuid4().hex
    imports: list[dict[str, str]] = []

    for key, value in list(result.items()):
        if not isinstance(value, str) or not value.strip():
            continue
        is_output = key in OUTPUT_PATH_KEYS
        is_input = key in INPUT_PATH_KEYS
        is_optional = key in OPTIONAL_PATH_KEYS
        if not (is_output or is_input or is_optional):
            continue

        if is_output:
            candidate = Path(value).expanduser()
            candidate = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
            if _inside(candidate, settings.ams_home):
                continue
            target = settings.imports_dir / (project_id or "unscoped") / import_id / "output"
            target.mkdir(parents=True, exist_ok=True)
            result[key] = str(target)
            imports.append({"key": key, "from": str(candidate), "to": str(target)})
            continue

        if is_optional:
            # voice_name 既可能是音源名，也可能是 pth 路径；只有看起来存在的路径才复制。
            candidate = Path(value).expanduser()
            if not (candidate.is_file() or candidate.is_dir()):
                continue
            source = _resolve_source(value)
        else:
            source = _resolve_source(value)

        if _inside(source, settings.ams_home):
            continue
        target = _copy_to(source, settings.imports_dir / (project_id or "unscoped") / import_id / source.name)
        result[key] = str(target)
        imports.append({"key": key, "from": str(source), "to": str(target)})

    if imports:
        result["_ams_imports"] = imports
    return result
