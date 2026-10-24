"""convert.rvc：调用 RVC runtime + scripts/rvc_convert.py 做换声。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError


def _require_str(params: dict[str, Any], key: str, label: str) -> str:
    value = params.get(key)
    if value is None or not str(value).strip():
        raise RunnerError("INVALID_PARAMS", f"缺少参数 {key}（{label}）")
    return str(value).strip()


class RvcConvertRunner(ProtocolProcessRunner):
    job_type = "convert.rvc"
    gpu = True
    timeout_sec = 1800.0
    description = "RVC 换声"

    def default_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "scripts" / "rvc_convert.py"

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        input_value = _require_str(params, "input_path", "输入音频路径")
        input_path = Path(input_value).expanduser()
        if not input_path.is_file():
            raise RunnerError(
                "INPUT_NOT_FOUND",
                f"输入音频不存在：{input_path}",
                hint="检查 input_path；支持 wav/flac/mp3/m4a 等 ffmpeg 可解码格式",
            )
        voice_name = _require_str(params, "voice_name", "音源名或 pth 路径")
        f0_up_key = int(params.get("f0_up_key", 0))
        if f0_up_key < -24 or f0_up_key > 24:
            raise RunnerError("INVALID_PARAMS", "f0_up_key 必须在 -24..24 之间")
        index_rate = float(params.get("index_rate", 0.75))
        if not 0.0 <= index_rate <= 1.0:
            raise RunnerError("INVALID_PARAMS", "index_rate 必须在 0..1 之间")
        f0_method = str(params.get("f0_method") or "rmvpe").lower()
        if f0_method not in {"pm", "harvest", "crepe", "rmvpe"}:
            raise RunnerError("INVALID_PARAMS", f"不支持的 f0_method：{f0_method}")
        filter_radius = int(params.get("filter_radius", 3))
        if not 0 <= filter_radius <= 7:
            raise RunnerError("INVALID_PARAMS", "filter_radius 必须在 0..7 之间")
        resample_sr = int(params.get("resample_sr", 0))
        if not 0 <= resample_sr <= 48000:
            raise RunnerError("INVALID_PARAMS", "resample_sr 必须在 0..48000 之间")
        rms_mix_rate = float(params.get("rms_mix_rate", 0.25))
        if not 0.0 <= rms_mix_rate <= 1.0:
            raise RunnerError("INVALID_PARAMS", "rms_mix_rate 必须在 0..1 之间")
        protect = float(params.get("protect", 0.33))
        if not 0.0 <= protect <= 0.5:
            raise RunnerError("INVALID_PARAMS", "protect 必须在 0..0.5 之间")
        index_value = str(params.get("index_path") or "").strip()
        index_path = Path(index_value).expanduser() if index_value else None
        if index_path is not None and not index_path.is_file():
            raise RunnerError("INPUT_NOT_FOUND", f"index 不存在：{index_path}")
        device = str(params.get("device") or "cuda:0")
        if device not in {"cpu", "cuda", "cuda:0", "cuda:1", "mps"}:
            raise RunnerError(
                "INVALID_PARAMS",
                f"不支持的 device：{device}",
                hint="可选 cpu / cuda / cuda:0 / cuda:1（Windows 无 mps）",
            )
        return {
            "input_path": str(input_path),
            "voice_name": voice_name,
            "f0_up_key": f0_up_key,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "index_path": str(index_path) if index_path else "",
            "filter_radius": filter_radius,
            "resample_sr": resample_sr,
            "rms_mix_rate": rms_mix_rate,
            "protect": protect,
            "device": device,
        }

    def build_command(
        self,
        cfg: BackendConfig,
        params: dict[str, Any],
        ctx: JobContext,
    ) -> tuple[list[str], Path | None, dict[str, str] | None]:
        python, root = cfg.require_rvc()
        script = self.resolved_script_path()
        if not script.is_file():
            raise RunnerError(
                "BACKEND_PATH_NOT_FOUND",
                f"换声脚本不存在：{script}",
                hint="gateway 安装包缺少 scripts/rvc_convert.py，请重新安装/更新 gateway 组件",
            )
        safe_voice = "".join(char if char.isalnum() or char in "-_." else "_" for char in params["voice_name"])
        output = ctx.work_dir / f"{Path(params['input_path']).stem}_{safe_voice}.wav"
        command = [
            str(python),
            "-B",
            str(script),
            "--input",
            params["input_path"],
            "--voice",
            params["voice_name"],
            "--output",
            str(output),
            "--rvc-root",
            str(root),
            "--f0-up-key",
            str(params["f0_up_key"]),
            "--index-rate",
            str(params["index_rate"]),
            "--f0-method",
            params["f0_method"],
            "--filter-radius",
            str(params["filter_radius"]),
            "--resample-sr",
            str(params["resample_sr"]),
            "--rms-mix-rate",
            str(params["rms_mix_rate"]),
            "--protect",
            str(params["protect"]),
            "--index-path",
            params.get("index_path") or "",
            "--device",
            params["device"],
        ]
        return command, ctx.work_dir, None
