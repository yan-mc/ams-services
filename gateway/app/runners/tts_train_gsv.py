"""tts.train.gsv：在 runtime 内执行 V5turbo LoRA + GPT 训练。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError

SAFE_ID = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


class TtsTrainGsvRunner(ProtocolProcessRunner):
    job_type = "tts.train.gsv"
    gpu = True
    timeout_sec = 21600.0
    description = "GPT-SoVITS V5turbo 音源炼制（LoRA + GPT -> voice.json）"

    def default_script_path(self) -> Path:
        # actual path is inside the runtime; build_command resolves it.
        return Path(__file__)

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset_value = str(params.get("dataset_dir") or "").strip()
        if not dataset_value:
            raise RunnerError("INVALID_PARAMS", "缺少 dataset_dir")
        dataset = Path(dataset_value).expanduser().resolve()
        if not dataset.is_dir():
            raise RunnerError("INPUT_NOT_FOUND", f"数据集目录不存在：{dataset}")
        wav_dir = dataset / "wav" if (dataset / "wav").is_dir() else dataset
        if not any(wav_dir.glob("*.wav")):
            raise RunnerError("INVALID_PARAMS", "数据集缺少 wav/*.wav")
        voice_id = str(params.get("voice_id") or "").strip()
        if not SAFE_ID.match(voice_id):
            raise RunnerError("INVALID_PARAMS", "voice_id 只能包含中英文/数字/下划线/连字符，1-64 字符")
        version = str(params.get("version") or "v5turbo")
        if version != "v5turbo":
            raise RunnerError("INVALID_PARAMS", "P2 训练仅支持 v5turbo；v2ProPlus 稍后接入")
        epochs_sovits = int(params.get("epochs_sovits") or 8)
        epochs_gpt = int(params.get("epochs_gpt") or 10)
        if not 1 <= epochs_sovits <= 16:
            raise RunnerError("INVALID_PARAMS", "epochs_sovits 必须在 1..16")
        if not 1 <= epochs_gpt <= 20:
            raise RunnerError("INVALID_PARAMS", "epochs_gpt 必须在 1..20")
        batch_sovits = int(params.get("batch_size_sovits") or 1)
        batch_gpt = int(params.get("batch_size_gpt") or 4)
        if not 1 <= batch_sovits <= 4:
            raise RunnerError("INVALID_PARAMS", "batch_size_sovits 必须在 1..4")
        if not 1 <= batch_gpt <= 8:
            raise RunnerError("INVALID_PARAMS", "batch_size_gpt 必须在 1..8")
        ref_audio = str(params.get("ref_audio") or "").strip()
        if ref_audio:
            ref = Path(ref_audio).expanduser().resolve()
            if not ref.is_file():
                raise RunnerError("INPUT_NOT_FOUND", f"参考音频不存在：{ref}")
            ref_audio = str(ref)
        return {
            "dataset_dir": str(dataset),
            "voice_id": voice_id,
            "name": str(params.get("name") or voice_id),
            "language": str(params.get("language") or "zh"),
            "version": version,
            "epochs_sovits": epochs_sovits,
            "epochs_gpt": epochs_gpt,
            "batch_size_sovits": batch_sovits,
            "batch_size_gpt": batch_gpt,
            "lora_rank": max(8, min(128, int(params.get("lora_rank") or 32))),
            "ref_audio": ref_audio,
            "ref_text": str(params.get("ref_text") or "").strip(),
        }

    def build_command(self, cfg: BackendConfig, params: dict[str, Any], ctx: JobContext) -> tuple[list[str], Path | None, dict[str, str] | None]:
        ams_home_value = os.environ.get("AMS_HOME", "").strip()
        if not ams_home_value:
            raise RunnerError("BACKEND_NOT_CONFIGURED", "AMS_HOME 未配置")
        ams_home = Path(ams_home_value).expanduser()
        root_value = os.environ.get("AMS_TTS_GSV_ROOT", "").strip()
        root = Path(root_value).expanduser() if root_value else ams_home / "runtime" / "tts-gsv" / "0.1.0"
        python = root / "runtime" / "python.exe"
        script = root / "tts_service" / "tts_train_service.py"
        if not python.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"runtime Python 不存在：{python}", hint="安装 runtime.tts-gsv-core")
        if not script.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"训练驱动不存在：{script}")
        home_value = os.environ.get("AMS_TTS_GSV_HOME", "").strip()
        home = Path(home_value).expanduser() if home_value else ams_home / "tts-gsv"
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), "-u", "-B", str(script),
            "--runtime-root", str(root), "--home", str(home),
            "--dataset-dir", params["dataset_dir"], "--voice-id", params["voice_id"],
            "--name", params["name"], "--language", params["language"],
            "--version", params["version"],
            "--epochs-sovits", str(params["epochs_sovits"]),
            "--epochs-gpt", str(params["epochs_gpt"]),
            "--batch-size-sovits", str(params["batch_size_sovits"]),
            "--batch-size-gpt", str(params["batch_size_gpt"]),
            "--lora-rank", str(params["lora_rank"]),
        ]
        if params.get("ref_audio"):
            command += ["--ref-audio", params["ref_audio"]]
        if params.get("ref_text"):
            command += ["--ref-text", params["ref_text"]]
        env = {
            "TEMP": str(tmp), "TMP": str(tmp),
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        }
        return command, root, env
