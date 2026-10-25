"""train.tts: Piper/VITS voice training on Windows Native runtime."""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Any
from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")

class TtsTrainRunner(ProtocolProcessRunner):
    job_type = "train.tts"
    gpu = True
    timeout_sec = 21600.0
    description = "TTS 音源炼制（Piper/VITS：数据集 -> onnx + onnx.json -> 注册）"

    def default_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "scripts" / "tts_train.py"

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset_value = str(params.get("dataset_dir") or "").strip()
        if not dataset_value:
            raise RunnerError("INVALID_PARAMS", "缺少 dataset_dir")
        dataset = Path(dataset_value).expanduser().resolve()
        if not dataset.is_dir():
            raise RunnerError("INPUT_NOT_FOUND", f"数据集目录不存在：{dataset}")
        if not (dataset / "metadata.csv").is_file() or not (dataset / "wav").is_dir():
            raise RunnerError("INVALID_PARAMS", "TTS 数据集需要 metadata.csv 和 wav/")
        model_name = str(params.get("model_name") or params.get("model") or "").strip()
        if not SAFE_NAME.match(model_name):
            raise RunnerError("INVALID_PARAMS", "模型名只能包含中英文/数字/下划线/连字符，1-64 字符")
        epochs = int(params.get("epochs", 1))
        if not 1 <= epochs <= 2000:
            raise RunnerError("INVALID_PARAMS", "epochs 必须在 1..2000")
        batch_size = int(params.get("batch_size", 2))
        if not 1 <= batch_size <= 32:
            raise RunnerError("INVALID_PARAMS", "batch_size 必须在 1..32")
        base = str(params.get("base_checkpoint") or "").strip()
        return {"dataset_dir":str(dataset),"model_name":model_name,"epochs":epochs,
                "batch_size":batch_size,"base_checkpoint":base,
                "quality":str(params.get("quality") or "medium"),
                "language":str(params.get("language") or "cmn"),
                "sample_rate":int(params.get("sample_rate") or 22050)}

    def build_command(self, cfg: BackendConfig, params: dict[str, Any], ctx: JobContext) -> tuple[list[str], Path | None, dict[str,str] | None]:
        ams = Path(os.environ.get("AMS_HOME", "")).expanduser()
        if not ams.is_dir():
            raise RunnerError("BACKEND_NOT_CONFIGURED", "AMS_HOME 未配置")
        train_root = ams / "runtime" / "tts-train" / "0.1.0"
        independent = train_root / "python" / "python.exe"
        python = independent if independent.is_file() else cfg.require_rvc()[0]
        script = self.resolved_script_path()
        if not script.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"TTS 训练脚本不存在：{script}")
        if not (train_root / "piper" / "src" / "python").is_dir():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"TTS 训练运行时不存在：{train_root}")
        command=[str(python),"-u","-B",str(script),
                 "--root",str(train_root),"--ams-home",str(ams),
                 "--dataset-dir",params["dataset_dir"],"--model-name",params["model_name"],
                 "--epochs",str(params["epochs"]),"--batch-size",str(params["batch_size"]),
                 "--quality",params["quality"],"--language",params["language"],
                 "--sample-rate",str(params["sample_rate"])]
        if params["base_checkpoint"]:
            command += ["--base-checkpoint",params["base_checkpoint"]]
        return command, None, None
