"""train.rvc: controlled RVC dataset training + index + voice registration."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


class RvcTrainRunner(ProtocolProcessRunner):
    job_type = "train.rvc"
    gpu = True
    timeout_sec = 7200.0
    description = "RVC 音源炼制（数据集 → pth + index → 注册音色）"

    def default_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "scripts" / "rvc_train.py"

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset_value = str(params.get("dataset_dir") or "").strip()
        if not dataset_value:
            raise RunnerError("INVALID_PARAMS", "缺少 dataset_dir（数据集目录）")
        dataset = Path(dataset_value).expanduser()
        if not dataset.is_dir():
            raise RunnerError("INPUT_NOT_FOUND", f"数据集目录不存在：{dataset}", hint="先用数据集工作室导入/切片，或选择包含 wav 的目录")
        model_name = str(params.get("model_name") or params.get("model") or "").strip()
        if not SAFE_NAME.match(model_name):
            raise RunnerError("INVALID_PARAMS", "模型名只能包含中英文/数字/下划线/连字符，1-64 字符")
        version = str(params.get("version") or "v2").lower()
        if version not in {"v1", "v2"}:
            raise RunnerError("INVALID_PARAMS", "version 必须是 v1/v2")
        sample_rate = str(params.get("sample_rate") or "48000").lower()
        if sample_rate in {"32000", "32k"}:
            sample_rate = "32k"
        elif sample_rate in {"40000", "40k"}:
            sample_rate = "40k"
        elif sample_rate in {"48000", "48k"}:
            sample_rate = "48k"
        else:
            raise RunnerError("INVALID_PARAMS", "sample_rate 必须是 32k/40k/48k")
        if version == "v2" and sample_rate == "40k":
            raise RunnerError("INVALID_PARAMS", "当前 runtime 没有 v2 40k 配置；40k 请使用 v1，或改用 48k v2")
        if_f0 = int(params.get("if_f0", 1))
        if if_f0 not in {0, 1}:
            raise RunnerError("INVALID_PARAMS", "if_f0 必须是 0/1")
        f0_method = str(params.get("f0_method") or "rmvpe").lower()
        if f0_method not in {"rmvpe", "harvest", "pm", "crepe"}:
            raise RunnerError("INVALID_PARAMS", f"不支持的 f0_method：{f0_method}")
        epochs = int(params.get("epochs", 100))
        if not 1 <= epochs <= 20000:
            raise RunnerError("INVALID_PARAMS", "epochs 必须在 1..20000")
        batch_size = int(params.get("batch_size", 4))
        if not 1 <= batch_size <= 64:
            raise RunnerError("INVALID_PARAMS", "batch_size 必须在 1..64")
        save_every = int(params.get("save_every", 10))
        if not 1 <= save_every <= epochs:
            raise RunnerError("INVALID_PARAMS", "save_every 必须在 1..epochs")
        gpu = str(params.get("gpu") or "0")
        if not re.match(r"^[0-9]+(\-[0-9]+)*$", gpu):
            raise RunnerError("INVALID_PARAMS", "gpu 必须是 0 或 0-1 形式")
        return {
            "dataset_dir": str(dataset), "model_name": model_name, "version": version,
            "sample_rate": sample_rate, "if_f0": if_f0, "f0_method": f0_method,
            "epochs": epochs, "batch_size": batch_size, "save_every": save_every, "gpu": gpu,
        }

    def build_command(self, cfg: BackendConfig, params: dict[str, Any], ctx: JobContext) -> tuple[list[str], Path | None, dict[str, str] | None]:
        python, root = cfg.require_rvc()
        script = self.resolved_script_path()
        if not script.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"训练脚本不存在：{script}")
        voice_dir = Path(os.environ.get("AMS_HOME", "")).expanduser() / "voices"
        command = [
            str(python), "-B", str(script),
            "--rvc-root", str(root),
            "--dataset-dir", params["dataset_dir"],
            "--exp-name", "ams_train_" + str(ctx.job_id),
            "--model-name", params["model_name"],
            "--version", params["version"],
            "--sample-rate", params["sample_rate"],
            "--if-f0", str(params["if_f0"]),
            "--f0-method", params["f0_method"],
            "--epochs", str(params["epochs"]),
            "--batch-size", str(params["batch_size"]),
            "--save-every", str(params["save_every"]),
            "--gpu", params["gpu"],
            "--voice-dir", str(voice_dir),
        ]
        return command, root, None
