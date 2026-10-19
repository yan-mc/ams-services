"""separate.pymss：调用 pymss 独立环境 + scripts/pymss_separate.py。"""

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


class PymssSeparateRunner(ProtocolProcessRunner):
    job_type = "separate.pymss"
    gpu = True
    timeout_sec = 3600.0
    description = "pymss 音乐源分离"

    def default_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "scripts" / "pymss_separate.py"

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        input_value = _require_str(params, "input_path", "输入音频路径")
        input_path = Path(input_value).expanduser()
        if not input_path.is_file():
            raise RunnerError(
                "INPUT_NOT_FOUND",
                f"输入音频不存在：{input_path}",
                hint="检查 input_path；首次使用某模型需先下载到 pymss models 目录",
            )
        model = _require_str(params, "model", "pymss 模型名，如 scnet_checkpoint_musdb18")
        device = str(params.get("device") or "auto")
        if device not in {"auto", "cpu", "cuda", "cuda:0", "cuda:1"}:
            raise RunnerError(
                "INVALID_PARAMS",
                f"不支持的 device：{device}",
                hint="可选 auto / cpu / cuda / cuda:0（GPU 需 pymss 环境安装 cu128 torch）",
            )
        chunk_value = params.get("chunk")
        chunk: int | None = None
        if chunk_value is not None:
            chunk = int(chunk_value)
            if chunk < 8000 or chunk > 2000000:
                raise RunnerError("INVALID_PARAMS", "chunk 必须在 8000..2000000 采样点之间")
        return {"input_path": str(input_path), "model": model, "device": device, "chunk": chunk}

    def build_command(
        self,
        cfg: BackendConfig,
        params: dict[str, Any],
        ctx: JobContext,
    ) -> tuple[list[str], Path | None, dict[str, str] | None]:
        python, root = cfg.require_pymss()
        script = self.resolved_script_path()
        if not script.is_file():
            raise RunnerError(
                "BACKEND_PATH_NOT_FOUND",
                f"pymss 分离脚本不存在：{script}",
                hint="gateway 安装包缺少 scripts/pymss_separate.py，请重新安装/更新 gateway 组件",
            )
        model_dir = root / "models"
        output_dir = ctx.work_dir / "stems"
        device = params["device"]
        if device.startswith("cuda"):
            device = "cuda"
        command = [
            str(python),
            "-B",
            str(script),
            "--input",
            params["input_path"],
            "--model",
            params["model"],
            "--device",
            device,
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
        ]
        if params.get("chunk"):
            command.extend(["--chunk", str(params["chunk"])])
        return command, ctx.work_dir, {"PYMSS_MODEL_DIR": str(model_dir)}