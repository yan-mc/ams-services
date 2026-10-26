"""后端 Runner 注册（main.create_app 调用；测试可注入自定义脚本）。"""

from __future__ import annotations

from pathlib import Path

from ..jobs.runner import RunnerRegistry
from .msvc import MsvcConvertRunner
from .pymss import PymssSeparateRunner
from .rvc import RvcConvertRunner
from .tts import TtsSynthesizeRunner
from .train import RvcTrainRunner
from .tts_train import TtsTrainRunner
from .tts_dataset import TtsDatasetPrepareRunner
from .dataset import DatasetPrepareRunner
from .mix_evaluate import EvaluateAudioRunner, MixMasterRunner

BACKEND_JOB_TYPES = (
    {
        "type": "convert.msvc",
        "gpu": False,
        "description": "MSVC 容器转 RVC v2 pth + conversion.json（RVC runtime + msvc2pth）",
    },
    {
        "type": "convert.rvc",
        "gpu": True,
        "description": "RVC 换声（RVC runtime + scripts/rvc_convert.py）",
    },
    {
        "type": "separate.pymss",
        "gpu": True,
        "description": "pymss 分离（pymss 独立环境 + scripts/pymss_separate.py）",
    },
    {
        "type": "train.rvc",
        "gpu": True,
        "description": "RVC 音源炼制（数据集 → pth + index → 注册音色）",
    },
    {
        "type": "train.tts",
        "gpu": True,
        "description": "TTS 音源炼制（数据集 → onnx + onnx.json → 注册）",
    },
    {
        "type": "tts.dataset.prepare",
        "gpu": False,
        "description": "TTS 数据集导入/校验（metadata.csv + wav）",
    },
    {
        "type": "tts.synthesize",
        "gpu": False,
        "description": "Piper 离线语音合成（runtime.tts）",
    },
    {
        "type": "dataset.prepare",
        "gpu": False,
        "description": "数据集导入/切片/校验",
    },
    {
        "type": "evaluate.audio",
        "gpu": False,
        "description": "音频客观评测",
    },
    {
        "type": "mix.master",
        "gpu": False,
        "description": "独立多轨混音/母带",
    },
)


def _description(job_type: str) -> str:
    """按 type 取描述，避免用数组下标（下标漂移曾导致描述与类型错位）。"""
    for item in BACKEND_JOB_TYPES:
        if item["type"] == job_type:
            return item["description"]
    raise KeyError(job_type)


def register_backend_runners(
    registry: RunnerRegistry,
    *,
    rvc_script: str | Path | None = None,
    pymss_script: str | Path | None = None,
) -> RunnerRegistry:
    """把所有真实后端注册进 registry（可覆盖脚本路径用于测试/实验）。"""
    registry.register(
        "convert.msvc",
        lambda: MsvcConvertRunner(),
        gpu=False,
        description=_description("convert.msvc"),
    )
    registry.register(
        "convert.rvc",
        lambda: RvcConvertRunner(script_path=rvc_script),
        gpu=True,
        description=_description("convert.rvc"),
    )
    registry.register(
        "separate.pymss",
        lambda: PymssSeparateRunner(script_path=pymss_script),
        gpu=True,
        description=_description("separate.pymss"),
    )
    registry.register(
        "train.rvc",
        lambda: RvcTrainRunner(),
        gpu=True,
        description=_description("train.rvc"),
    )
    registry.register(
        "train.tts",
        lambda: TtsTrainRunner(),
        gpu=True,
        description=_description("train.tts"),
    )
    registry.register(
        "tts.dataset.prepare",
        lambda: TtsDatasetPrepareRunner(),
        gpu=False,
        description=_description("tts.dataset.prepare"),
    )
    registry.register(
        "tts.synthesize",
        lambda: TtsSynthesizeRunner(),
        gpu=False,
        description=_description("tts.synthesize"),
    )
    registry.register(
        "dataset.prepare",
        lambda: DatasetPrepareRunner(),
        gpu=False,
        description=_description("dataset.prepare"),
    )
    registry.register(
        "evaluate.audio",
        lambda: EvaluateAudioRunner(),
        gpu=False,
        description=_description("evaluate.audio"),
    )
    registry.register(
        "mix.master",
        lambda: MixMasterRunner(),
        gpu=False,
        description=_description("mix.master"),
    )
    return registry