"""后端 Runner 注册（main.create_app 调用；测试可注入自定义脚本）。"""

from __future__ import annotations

from pathlib import Path

from ..jobs.runner import RunnerRegistry
from .msvc import MsvcConvertRunner
from .pymss import PymssSeparateRunner
from .rvc import RvcConvertRunner
from .tts import TtsSynthesizeRunner
from .train import RvcTrainRunner
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


def register_backend_runners(
    registry: RunnerRegistry,
    *,
    rvc_script: str | Path | None = None,
    pymss_script: str | Path | None = None,
) -> RunnerRegistry:
    """把三个真实后端注册进 registry（可覆盖脚本路径用于测试/实验）。"""
    registry.register(
        "convert.msvc",
        lambda: MsvcConvertRunner(),
        gpu=False,
        description=BACKEND_JOB_TYPES[0]["description"],
    )
    registry.register(
        "convert.rvc",
        lambda: RvcConvertRunner(script_path=rvc_script),
        gpu=True,
        description=BACKEND_JOB_TYPES[1]["description"],
    )
    registry.register(
        "separate.pymss",
        lambda: PymssSeparateRunner(script_path=pymss_script),
        gpu=True,
        description=BACKEND_JOB_TYPES[2]["description"],
    )
    registry.register(
        "tts.synthesize",
        lambda: TtsSynthesizeRunner(),
        gpu=False,
        description=BACKEND_JOB_TYPES[4]["description"],
    )
    registry.register(
        "train.rvc",
        lambda: RvcTrainRunner(),
        gpu=True,
        description=BACKEND_JOB_TYPES[3]["description"],
    )
    registry.register(
        "dataset.prepare",
        lambda: DatasetPrepareRunner(),
        gpu=False,
        description=BACKEND_JOB_TYPES[5]["description"],
    )
    registry.register(
        "evaluate.audio",
        lambda: EvaluateAudioRunner(),
        gpu=False,
        description=BACKEND_JOB_TYPES[6]["description"],
    )
    registry.register(
        "mix.master",
        lambda: MixMasterRunner(),
        gpu=False,
        description=BACKEND_JOB_TYPES[7]["description"],
    )
    return registry
