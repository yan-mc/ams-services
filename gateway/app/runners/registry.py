"""后端 Runner 注册（main.create_app 调用；测试可注入自定义脚本）。"""

from __future__ import annotations

from pathlib import Path

from ..jobs.runner import RunnerRegistry
from .msvc import MsvcConvertRunner
from .pymss import PymssSeparateRunner
from .rvc import RvcConvertRunner

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
    return registry
