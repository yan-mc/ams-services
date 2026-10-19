"""真实后端 Runner（M1）。

三个外部进程后端：
- convert.msvc：RVC runtime + ams-tools/msvc2pth.py（.msvc -> RVC v2 .pth）
- convert.rvc：RVC runtime + app/runners/scripts/rvc_convert.py（换声）
- separate.pymss：pymss 独立环境 + app/runners/scripts/pymss_separate.py（分离）

路径/解释器全部来自环境变量（见 config.py），未配置时任务失败并返回
BACKEND_NOT_CONFIGURED（附带可操作 hint），不静默降级。

注册入口：register_backend_runners(registry)；main.create_app 会调用。
"""

from __future__ import annotations

from .config import BackendConfig, RunnerError
from .registry import BACKEND_JOB_TYPES, register_backend_runners

__all__ = ["BackendConfig", "RunnerError", "BACKEND_JOB_TYPES", "register_backend_runners"]
