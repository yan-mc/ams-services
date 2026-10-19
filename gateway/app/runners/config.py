"""真实后端配置与错误类型。

所有路径/解释器来自环境变量（不写死、不依赖系统 PATH）：

| 环境变量 | 用途 | 必需任务 |
|---|---|---|
| AMS_RVC_PYTHON | RVC 整合包自带 python.exe | convert.msvc / convert.rvc |
| AMS_RVC_ROOT | RVC 整合包根目录（含 assets/、ffmpeg.exe） | convert.rvc |
| AMS_PYMSS_PYTHON | pymss 独立环境 python.exe | separate.pymss |
| AMS_PYMSS_ROOT | pymss 环境根目录（模型目录为 <root>/models） | separate.pymss |
| AMS_MSVC_TOOL | msvc2pth.py 路径 | convert.msvc |

任务失败时统一返回 RunnerError：code 供 Gateway 写入 Job.error.code，
hint 给人/Agent 可操作建议（README 有说明）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_RVC_PYTHON = "AMS_RVC_PYTHON"
ENV_RVC_ROOT = "AMS_RVC_ROOT"
ENV_PYMSS_PYTHON = "AMS_PYMSS_PYTHON"
ENV_PYMSS_ROOT = "AMS_PYMSS_ROOT"
ENV_MSVC_TOOL = "AMS_MSVC_TOOL"


class RunnerError(Exception):
    """后端可预期错误：code 会写入 Job.error.code，hint 供排障。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: Any = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.hint = hint


@dataclass(slots=True)
class BackendConfig:
    """从环境变量解析出的后端路径（读一次，用于单次任务）。"""

    rvc_python: str | None = None
    rvc_root: str | None = None
    pymss_python: str | None = None
    pymss_root: str | None = None
    msvc_tool: str | None = None

    @classmethod
    def from_env(cls) -> "BackendConfig":
        return cls(
            rvc_python=os.environ.get(ENV_RVC_PYTHON),
            rvc_root=os.environ.get(ENV_RVC_ROOT),
            pymss_python=os.environ.get(ENV_PYMSS_PYTHON),
            pymss_root=os.environ.get(ENV_PYMSS_ROOT),
            msvc_tool=os.environ.get(ENV_MSVC_TOOL),
        )

    # ------------------------------------------------------------------ 通用解析
    @staticmethod
    def _require_value(env_name: str, value: str | None, label: str, hint: str) -> Path:
        if not value:
            raise RunnerError(
                "BACKEND_NOT_CONFIGURED",
                f"未配置 {label}（环境变量 {env_name} 为空）",
                hint=hint,
            )
        return Path(value).expanduser()

    @classmethod
    def _require_file(cls, env_name: str, value: str | None, label: str, hint: str) -> Path:
        path = cls._require_value(env_name, value, label, hint)
        if not path.is_file():
            raise RunnerError(
                "BACKEND_PATH_NOT_FOUND",
                f"{label} 路径不存在或不是文件：{path}",
                detail={"env": env_name, "path": str(path)},
                hint=hint,
            )
        return path

    @classmethod
    def _require_dir(cls, env_name: str, value: str | None, label: str, hint: str) -> Path:
        path = cls._require_value(env_name, value, label, hint)
        if not path.is_dir():
            raise RunnerError(
                "BACKEND_PATH_NOT_FOUND",
                f"{label} 路径不存在或不是目录：{path}",
                detail={"env": env_name, "path": str(path)},
                hint=hint,
            )
        return path

    # ------------------------------------------------------------------ 组合
    def require_msvc(self) -> tuple[Path, Path]:
        """返回 (rvc_python, msvc_tool)。"""
        python = self._require_file(
            ENV_RVC_PYTHON,
            self.rvc_python,
            "RVC runtime Python",
            "在启动器/环境设置中配置 AMS_RVC_PYTHON（RVC 整合包 runtime\\python.exe）",
        )
        tool = self._require_file(
            ENV_MSVC_TOOL,
            self.msvc_tool,
            "msvc2pth 工具",
            "配置 AMS_MSVC_TOOL 指向 ams-tools/msvc2pth/msvc2pth.py",
        )
        return python, tool

    def require_rvc(self) -> tuple[Path, Path]:
        """返回 (rvc_python, rvc_root)。"""
        python = self._require_file(
            ENV_RVC_PYTHON,
            self.rvc_python,
            "RVC runtime Python",
            "配置 AMS_RVC_PYTHON（RVC 整合包 runtime\\python.exe）",
        )
        root = self._require_dir(
            ENV_RVC_ROOT,
            self.rvc_root,
            "RVC 整合包目录",
            "配置 AMS_RVC_ROOT（含 assets/weights、assets/hubert、ffmpeg.exe 的 RVC 根目录）",
        )
        return python, root

    def require_pymss(self) -> tuple[Path, Path]:
        """返回 (pymss_python, pymss_root)；模型目录 = root/models。"""
        python = self._require_file(
            ENV_PYMSS_PYTHON,
            self.pymss_python,
            "pymss 环境 Python",
            "配置 AMS_PYMSS_PYTHON（pymss 独立 venv 的 python.exe）",
        )
        root = self._require_dir(
            ENV_PYMSS_ROOT,
            self.pymss_root,
            "pymss 环境根目录",
            "配置 AMS_PYMSS_ROOT（含 models/ 子目录的 pymss 实验/服务目录）",
        )
        return python, root
