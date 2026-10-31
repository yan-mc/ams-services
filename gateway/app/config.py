"""Gateway 配置。

配置优先级：显式参数 > 环境变量 > 开发默认值。
所有路径都来自 AMS_HOME（见 00 第 4 节），本模块只做解析与目录准备，不写业务数据。

注意（MVP 开发默认）：若未设置 AMS_HOME，默认落在 gateway/.ams_home_dev，
避免误写系统 %LOCALAPPDATA%/AMS；正式运行时由启动器显式注入 AMS_HOME。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def default_dev_home() -> Path:
    """未配置 AMS_HOME 时的开发默认目录（gateway 目录内，已被 .gitignore 忽略）。"""
    return Path(__file__).resolve().parents[1] / ".ams_home_dev"


@dataclass(slots=True)
class Settings:
    """Gateway 运行配置（只含轻量标量/路径）。"""

    ams_home: Path
    host: str = "127.0.0.1"
    port: int = 0
    cpu_concurrency: int = 2
    log_level: str = "INFO"
    allow_non_loopback: bool = False
    rvc_python: str | None = None
    event_ring: int = 500
    shutdown_grace_seconds: float = 2.0
    version: str = "2.0.0"

    # ---- 派生路径（00 第 4 节目录约定） ----
    @property
    def config_dir(self) -> Path:
        return self.ams_home / "config"

    @property
    def runtime_dir(self) -> Path:
        return self.ams_home / "runtime"

    @property
    def logs_dir(self) -> Path:
        return self.ams_home / "logs" / "gateway"

    @property
    def voices_dir(self) -> Path:
        return self.ams_home / "voices"

    @property
    def artifacts_dir(self) -> Path:
        return self.ams_home / "artifacts"

    # ---- GPT-SoVITS（新 TTS 管线） ----
    @property
    def tts_gsv_home(self) -> Path:
        return self.ams_home / "tts-gsv"

    @property
    def tts_gsv_voices_dir(self) -> Path:
        return self.tts_gsv_home / "voices"

    @property
    def projects_dir(self) -> Path:
        return self.ams_home / "projects"

    @property
    def imports_dir(self) -> Path:
        return self.projects_dir / "imports"

    @property
    def tmp_dir(self) -> Path:
        return self.ams_home / "tmp"

    @property
    def db_path(self) -> Path:
        return self.config_dir / "state.db"

    @property
    def token_file(self) -> Path:
        return self.runtime_dir / "gateway.token"

    @property
    def ports_file(self) -> Path:
        return self.runtime_dir / "ports.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.ams_home,
            self.config_dir,
            self.runtime_dir,
            self.logs_dir,
            self.voices_dir,
            self.tts_gsv_home,
            self.tts_gsv_voices_dir,
            self.tts_gsv_home / "logs",
            self.tts_gsv_home / "tmp",
            self.artifacts_dir,
            self.projects_dir,
            self.imports_dir,
            self.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.port < 0 or self.port > 65535:
            raise ValueError(f"端口非法：{self.port}")
        if self.cpu_concurrency < 1:
            raise ValueError("cpu_concurrency 必须 >= 1")
        if self.host not in LOOPBACK_HOSTS and not self.allow_non_loopback:
            raise ValueError(
                f"Gateway 仅允许绑定回环地址，当前 host={self.host!r}；"
                "如确需非回环（不推荐），设置 AMS_GATEWAY_ALLOW_NON_LOOPBACK=1。"
            )

    @classmethod
    def from_env(
        cls,
        *,
        ams_home: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        log_level: str | None = None,
    ) -> "Settings":
        home_value = ams_home if ams_home is not None else _env("AMS_HOME")
        resolved_home = Path(home_value).expanduser() if home_value else default_dev_home()
        resolved_host = host if host is not None else (_env("AMS_GATEWAY_HOST", "127.0.0.1") or "127.0.0.1")
        resolved_port = port if port is not None else _env_int("AMS_GATEWAY_PORT", 0)
        settings = cls(
            ams_home=resolved_home.resolve(),
            host=resolved_host,
            port=resolved_port,
            cpu_concurrency=_env_int("AMS_GATEWAY_CPU_CONCURRENCY", 2),
            log_level=(log_level or _env("AMS_GATEWAY_LOG_LEVEL", "INFO") or "INFO").upper(),
            allow_non_loopback=_env_bool("AMS_GATEWAY_ALLOW_NON_LOOPBACK", False),
            rvc_python=_env("AMS_RVC_PYTHON"),
            event_ring=_env_int("AMS_GATEWAY_EVENT_RING", 500),
            shutdown_grace_seconds=float(_env("AMS_GATEWAY_SHUTDOWN_GRACE", "2.0") or "2.0"),
        )
        settings.validate()
        return settings
