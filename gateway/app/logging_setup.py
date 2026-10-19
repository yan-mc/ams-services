"""日志初始化：控制台 + 滚动文件（AMS_HOME/logs/gateway）。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(logs_dir: Path, level: str = "INFO") -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ams.gateway")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # 避免重复初始化（例如测试里多次 create_app）
    if logger.handlers:
        return logger

    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        logs_dir / "gateway.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger
