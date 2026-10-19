"""Bearer token 鉴权（本地回环场景）。

约定（00 第 3/8 节）：
- token 首次启动生成，存放 AMS_HOME/runtime/gateway.token（32+ 字节随机）；
- 所有 /api/v1 请求必须带 Authorization: Bearer <token>（包括 /health）；
- 比较使用常量时间，避免时序侧信道。
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from fastapi import Request

from .errors import ApiError

TOKEN_BYTES = 32
MIN_TOKEN_LENGTH = 32


def load_or_create_token(path: Path) -> str:
    """读取 token；不存在则安全生成并落盘（原子替换 + 尽量收紧权限）。"""
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(f"token 文件内容过短/损坏：{path}")
        return token

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
    finally:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows 下权限位语义有限，失败不影响功能
        pass
    return token


def token_matches(expected: str, provided: str) -> bool:
    if not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


async def require_token(request: Request) -> str:
    """FastAPI 依赖：校验 Bearer token。"""
    expected: str = request.app.state.token
    header = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    scheme, _, credentials = header.partition(" ")
    if not header or scheme.lower() != "bearer":
        raise ApiError(
            401,
            "AUTH_REQUIRED",
            "缺少 Bearer token（Authorization: Bearer <token>）",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not token_matches(expected, credentials.strip()):
        raise ApiError(
            401,
            "AUTH_INVALID",
            "Bearer token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.strip()
