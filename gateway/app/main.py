"""Gateway 启动入口（M0.3）。

用法：
    .venv\\Scripts\\python.exe -m app --ams-home F:\\tmp\\AMS --port 0
    ams-gateway --help

约定（00 第 3/8 节）：
- 只监听回环地址（127.0.0.1），端口默认 0（动态分配）；
- 实际端口写入 AMS_HOME/runtime/ports.json，token 写入 AMS_HOME/runtime/gateway.token；
- 所有 /api/v1 请求需要 Bearer token。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import health as health_api
from .api import jobs as jobs_api
from .api import projects as projects_api
from .api import tts as tts_api
from .api import voices as voices_api
from .config import Settings
from .db import Database
from .errors import register_exception_handlers
from .jobs.events import EventBus
from .jobs.runner import RunnerRegistry, build_default_registry
from .jobs.service import JobService
from .runners import register_backend_runners
from .logging_setup import setup_logging
from .security import load_or_create_token
from .voices.pth_parser import MetadataParser, build_parser

API_PREFIX = "/api/v1"
CORS_ORIGIN_REGEX = (
    r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|http://tauri\.localhost)$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_app(
    settings: Settings | None = None,
    *,
    token: str | None = None,
    registry: RunnerRegistry | None = None,
    voice_parser: MetadataParser | None = None,
) -> FastAPI:
    """构建 FastAPI 应用（工厂函数，测试可注入临时 AMS_HOME/token/parser）。"""
    settings = settings or Settings.from_env()
    settings.validate()
    settings.ensure_dirs()

    logger = setup_logging(settings.logs_dir, settings.log_level)
    resolved_token = token or load_or_create_token(settings.token_file)
    db = Database(settings.db_path)
    bus = EventBus()
    resolved_registry = registry or build_default_registry()
    if registry is None:
        # 默认 registry：内置 mock（ping/mock.long）+ 三个真实后端
        register_backend_runners(resolved_registry)
    service = JobService(db=db, bus=bus, registry=resolved_registry, settings=settings)
    resolved_parser = voice_parser or build_parser(settings.rvc_python)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.started_at = _utc_now()
        app.state.started_monotonic = time.monotonic()
        recovery = service.start()
        logger.info(
            "Gateway 就绪 version=%s host=%s port=%s ams_home=%s 恢复=%s",
            settings.version,
            settings.host,
            getattr(app.state, "port", settings.port),
            settings.ams_home,
            recovery,
        )
        try:
            yield
        finally:
            logger.info("Gateway 正在关闭 ...")
            await service.stop()
            db.close()

    app = FastAPI(
        title="AMS Gateway",
        description="AMS 本地任务/服务/产物网关（M0.3 MVP）",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.state.settings = settings
    app.state.token = resolved_token
    app.state.db = db
    app.state.bus = bus
    app.state.service = service
    app.state.registry = resolved_registry
    app.state.voice_parser = resolved_parser
    app.state.logger = logger
    app.state.started_at = None
    app.state.started_monotonic = time.monotonic()
    app.state.port = settings.port

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception(
                "http %s %s -> EXC %.1fms rid=%s",
                request.method,
                request.url.path,
                elapsed,
                request_id,
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "http %s %s -> %s %.1fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    register_exception_handlers(app)
    app.include_router(health_api.router, prefix=API_PREFIX)
    app.include_router(jobs_api.router, prefix=API_PREFIX)
    app.include_router(voices_api.router, prefix=API_PREFIX)
    app.include_router(projects_api.router, prefix=API_PREFIX)
    app.include_router(tts_api.router, prefix=API_PREFIX)
    return app


def _write_ports_file(settings: Settings, port: int) -> None:
    payload = {
        "schema": "ams.ports/1",
        "gateway": {
            "host": settings.host,
            "port": port,
            "pid": os.getpid(),
            "api_prefix": API_PREFIX,
            "token_file": str(settings.token_file),
            "started_at": _utc_now(),
        },
    }
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.ports_file.with_name(settings.ports_file.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, settings.ports_file)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ams-gateway", description="AMS Gateway（M0.3 MVP）")
    parser.add_argument("--ams-home", default=None, help="AMS_HOME 数据目录（默认读环境变量 AMS_HOME）")
    parser.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1（仅允许回环）")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 0=动态分配")
    parser.add_argument("--log-level", default=None, help="日志级别：DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--print-token", action="store_true", help="打印 token 文件路径（不打印 token 内容）")
    parser.add_argument("--version", action="version", version=f"ams-gateway {__version__}")
    return parser


def run(settings: Settings) -> None:
    """启动 uvicorn；先 bind socket 以获取动态端口并写 ports.json。"""
    import uvicorn

    app = create_app(settings)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((settings.host, settings.port))
    sock.listen(128)
    actual_port = int(sock.getsockname()[1])
    app.state.port = actual_port
    _write_ports_file(settings, actual_port)
    logger = app.state.logger
    logger.info("监听 http://%s:%s%s", settings.host, actual_port, API_PREFIX)
    logger.info("token 文件：%s", settings.token_file)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=actual_port,
        log_level=settings.log_level.lower(),
        lifespan="on",
        access_log=False,
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    settings = Settings.from_env(
        ams_home=args.ams_home,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    settings.ensure_dirs()
    token_path = settings.token_file
    if args.print_token:
        print(str(token_path))
    run(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
