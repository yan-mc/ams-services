"""统一错误模型与异常处理器。

错误响应格式（00 第 8 节）：
    {"ok": false, "error": {"code": "...", "message": "...", "detail": ...?}}
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("ams.gateway.errors")


class ApiError(Exception):
    """业务错误：由路由/服务层抛出，统一转成 JSON 错误响应。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        self.headers = headers or {}


def error_body(code: str, message: str, detail: object | None = None) -> dict:
    body: dict = {"ok": False, "error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器（业务错误 / 参数校验 / 兜底 500）。"""

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("api_error path=%s code=%s msg=%s", request.url.path, exc.code, exc.message)
        else:
            logger.info("api_error path=%s code=%s msg=%s", request.url.path, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = []
        for item in exc.errors():
            detail.append(
                {
                    "loc": list(item.get("loc", [])),
                    "msg": item.get("msg", ""),
                    "type": item.get("type", ""),
                }
            )
        logger.info("validation_error path=%s errors=%s", request.url.path, detail)
        return JSONResponse(
            status_code=422,
            content=error_body("INVALID_PARAMS", "请求参数校验失败", detail),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        incident = uuid.uuid4().hex[:12]
        logger.exception("unhandled_error incident=%s path=%s", incident, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(
                "INTERNAL_ERROR",
                "Gateway 内部错误，请查看日志",
                {"incident": incident},
            ),
        )
