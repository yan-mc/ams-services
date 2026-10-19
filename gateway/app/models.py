"""API 请求/响应模型（pydantic v2）。

响应对象字段对齐 00 第 8 节 Job 对象定义；列表接口使用 {"ok": true, "data": [...], "total": n}。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "stopping", "succeeded", "failed", "cancelled"]


class JobCreateRequest(BaseModel):
    """POST /api/v1/jobs 请求体。"""

    type: str = Field(..., min_length=1, max_length=64, description="任务类型，如 ping / mock.long")
    params: dict[str, Any] = Field(default_factory=dict, description="任务参数（按任务类型定义）")
    project_id: str | None = Field(default=None, max_length=128, description="关联工程 ID（可选）")
    agent_session_id: str | None = Field(
        default=None, max_length=128, description="Agent 会话 ID（可选，用于审计）"
    )


class JobOut(BaseModel):
    """Job 对象（00 第 8 节）。"""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    status: JobStatus
    progress: float = 0.0
    stage: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_session_id: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: dict[str, Any] | None = None
    engine: dict[str, Any] = Field(default_factory=dict)
    gpu_lock: dict[str, Any] | None = None
    cancel_requested: bool = False
    resumable: bool = False


class JobListResponse(BaseModel):
    ok: bool = True
    data: list[JobOut]
    total: int
    limit: int
    offset: int


class ArtifactOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    job_id: str
    kind: str
    name: str
    path: str
    bytes: int = 0
    sha256: str | None = None
    created_at: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ArtifactListResponse(BaseModel):
    ok: bool = True
    data: list[ArtifactOut]


class VoiceScanRequest(BaseModel):
    """POST /api/v1/voices/scan 请求体；path 为空时扫描默认 voices 目录。"""

    path: str | None = Field(default=None, description="要扫描的 voices 根目录（可选，默认 AMS_HOME/voices）")


class VoiceScanResponse(BaseModel):
    ok: bool = True
    data: dict[str, Any]
