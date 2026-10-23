"""工程素材导入 API：外部文件默认复制到 AMS_HOME/projects。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..paths import materialize_file
from ..security import require_token

router = APIRouter(tags=["projects"], dependencies=[Depends(require_token)])


class ProjectImportRequest(BaseModel):
    source_path: str = Field(..., min_length=1, description="外部素材路径")
    project_id: str | None = Field(default=None, max_length=128)


@router.post("/projects/import")
def import_project_asset(body: ProjectImportRequest, request: Request) -> dict:
    settings = request.app.state.settings
    target = materialize_file(
        body.source_path,
        settings=settings,
        project_id=body.project_id,
    )
    return {"ok": True, "data": {"path": target, "source_path": body.source_path}}