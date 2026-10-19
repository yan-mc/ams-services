"""pytest 公共夹具：临时 AMS_HOME、FastAPI TestClient、Bearer token。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def ams_home(tmp_path: Path) -> Path:
    home = tmp_path / "AMS_HOME"
    (home / "voices").mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture()
def app(ams_home: Path):
    settings = Settings.from_env(
        ams_home=ams_home,
        host="127.0.0.1",
        port=0,
        log_level="WARNING",
    )
    return create_app(settings)


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth(app) -> dict[str, str]:
    return {"Authorization": f"Bearer {app.state.token}"}


def wait_for_status(
    client: TestClient,
    auth: dict[str, str],
    job_id: str,
    statuses: Iterable[str],
    timeout: float = 15.0,
) -> dict[str, Any]:
    """轮询任务直到进入目标状态之一。"""
    wanted = set(statuses)
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=auth)
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in wanted:
            return last
        time.sleep(0.05)
    raise AssertionError(f"等待任务状态超时：wanted={wanted} last={last}")


def write_file(path: Path, size: int = 1024, fill: bytes = b"\x00") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((fill or b"\x00") * size)
    return path
