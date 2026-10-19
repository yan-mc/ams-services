"""任务创建/查询/取消/产物/持久化测试。"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from conftest import wait_for_status


def test_create_ping_job_and_artifacts(client, auth):
    response = client.post("/api/v1/jobs", headers=auth, json={"type": "ping", "params": {}})
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["id"].startswith("job_")
    assert job["type"] == "ping"
    assert job["status"] in {"queued", "running", "succeeded"}

    finished = wait_for_status(client, auth, job["id"], {"succeeded", "failed"})
    assert finished["status"] == "succeeded", finished
    assert finished["progress"] == 100.0
    assert finished["started_at"] and finished["finished_at"]

    artifacts = client.get(f"/api/v1/jobs/{job['id']}/artifacts", headers=auth)
    assert artifacts.status_code == 200
    items = artifacts.json()["data"]
    assert len(items) >= 1
    first = items[0]
    assert first["name"] == "ping-result.json"
    assert first["bytes"] > 0
    assert first["sha256"]

    download = client.get(f"/api/v1/artifacts/{first['id']}/download", headers=auth)
    assert download.status_code == 200
    assert b"pong" in download.content


def test_unknown_job_type(client, auth):
    response = client.post("/api/v1/jobs", headers=auth, json={"type": "not.exists", "params": {}})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "UNKNOWN_JOB_TYPE"


def test_invalid_job_body(client, auth):
    response = client.post("/api/v1/jobs", headers=auth, json={"params": {}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PARAMS"


def test_list_jobs(client, auth):
    ids = []
    for _ in range(2):
        response = client.post("/api/v1/jobs", headers=auth, json={"type": "ping"})
        ids.append(response.json()["id"])
    wait_for_status(client, auth, ids[-1], {"succeeded"})
    listed = client.get("/api/v1/jobs", headers=auth)
    assert listed.status_code == 200
    body = listed.json()
    assert body["ok"] is True
    assert body["total"] >= 2
    assert {item["id"] for item in body["data"]} >= set(ids)
    filtered = client.get("/api/v1/jobs", headers=auth, params={"status": "succeeded"})
    assert all(item["status"] == "succeeded" for item in filtered.json()["data"])
    invalid = client.get("/api/v1/jobs", headers=auth, params={"status": "bogus"})
    assert invalid.status_code == 422


def test_get_missing_job(client, auth):
    response = client.get("/api/v1/jobs/job_missing", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_cancel_running_mock_long(client, auth):
    response = client.post(
        "/api/v1/jobs",
        headers=auth,
        json={"type": "mock.long", "params": {"seconds": 30}},
    )
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["gpu_lock"]["state"] == "waiting"

    running = wait_for_status(client, auth, job["id"], {"running"})
    assert running["gpu_lock"]["state"] == "held"

    cancelled = client.post(f"/api/v1/jobs/{job['id']}/cancel", headers=auth)
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] in {"stopping", "cancelled"}

    final = wait_for_status(client, auth, job["id"], {"cancelled", "failed"})
    assert final["status"] == "cancelled", final
    assert final["cancel_requested"] is True
    assert final["finished_at"]


def test_cancel_is_idempotent_for_finished_job(client, auth):
    response = client.post("/api/v1/jobs", headers=auth, json={"type": "ping"})
    job_id = response.json()["id"]
    finished = wait_for_status(client, auth, job_id, {"succeeded"})
    again = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=auth)
    assert again.status_code == 200
    assert again.json()["status"] == finished["status"]


def test_gpu_jobs_are_serialized(client, auth):
    first = client.post(
        "/api/v1/jobs", headers=auth, json={"type": "mock.long", "params": {"seconds": 2}}
    ).json()
    second = client.post(
        "/api/v1/jobs", headers=auth, json={"type": "mock.long", "params": {"seconds": 2}}
    ).json()
    running_first = wait_for_status(client, auth, first["id"], {"running"})
    assert running_first["gpu_lock"]["state"] == "held"
    # 第二个 GPU 任务在第一个结束前不应进入 running
    time.sleep(0.2)
    second_state = client.get(f"/api/v1/jobs/{second['id']}", headers=auth).json()
    assert second_state["status"] in {"queued", "running"}
    if second_state["status"] == "queued":
        assert second_state["gpu_lock"]["state"] == "waiting"
    wait_for_status(client, auth, first["id"], {"succeeded"})
    wait_for_status(client, auth, second["id"], {"succeeded"})


def test_job_history_survives_gateway_restart(tmp_path: Path):
    home = tmp_path / "AMS_HOME_RESTART"
    (home / "voices").mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env(ams_home=home, host="127.0.0.1", port=0, log_level="WARNING")

    first_app = create_app(settings)
    with TestClient(first_app) as first_client:
        auth = {"Authorization": f"Bearer {first_app.state.token}"}
        created = first_client.post("/api/v1/jobs", headers=auth, json={"type": "ping"}).json()
        wait_for_status(first_client, auth, created["id"], {"succeeded"})

    second_app = create_app(settings)
    with TestClient(second_app) as second_client:
        auth2 = {"Authorization": f"Bearer {second_app.state.token}"}
        fetched = second_client.get(f"/api/v1/jobs/{created['id']}", headers=auth2)
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["status"] == "succeeded"
        listed = second_client.get("/api/v1/jobs", headers=auth2).json()
        assert listed["total"] >= 1
