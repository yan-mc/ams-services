"""健康检查与鉴权测试。"""

from __future__ import annotations


def test_health_requires_token(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 401
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "AUTH_REQUIRED"


def test_health_rejects_wrong_token(client):
    response = client.get("/api/v1/health", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_INVALID"


def test_health_ok(client, auth, app):
    response = client.get("/api/v1/health", headers=auth)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["version"] == app.state.settings.version
    assert body["auth"]["scheme"] == "bearer"
    assert body["db"]["exists"] is True
    assert body["voices"]["exists"] is True
    types = {item["type"] for item in body["job_types"]}
    assert {"ping", "mock.long"} <= types
    gpu_types = {item["type"] for item in body["job_types"] if item["gpu"]}
    assert "mock.long" in gpu_types
    assert body["workers"]["gpu"] == 1


def test_token_file_created(app):
    token_file = app.state.settings.token_file
    assert token_file.is_file()
    token = token_file.read_text(encoding="utf-8").strip()
    assert len(token) >= 32
    assert token == app.state.token
