"""SSE 事件流测试（历史回放 + 终态关闭）。"""

from __future__ import annotations

from conftest import wait_for_status


def _collect_sse(client, auth, job_id: str, timeout: float = 10.0) -> list[tuple[str, str]]:
    frames: list[tuple[str, str]] = []
    event_kind = ""
    data_lines: list[str] = []
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events", headers=auth) as response:
        assert response.status_code == 200, response.read()
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("text/event-stream"), content_type
        for line in response.iter_lines():
            if line == "":
                if event_kind or data_lines:
                    frames.append((event_kind, "\n".join(data_lines)))
                event_kind = ""
                data_lines = []
                if any(kind == "status" and '"succeeded"' in data or kind == "status" and '"cancelled"' in data for kind, data in frames[-1:]):
                    break
                continue
            if line.startswith("event:"):
                event_kind = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
    return frames


def test_sse_replays_history_for_finished_job(client, auth):
    created = client.post("/api/v1/jobs", headers=auth, json={"type": "ping"}).json()
    wait_for_status(client, auth, created["id"], {"succeeded"})
    frames = _collect_sse(client, auth, created["id"])
    kinds = [kind for kind, _ in frames]
    assert frames, "SSE 至少应回放一条事件"
    assert "status" in kinds
    status_payloads = [data for kind, data in frames if kind == "status"]
    assert any('"succeeded"' in data for data in status_payloads)
    assert any(kind in {"progress", "log"} for kind in kinds)


def test_sse_cancel_emits_cancelled_status(client, auth):
    created = client.post(
        "/api/v1/jobs", headers=auth, json={"type": "mock.long", "params": {"seconds": 30}}
    ).json()
    wait_for_status(client, auth, created["id"], {"running"})
    client.post(f"/api/v1/jobs/{created['id']}/cancel", headers=auth)
    wait_for_status(client, auth, created["id"], {"cancelled"})
    frames = _collect_sse(client, auth, created["id"])
    status_payloads = [data for kind, data in frames if kind == "status"]
    assert any('"cancelled"' in data for data in status_payloads)


def test_sse_missing_job_returns_404(client, auth):
    response = client.get("/api/v1/jobs/job_missing/events", headers=auth)
    assert response.status_code == 404
