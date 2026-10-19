"""真实后端 Runner 测试（fake backend，无需真实模型/显存）。

覆盖：convert.msvc / convert.rvc / separate.pymss 的成功、失败、取消、
未配置、参数校验；App 层注册与一次端到端 Job。
真实后端 smoke 可通过 AMS_TEST_REAL_BACKENDS=1 手动开启（默认 skip）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.jobs.runner import JobCancelled, JobContext, build_default_registry
from app.main import create_app
from app.runners.config import RunnerError
from app.runners.msvc import MsvcConvertRunner
from app.runners.pymss import PymssSeparateRunner
from app.runners.registry import BACKEND_JOB_TYPES, register_backend_runners
from app.runners.rvc import RvcConvertRunner

from conftest import wait_for_status

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PROTOCOL = TESTS_DIR / "fake_protocol_backend.py"
FAKE_MSVC = TESTS_DIR / "fake_msvc2pth.py"


# --------------------------------------------------------------------- helpers
def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_info = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_info, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == still_active
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _configure_backend_env(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    rvc_root = tmp_path / "rvc_root"
    (rvc_root / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (rvc_root / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    pymss_root = tmp_path / "pymss_root"
    (pymss_root / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AMS_RVC_PYTHON", sys.executable)
    monkeypatch.setenv("AMS_RVC_ROOT", str(rvc_root))
    monkeypatch.setenv("AMS_PYMSS_PYTHON", sys.executable)
    monkeypatch.setenv("AMS_PYMSS_ROOT", str(pymss_root))
    monkeypatch.setenv("AMS_MSVC_TOOL", str(FAKE_MSVC))
    return rvc_root, pymss_root


def _input_audio(tmp_path: Path) -> Path:
    path = tmp_path / "input.wav"
    path.write_bytes(b"RIFF....fake")
    return path


def _make_ctx(runner, params: dict, work_dir: Path) -> tuple[JobContext, list, list]:
    progress_events: list[tuple[float, str, str]] = []
    logs: list[tuple[str, str]] = []

    async def on_progress(percent, stage, message=None):
        progress_events.append((float(percent), str(stage), str(message or "")))

    def on_log(level, message):
        logs.append((str(level), str(message)))

    ctx = JobContext(
        job_id="job_fake",
        job_type=runner.job_type,
        params=params,
        work_dir=work_dir,
        progress_cb=on_progress,
        log_cb=on_log,
    )
    return ctx, progress_events, logs


def _run_runner(runner, params: dict, work_dir: Path):
    ctx, progress_events, logs = _make_ctx(runner, params, work_dir)
    artifacts = asyncio.run(runner.run(ctx))
    return artifacts, progress_events, logs


# --------------------------------------------------------------------- convert.msvc
def test_msvc_runner_success(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    input_file = tmp_path / "voice.msvc"
    input_file.write_bytes(b"FAKE_MSVC_CONTAINER")
    out_dir = tmp_path / "converted"
    runner = MsvcConvertRunner()
    artifacts, progress_events, logs = _run_runner(
        runner,
        {"input_path": str(input_file), "out_dir": str(out_dir), "name": "fake_voice"},
        tmp_path / "work",
    )
    names = {item.name for item in artifacts}
    assert "fake_voice.pth" in names
    assert "conversion.json" in names
    for item in artifacts:
        assert Path(item.path).is_file(), item
    assert progress_events, "应上报进度"
    assert progress_events[-1][0] == 100.0
    assert any("[calibrate]" in message for _level, message in logs) or any(
        "fake" in message for _level, message in logs
    )


def test_msvc_runner_failure(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_MSVC_MODE", "fail")
    input_file = tmp_path / "broken.msvc"
    input_file.write_bytes(b"x")
    runner = MsvcConvertRunner()
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            runner,
            {"input_path": str(input_file), "out_dir": str(tmp_path / "out")},
            tmp_path / "work",
        )
    assert exc_info.value.code == "BACKEND_FAILED"
    assert exc_info.value.hint


# --------------------------------------------------------------------- convert.rvc
def test_rvc_runner_success(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    runner = RvcConvertRunner(script_path=FAKE_PROTOCOL)
    artifacts, progress_events, _logs = _run_runner(
        runner,
        {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo", "f0_up_key": 0},
        tmp_path / "work",
    )
    assert artifacts and all(Path(item.path).is_file() for item in artifacts)
    assert len(progress_events) >= 3


def test_rvc_runner_failure(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_PROTOCOL_MODE", "fail")
    runner = RvcConvertRunner(script_path=FAKE_PROTOCOL)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            runner,
            {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"},
            tmp_path / "work",
        )
    assert exc_info.value.code == "FAKE_BACKEND_FAILED"


def test_rvc_runner_cancel_kills_process_tree(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_PROTOCOL_MODE", "cancel")
    runner = RvcConvertRunner(script_path=FAKE_PROTOCOL)
    params = {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"}
    work_dir = tmp_path / "work"
    ctx, _progress, _logs = _make_ctx(runner, params, work_dir)

    async def scenario() -> int:
        task = asyncio.create_task(runner.run(ctx))
        pid = 0
        pid_file = work_dir / "fake.pid"
        for _ in range(200):
            if pid_file.is_file() and pid_file.read_text(encoding="utf-8").strip():
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                break
            await asyncio.sleep(0.02)
        assert pid, "fake 后端未写出 pid 文件"
        ctx.request_cancel()
        with pytest.raises(JobCancelled):
            await asyncio.wait_for(task, timeout=20)
        return pid

    pid = asyncio.run(scenario())
    time.sleep(1.0)
    assert not _pid_alive(pid), "取消后子进程树应被杀死"


# --------------------------------------------------------------------- separate.pymss
def test_pymss_runner_success(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    runner = PymssSeparateRunner(script_path=FAKE_PROTOCOL)
    artifacts, progress_events, _logs = _run_runner(
        runner,
        {
            "input_path": str(_input_audio(tmp_path)),
            "model": "fake-model",
            "device": "cpu",
            "chunk": 48000,
        },
        tmp_path / "work",
    )
    assert len(artifacts) == 2
    assert progress_events[-1][0] == 100.0


# --------------------------------------------------------------------- 未配置/参数校验
def test_backend_not_configured(tmp_path, monkeypatch):
    for key in (
        "AMS_RVC_PYTHON",
        "AMS_RVC_ROOT",
        "AMS_PYMSS_PYTHON",
        "AMS_PYMSS_ROOT",
        "AMS_MSVC_TOOL",
    ):
        monkeypatch.delenv(key, raising=False)
    runner = RvcConvertRunner(script_path=FAKE_PROTOCOL)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            runner,
            {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"},
            tmp_path / "work",
        )
    assert exc_info.value.code == "BACKEND_NOT_CONFIGURED"
    assert "AMS_RVC_PYTHON" in exc_info.value.hint


def test_backend_path_not_found(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AMS_RVC_ROOT", str(tmp_path / "not-exists"))
    runner = RvcConvertRunner(script_path=FAKE_PROTOCOL)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            runner,
            {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"},
            tmp_path / "work",
        )
    assert exc_info.value.code == "BACKEND_PATH_NOT_FOUND"


def test_missing_input_param(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(RvcConvertRunner(script_path=FAKE_PROTOCOL), {"voice_name": "manbo"}, tmp_path / "work")
    assert exc_info.value.code == "INVALID_PARAMS"


def test_input_not_found(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            RvcConvertRunner(script_path=FAKE_PROTOCOL),
            {"input_path": str(tmp_path / "missing.wav"), "voice_name": "manbo"},
            tmp_path / "work",
        )
    assert exc_info.value.code == "INPUT_NOT_FOUND"


def test_invalid_index_rate(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    with pytest.raises(RunnerError) as exc_info:
        _run_runner(
            RvcConvertRunner(script_path=FAKE_PROTOCOL),
            {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo", "index_rate": 1.5},
            tmp_path / "work",
        )
    assert exc_info.value.code == "INVALID_PARAMS"


# --------------------------------------------------------------------- App 集成
def test_default_app_registers_backend_job_types(client, auth):
    response = client.get("/api/v1/health", headers=auth)
    assert response.status_code == 200
    job_types = {item["type"]: item for item in response.json()["job_types"]}
    for item in BACKEND_JOB_TYPES:
        assert item["type"] in job_types
        assert job_types[item["type"]]["gpu"] == item["gpu"]
    # 原有 mock 任务不能被破坏
    assert {"ping", "mock.long"} <= set(job_types)


def _make_app_with_fake_backends(tmp_path: Path):
    settings = Settings.from_env(ams_home=tmp_path / "AMS_APP", host="127.0.0.1", port=0, log_level="WARNING")
    registry = build_default_registry()
    register_backend_runners(registry, rvc_script=FAKE_PROTOCOL, pymss_script=FAKE_PROTOCOL)
    return create_app(settings, registry=registry), settings


def test_app_job_with_fake_backend_succeeds(tmp_path, monkeypatch):
    _configure_backend_env(monkeypatch, tmp_path)
    app, _settings = _make_app_with_fake_backends(tmp_path)
    with TestClient(app) as test_client:
        auth = {"Authorization": f"Bearer {app.state.token}"}
        response = test_client.post(
            "/api/v1/jobs",
            headers=auth,
            json={
                "type": "convert.rvc",
                "params": {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"},
            },
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["id"]
        finished = wait_for_status(test_client, auth, job_id, {"succeeded", "failed"})
        assert finished["status"] == "succeeded", finished
        artifacts = test_client.get(f"/api/v1/jobs/{job_id}/artifacts", headers=auth).json()["data"]
        assert artifacts and all(Path(item["path"]).is_file() for item in artifacts)


def test_app_unconfigured_backend_job_fails(tmp_path, monkeypatch):
    for key in (
        "AMS_RVC_PYTHON",
        "AMS_RVC_ROOT",
        "AMS_PYMSS_PYTHON",
        "AMS_PYMSS_ROOT",
        "AMS_MSVC_TOOL",
    ):
        monkeypatch.delenv(key, raising=False)
    app, _settings = _make_app_with_fake_backends(tmp_path)
    with TestClient(app) as test_client:
        auth = {"Authorization": f"Bearer {app.state.token}"}
        response = test_client.post(
            "/api/v1/jobs",
            headers=auth,
            json={
                "type": "convert.rvc",
                "params": {"input_path": str(_input_audio(tmp_path)), "voice_name": "manbo"},
            },
        )
        job_id = response.json()["id"]
        finished = wait_for_status(test_client, auth, job_id, {"succeeded", "failed", "cancelled"})
        assert finished["status"] == "failed"
        assert finished["error"]["code"] == "BACKEND_NOT_CONFIGURED"
        assert finished["error"].get("hint")


# --------------------------------------------------------------------- 真实后端 smoke（手动开启）
@pytest.mark.skipif(
    not os.environ.get("AMS_TEST_REAL_BACKENDS"),
    reason="设置 AMS_TEST_REAL_BACKENDS=1（并配置 AMS_* 与输入文件）后运行真实后端 smoke",
)
def test_real_backend_smoke_if_enabled(tmp_path):
    input_path = os.environ.get("AMS_TEST_REAL_INPUT", "")
    if not input_path:
        pytest.skip("AMS_TEST_REAL_INPUT 未设置")
    runner = RvcConvertRunner()
    artifacts, _progress, _logs = _run_runner(
        runner,
        {"input_path": input_path, "voice_name": os.environ.get("AMS_TEST_REAL_VOICE", "manbo")},
        tmp_path / "work",
    )
    assert artifacts
