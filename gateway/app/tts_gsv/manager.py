"""GPT-SoVITS 服务进程管理（Gateway 侧）。

职责：按 job 的 voice 配置启动 runtime 内的 headless tts_service，健康检查、
发起一次合成、无论成败都回收进程，避免 GPU 显存泄漏。
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
HEALTH_TIMEOUT = 240.0
SYNTH_TIMEOUT = 900.0


class GsvError(RuntimeError):
    def __init__(self, code: str, message: str, *, hint: str = "", detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.detail = detail or {}


@dataclass(slots=True)
class VoiceSpec:
    voice_id: str
    voice_dir: pathlib.Path
    version: str
    sovits: pathlib.Path
    gpt: pathlib.Path
    ref_audio: pathlib.Path
    ref_text: str
    language: str = "zh"
    display_name: str = ""


def ams_home() -> pathlib.Path:
    value = os.environ.get("AMS_HOME", "").strip()
    if not value:
        raise GsvError("BACKEND_NOT_CONFIGURED", "AMS_HOME 未配置", hint="由启动器注入 AMS_HOME")
    path = pathlib.Path(value).expanduser()
    if not path.is_dir():
        raise GsvError("BACKEND_PATH_NOT_FOUND", f"AMS_HOME 不存在：{path}")
    return path


def gsv_root() -> pathlib.Path:
    value = os.environ.get("AMS_TTS_GSV_ROOT", "").strip()
    if value:
        root = pathlib.Path(value).expanduser()
    else:
        root = ams_home() / "runtime" / "tts-gsv" / "0.1.0"
    if not root.is_dir():
        raise GsvError(
            "BACKEND_NOT_CONFIGURED",
            f"GPT-SoVITS 运行时不存在：{root}",
            hint="安装 runtime.tts-gsv-core 组件，或设置 AMS_TTS_GSV_ROOT",
        )
    return root


def gsv_home() -> pathlib.Path:
    value = os.environ.get("AMS_TTS_GSV_HOME", "").strip()
    home = pathlib.Path(value).expanduser() if value else ams_home() / "tts-gsv"
    return home


def _resolve_voice_path(voice_dir: pathlib.Path, value: Any) -> pathlib.Path:
    p = pathlib.Path(str(value or "")).expanduser()
    if not p.is_absolute():
        p = voice_dir / p
    return p.resolve()


def load_voice(
    *,
    voice_id: str = "",
    voice_dir: str = "",
    overrides: dict[str, Any] | None = None,
) -> VoiceSpec:
    overrides = overrides or {}
    voices_root = gsv_home() / "voices"
    if voice_dir:
        vdir = pathlib.Path(voice_dir).expanduser().resolve()
    elif voice_id:
        if not voice_id or any(ch in voice_id for ch in "\\/:*?\"<>|"):
            raise GsvError("INVALID_PARAMS", "voice_id 非法")
        vdir = (voices_root / voice_id).resolve()
    else:
        raise GsvError("INVALID_PARAMS", "缺少 voice_id 或 voice_dir")
    manifest = vdir / "voice.json"
    if not manifest.is_file():
        raise GsvError("INPUT_NOT_FOUND", f"音色配置不存在：{manifest}", hint="先创建/导入音色包")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise GsvError("INVALID_PARAMS", f"音色 voice.json 无法解析：{exc}") from exc
    merged = dict(data)
    merged.update({k: v for k, v in overrides.items() if v not in (None, "")})
    version = str(merged.get("version") or "v5turbo")
    if version not in {"v5turbo", "v5dev", "v2ProPlus", "v2Pro"}:
        raise GsvError("INVALID_PARAMS", f"不支持的 TTS 版本：{version}")
    sovits = _resolve_voice_path(vdir, merged.get("sovits"))
    gpt = _resolve_voice_path(vdir, merged.get("gpt"))
    ref_audio = _resolve_voice_path(vdir, merged.get("ref_audio"))
    for label, path in (("sovits", sovits), ("gpt", gpt), ("ref_audio", ref_audio)):
        if not path.is_file():
            raise GsvError("INPUT_NOT_FOUND", f"音色缺少 {label}：{path}")
    ref_text = str(merged.get("ref_text") or "").strip()
    if not ref_text:
        raise GsvError("INVALID_PARAMS", "音色缺少 ref_text（参考音频文本）")
    return VoiceSpec(
        voice_id=str(merged.get("id") or voice_id or vdir.name),
        voice_dir=vdir,
        version=version,
        sovits=sovits,
        gpt=gpt,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=str(merged.get("language") or "zh"),
        display_name=str(merged.get("name") or merged.get("display_name") or vdir.name),
    )


def _service_paths(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    python = root / "runtime" / "python.exe"
    if not python.is_file():
        python = root / "runtime" / "python" / "python.exe"
    script = root / "tts_service" / "tts_service.py"
    if not python.is_file():
        raise GsvError("BACKEND_PATH_NOT_FOUND", f"GPT-SoVITS Python 不存在：{python}", hint="重装 runtime.tts-gsv-core")
    if not script.is_file():
        raise GsvError("BACKEND_PATH_NOT_FOUND", f"tts_service 脚本不存在：{script}", hint="runtime 组件缺少 tts_service")
    return python, script


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, *, token: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def _wait_health(proc: asyncio.subprocess.Process, url: str, token: str, service_log: pathlib.Path) -> dict[str, Any]:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    last_error = ""
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            tail = ""
            try:
                tail = service_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                pass
            raise GsvError(
                "BACKEND_START_FAILED",
                f"tts_service 启动退出（code={proc.returncode}）",
                detail={"log_tail": tail, "service_log": str(service_log)},
            )
        try:
            return await asyncio.to_thread(_http_json, url + "/health", token=token, timeout=2.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
            await asyncio.sleep(0.5)
    raise GsvError(
        "BACKEND_TIMEOUT",
        "tts_service 健康检查超时",
        detail={"last_error": last_error, "service_log": str(service_log)},
    )


async def run_synthesis(
    spec: VoiceSpec,
    *,
    text: str,
    output_path: pathlib.Path,
    speed: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    root = gsv_root()
    python, script = _service_paths(root)
    home = gsv_home()
    logs = home / "logs"
    tmp = home / "tmp"
    logs.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    service_log = logs / f"tts-service-{stamp}-{secrets.token_hex(3)}.log"
    port = _free_port()
    token = secrets.token_hex(16)
    command = [
        str(python), "-u", "-B", str(script),
        "--host", "127.0.0.1", "--port", str(port),
        "--sovits", str(spec.sovits), "--gpt", str(spec.gpt),
        "--version", spec.version,
        "--ref-audio", str(spec.ref_audio), "--ref-text", spec.ref_text,
        "--language", spec.language, "--device", "cuda", "--token", token,
    ]
    env = dict(os.environ)
    env.update({
        "TEMP": str(tmp), "TMP": str(tmp),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    })
    log_handle = open(service_log, "a", encoding="utf-8", errors="replace")
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(root),
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        await _wait_health(proc, f"http://127.0.0.1:{port}", token, service_log)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _http_json,
                f"http://127.0.0.1:{port}/synthesize",
                token=token,
                payload={"text": text, "speed": speed, "seed": seed, "output_path": str(output_path)},
                timeout=SYNTH_TIMEOUT,
            ),
            timeout=SYNTH_TIMEOUT,
        )
        if not result.get("ok"):
            raise GsvError("TTS_FAILED", str(result.get("error") or "tts_service 合成失败"), detail=result)
        result["service_log"] = str(service_log)
        result["service_port"] = port
        return result
    except GsvError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GsvError("TTS_FAILED", f"tts_service 合成异常：{type(exc).__name__}: {exc}", detail={"service_log": str(service_log)}) from exc
    finally:
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.to_thread(_http_json, f"http://127.0.0.1:{port}/shutdown", token=token, payload={}, timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
        log_handle.close()
