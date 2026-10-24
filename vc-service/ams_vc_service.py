#!/usr/bin/env python3
"""AMS Stage B VC service - local device/stream MVP.

Run with the managed RVC runtime Python:
  runtime\python.exe ams_vc_service.py --port 7801

The service only binds 127.0.0.1.  It exposes:
  GET  /vc/health
  GET  /vc/devices
  GET  /vc/status
  POST /vc/session/start
  POST /vc/session/stop
  POST /vc/session/switch-voice

The initial mode is passthrough; RVC mode is enabled when a .pth is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


# Keep every writable temp/cache location under <install-root>/data, even when
# the service is launched manually instead of by the Host plugin.
def _prepare_data_env() -> None:
    home = os.environ.get("AMS_HOME", "").strip()
    if not home:
        home = str(Path(__file__).resolve().parents[2])
    if not home:
        return
    os.environ.setdefault("AMS_HOME", home)
    temp = Path(home) / "tmp"
    cache = Path(home) / "cache"
    paths = {
        "TEMP": temp,
        "TMP": temp,
        "TMPDIR": temp,
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "HF_HOME": cache / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "XDG_CACHE_HOME": cache,
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))


_prepare_data_env()

import numpy as np
import sounddevice as sd


class RvcEngine:
    """Small headless real-time RVC callback adapted from the RVC GUI buffer flow."""

    def __init__(self, voice: str, index_path: str | None, pitch: int, index_rate: float, f0method: str = "rmvpe", block_time: float = 0.1, extra_time: float = 0.25):
        root = os.environ.get("AMS_RVC_ROOT", "").strip()
        if root:
            os.chdir(root)
            if root not in sys.path:
                sys.path.insert(0, root)
        if index_path and not Path(index_path).is_file():
            index_path = None
            index_rate = 0.0
        if not index_path:
            index_rate = 0.0
        import librosa
        import torch
        import torchaudio.transforms as tat
        from configs.config import Config
        from infer.lib import rtrvc

        self.config = Config()
        self.config.use_jit = False
        if os.environ.get("AMS_VC_DEVICE", "").lower() == "cpu":
            self.config.device = torch.device("cpu")
            self.config.is_half = False
        self.index_warning = ""
        self.rvc = rtrvc.RVC(
            int(pitch), 0.0, voice, index_path or "", float(index_rate),
            1, None, None, self.config, None,
        )
        if not hasattr(self.rvc, "tgt_sr") and index_path:
            # A stale/incompatible faiss index must not make realtime unusable;
            # retry without it and surface a non-fatal warning in status.
            self.index_warning = "索引文件加载失败，已自动禁用 index"
            index_path = None
            index_rate = 0.0
            self.rvc = rtrvc.RVC(
                int(pitch), 0.0, voice, "", 0.0,
                1, None, None, self.config, None,
            )
        if not hasattr(self.rvc, "tgt_sr"):
            raise RuntimeError("RVC model load failed; check pth/index and runtime log")
        self.sr = int(self.rvc.tgt_sr)
        self.zc = self.sr // 100
        self.block_time = max(0.03, min(0.5, float(block_time)))
        self.block_frame = int(round(self.block_time * self.sr / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(round(0.05 * self.sr / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(max(0.05, min(2.5, float(extra_time))) * self.sr / self.zc)) * self.zc
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device, dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device, dtype=torch.float32,
        )
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.fade_in = (torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)) ** 2)
        self.fade_out = 1 - self.fade_in
        self.resampler = tat.Resample(orig_freq=self.sr, new_freq=16000, dtype=torch.float32).to(self.config.device)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.frames = 0
        self.last_error = ""
        self.infer_ms = 0.0
        self.max_infer_ms = 0.0
        self.callback_ms = 0.0
        self._librosa = librosa
        self._torch = torch
        self._f0method = f0method

    def callback(self, indata, outdata, frames, time_info, status):
        torch = self._torch
        callback_started = time.perf_counter()
        try:
            mono = np.asarray(indata)
            if mono.ndim > 1:
                mono = mono[:, 0]
            mono = mono.astype(np.float32, copy=False)
            if mono.shape[0] < self.block_frame:
                mono = np.pad(mono, (0, self.block_frame - mono.shape[0]))
            if mono.shape[0] > self.block_frame:
                mono = mono[-self.block_frame:]
            self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
            self.input_wav[-self.block_frame:] = torch.from_numpy(mono).to(self.config.device)
            self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
            res = self.resampler(self.input_wav[-self.block_frame - 2 * self.zc:])
            n = min(res.shape[0], self.input_wav_res.shape[0])
            self.input_wav_res[-n:] = res[-n:]
            infer_started = time.perf_counter()
            infer = self.rvc.infer(
                self.input_wav_res, self.block_frame_16k, self.skip_head,
                self.return_length, self._f0method,
            )
            self.infer_ms = round((time.perf_counter() - infer_started) * 1000.0, 2)
            self.max_infer_ms = max(self.max_infer_ms, self.infer_ms)
            conv_input = infer[None, None, :self.sola_buffer_frame + self.sola_search_frame]
            cor_nom = torch.nn.functional.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(torch.nn.functional.conv1d(conv_input ** 2, torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device)) + 1e-8)
            sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
            infer = infer[sola_offset:]
            infer[:self.sola_buffer_frame] *= self.fade_in
            infer[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out
            self.sola_buffer[:] = infer[self.block_frame:self.block_frame + self.sola_buffer_frame]
            out = infer[:self.block_frame]
            if out.shape[0] < frames:
                out = torch.cat([out, torch.zeros(frames - out.shape[0], device=self.config.device)])
            outdata[:] = out[:frames].unsqueeze(1).cpu().numpy()
            self.frames += int(frames)
            self.callback_ms = round((time.perf_counter() - callback_started) * 1000.0, 2)
            with STATE_LOCK:
                STATE["frames"] += int(frames)
        except Exception as exc:
            self.last_error = str(exc)
            with STATE_LOCK:
                STATE["last_error"] = str(exc)
            outdata.fill(0)

STATE_LOCK = threading.Lock()
STREAM = None
RVC_ENGINE = None
ENGINE_HOLDER = {"engine": None}
STATE = {
    "running": False,
    "mode": "idle",
    "input_device": None,
    "output_device": None,
    "samplerate": None,
    "voice": None,
    "index": None,
    "pitch": 0,
    "index_rate": 0.0,
    "started_at": None,
    "last_error": "",
    "frames": 0,
    "overflows": 0,
}


def list_devices() -> dict:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = []
    outputs = []
    for index, item in enumerate(devices):
        name = str(item.get("name", ""))
        if int(item.get("max_input_channels", 0)) > 0:
            inputs.append({
                "index": index,
                "name": name,
                "hostapi": str(item.get("hostapi", "")),
                "hostapi_name": str(hostapis[int(item.get("hostapi", 0))].get("name", "")),
                "channels": int(item.get("max_input_channels", 0)),
                "samplerate": float(item.get("default_samplerate", 0)),
                "kind": "loopback" if ("Stereo Mix" in name or "立体声混音" in name or "CABLE-A Output" in name or "CABLE Output" in name or "VMic" in name) else "input",
            })
        if int(item.get("max_output_channels", 0)) > 0:
            outputs.append({
                "index": index,
                "name": name,
                "hostapi": str(item.get("hostapi", "")),
                "hostapi_name": str(hostapis[int(item.get("hostapi", 0))].get("name", "")),
                "channels": int(item.get("max_output_channels", 0)),
                "samplerate": float(item.get("default_samplerate", 0)),
                "kind": "virtual" if ("VSpk" in name or "CABLE Input" in name or "CABLE-A" in name or "CABLE" in name) else "output",
            })
    cable_in = [x for x in outputs if x["kind"] == "virtual"]
    cable_out = [x for x in inputs if x["kind"] == "loopback"]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "virtual_outputs": cable_in,
        "loopback_inputs": cable_out,
        "has_vb_cable": bool(cable_in),
        "default_input": int(sd.default.device[0]),
        "default_output": int(sd.default.device[1]),
    }


def _pick_device(kind: str, preferred: int | None, name_contains: str) -> int:
    devices = sd.query_devices()
    if preferred is not None:
        return int(preferred)
    for index, item in enumerate(devices):
        name = str(item.get("name", ""))
        if name_contains.lower() in name.lower():
            if kind == "input" and int(item.get("max_input_channels", 0)) > 0:
                return index
            if kind == "output" and int(item.get("max_output_channels", 0)) > 0:
                return index
    default = sd.default.device[0 if kind == "input" else 1]
    if default is not None and int(default) >= 0:
        return int(default)
    raise RuntimeError("no " + kind + " audio device")


def _match_hostapi(input_device: int, output_device: int) -> tuple[int, int]:
    """PortAudio cannot combine devices from different host APIs in one stream.
    For a system-loopback input (WDM-KS Stereo Mix), prefer the VB-Cable output
    on the same host API instead of failing with an opaque error."""
    devices = sd.query_devices()
    in_info = devices[input_device]
    out_info = devices[output_device]
    if int(in_info.get("hostapi", -1)) == int(out_info.get("hostapi", -2)):
        return input_device, output_device
    hostapi = int(in_info.get("hostapi", -1))
    for index, item in enumerate(devices):
        name = str(item.get("name", ""))
        if int(item.get("hostapi", -1)) == hostapi and int(item.get("max_output_channels", 0)) > 0 and ("VSpk" in name or "CABLE" in name):
            return input_device, index
    raise RuntimeError("输入和输出设备属于不同音频后端，请选择同一后端（例如 WDM-KS 的立体声混音 + WDM-KS 的 CABLE 输出）")


class SplitStream:
    def __init__(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream

    def start(self):
        self.output_stream.start()
        self.input_stream.start()

    def stop(self):
        try: self.input_stream.stop()
        except Exception: pass
        try: self.output_stream.stop()
        except Exception: pass

    def close(self):
        try: self.input_stream.close()
        except Exception: pass
        try: self.output_stream.close()
        except Exception: pass


def _stream_callback(indata, outdata, frames, time_info, status):
    if status:
        with STATE_LOCK:
            STATE["overflows"] += 1
    outdata[:] = indata
    with STATE_LOCK:
        STATE["frames"] += int(frames)


def start_session(params: dict) -> dict:
    global STREAM, RVC_ENGINE
    with STATE_LOCK:
        if STATE["running"]:
            raise RuntimeError("session already running")
    input_device = _pick_device("input", params.get("input_device"), params.get("input_name") or "USB Audio")
    output_device = _pick_device("output", params.get("output_device"), params.get("output_name") or "VSpk")
    input_device, output_device = _match_hostapi(input_device, output_device)
    mode = str(params.get("mode") or "passthrough").lower()
    engine = None
    if mode == "rvc":
        voice = str(params.get("voice") or "").strip()
        if not voice:
            raise RuntimeError("rvc mode requires voice (.pth) path")
        engine = RvcEngine(
            voice,
            str(params.get("index_path") or "") or None,
            int(params.get("pitch") or 0),
            float(params.get("index_rate") or 0),
            str(params.get("f0method") or "rmvpe"),
            float(params.get("block_time") or 0.1),
            float(params.get("extra_time") or 0.25),
        )
        samplerate = int(engine.sr)
        blocksize = int(engine.block_frame)
        ENGINE_HOLDER["engine"] = engine

        def callback(indata, outdata, frames, time_info, status):
            current = ENGINE_HOLDER.get("engine")
            if current is None:
                outdata.fill(0)
                return
            current.callback(indata, outdata, frames, time_info, status)
    else:
        samplerate = int(params.get("samplerate") or sd.query_devices(output_device)["default_samplerate"] or 48000)
        if samplerate <= 0:
            samplerate = 48000
        blocksize = max(256, samplerate // 100)
        callback = _stream_callback
    # WDM-KS inputs (notably Stereo Mix) often cannot be paired with a
    # different-host duplex stream. Use independent input/output streams for
    # loopback devices; the microphone path keeps the lower-latency duplex.
    input_name = str(sd.query_devices(input_device).get("name", ""))
    is_loopback = any(token in input_name for token in ("立体声混音", "Stereo Mix", "CABLE-A Output", "CABLE Output", "VMic"))
    if is_loopback:
        frames_queue = queue.Queue(maxsize=8)
        in_resampler = None
        out_resampler = None
        if engine is not None:
            # WDM-KS Stereo Mix usually rejects the model rate (40k). Use the
            # device-native 48k stream and resample into/out of the RVC engine.
            import torch
            import torchaudio.transforms as tat
            device_sr = 48000
            samplerate = device_sr
            blocksize = int(round(0.25 * device_sr))
            in_resampler = tat.Resample(orig_freq=device_sr, new_freq=engine.sr, dtype=torch.float32)
            out_resampler = tat.Resample(orig_freq=engine.sr, new_freq=device_sr, dtype=torch.float32)

        def input_cb(indata, frames, time_info, status):
            try:
                current = ENGINE_HOLDER.get("engine")
                if current is not None:
                    mono = np.asarray(indata, dtype=np.float32)
                    if mono.ndim > 1: mono = mono[:, 0]
                    res = in_resampler(torch.from_numpy(mono.copy())).numpy()
                    if res.shape[0] < current.block_frame:
                        res = np.pad(res, (0, current.block_frame - res.shape[0]))
                    if res.shape[0] > current.block_frame:
                        res = res[-current.block_frame:]
                    tmp = np.zeros((current.block_frame, 1), dtype="float32")
                    current.callback(res.reshape(-1, 1), tmp, current.block_frame, time_info, status)
                    out = out_resampler(torch.from_numpy(tmp[:, 0].copy())).numpy()
                    data = out.reshape(-1, 1)
                else:
                    data = indata.copy()
                try:
                    frames_queue.put_nowait(data)
                except queue.Full:
                    try: frames_queue.get_nowait()
                    except queue.Empty: pass
                    try: frames_queue.put_nowait(data)
                    except queue.Full: pass
            except Exception as exc:
                with STATE_LOCK:
                    STATE["last_error"] = str(exc)

        def output_cb(outdata, frames, time_info, status):
            try:
                data = frames_queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
            else:
                if data.shape[0] < frames:
                    data = np.pad(data, ((0, frames - data.shape[0]), (0, 0)))
                outdata[:] = data[:frames]
            with STATE_LOCK:
                STATE["frames"] += int(frames)

        input_stream = sd.InputStream(device=input_device, samplerate=samplerate, channels=1, dtype="float32", blocksize=blocksize, callback=input_cb)
        output_stream = sd.OutputStream(device=output_device, samplerate=samplerate, channels=1, dtype="float32", blocksize=blocksize, callback=output_cb)
        stream = SplitStream(input_stream, output_stream)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
    else:
        # Use a conservative one-channel stream. PortAudio/MME usually accepts 48k.
        try:
            stream = sd.Stream(
                device=(input_device, output_device),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            )
            stream.start()
        except Exception:
            if engine is not None:
                raise
            try:
                samplerate = 44100
                stream = sd.Stream(
                    device=(input_device, output_device),
                    samplerate=samplerate,
                    channels=1,
                    dtype="float32",
                    blocksize=441,
                    callback=callback,
                )
                stream.start()
            except Exception:
                raise
    with STATE_LOCK:
        STREAM = stream
        RVC_ENGINE = engine
        ENGINE_HOLDER["engine"] = engine
        STATE.update({
            "running": True,
            "mode": "rvc" if engine is not None else "passthrough",
            "input_device": input_device,
            "output_device": output_device,
            "samplerate": samplerate,
            "started_at": time.time(),
            "last_error": (engine.last_error or engine.index_warning) if engine is not None else "",
            "voice": params.get("voice"),
            "index": params.get("index_path"),
            "pitch": int(params.get("pitch") or 0),
            "index_rate": float(params.get("index_rate") or 0),
        })
    return status_payload()


def stop_session() -> dict:
    global STREAM, RVC_ENGINE
    with STATE_LOCK:
        stream = STREAM
        STREAM = None
        RVC_ENGINE = None
        ENGINE_HOLDER["engine"] = None
        STATE["running"] = False
        STATE["mode"] = "idle"
        STATE["started_at"] = None
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            with STATE_LOCK:
                STATE["last_error"] = str(exc)
    return status_payload()


def status_payload() -> dict:
    with STATE_LOCK:
        data = dict(STATE)
        engine = RVC_ENGINE
    data["devices"] = "call /vc/devices"
    if engine is not None:
        data["stats"] = {
            "infer_ms": engine.infer_ms,
            "max_infer_ms": engine.max_infer_ms,
            "callback_ms": engine.callback_ms,
            "block_ms": round(engine.block_frame / engine.sr * 1000.0, 1),
            "buffer_ms": round((engine.extra_frame + engine.crossfade_frame) / engine.sr * 1000.0, 1),
            "estimated_latency_ms": round((engine.extra_frame + engine.crossfade_frame + engine.block_frame) / engine.sr * 1000.0 + engine.infer_ms, 1),
        }
    return {"ok": True, "data": data}


class Handler(BaseHTTPRequestHandler):
    server_version = "AMS-VC/0.1"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/vc/health":
                self._send(200, {"ok": True, "data": {"service": "ams-vc", "version": "0.1", "running": STATE["running"]}})
            elif path == "/vc/devices":
                self._send(200, {"ok": True, "data": list_devices()})
            elif path == "/vc/status":
                self._send(200, status_payload())
            else:
                self._send(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": path}})
        except Exception as exc:
            self._send(500, {"ok": False, "error": {"code": "VC_ERROR", "message": str(exc)}})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/vc/session/start":
                self._send(200, start_session(body))
            elif path == "/vc/session/stop":
                self._send(200, stop_session())
            elif path == "/vc/session/switch-voice":
                with STATE_LOCK:
                    running = bool(STATE["running"])
                    input_device = STATE["input_device"]
                    output_device = STATE["output_device"]
                if running:
                    params = dict(body)
                    params["input_device"] = input_device
                    params["output_device"] = output_device
                    params["mode"] = "rvc"
                    new_engine = RvcEngine(
                        str(params.get("voice") or "").strip(),
                        str(params.get("index_path") or "") or None,
                        int(params.get("pitch") or 0),
                        float(params.get("index_rate") or 0),
                        str(params.get("f0method") or "rmvpe"),
                        float(params.get("block_time") or 0.1),
                        float(params.get("extra_time") or 0.25),
                    )
                    with STATE_LOCK:
                        current_sr = STATE.get("samplerate")
                    if current_sr and int(current_sr) == int(new_engine.sr):
                        global RVC_ENGINE
                        ENGINE_HOLDER["engine"] = new_engine
                        with STATE_LOCK:
                            RVC_ENGINE = new_engine
                            STATE["voice"] = params.get("voice")
                            STATE["index"] = params.get("index_path")
                            STATE["pitch"] = int(params.get("pitch") or 0)
                            STATE["index_rate"] = float(params.get("index_rate") or 0)
                            STATE["last_error"] = new_engine.last_error or new_engine.index_warning
                        self._send(200, status_payload())
                    else:
                        stop_session()
                        self._send(200, start_session(params))
                else:
                    with STATE_LOCK:
                        STATE["voice"] = body.get("voice")
                        STATE["index"] = body.get("index_path")
                        STATE["pitch"] = int(body.get("pitch") or STATE["pitch"] or 0)
                        STATE["index_rate"] = float(body.get("index_rate") or STATE["index_rate"] or 0)
                    self._send(200, status_payload())
            else:
                self._send(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": path}})
        except Exception as exc:
            with STATE_LOCK:
                STATE["last_error"] = str(exc)
            self._send(400, {"ok": False, "error": {"code": "VC_ERROR", "message": str(exc), "detail": traceback.format_exc()}})

    def log_message(self, fmt, *args):
        print("[vc] " + (fmt % args), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7801)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("ams-vc listening on http://%s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_session()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
