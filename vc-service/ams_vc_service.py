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
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import sounddevice as sd


class RvcEngine:
    """Small headless real-time RVC callback adapted from the RVC GUI buffer flow."""

    def __init__(self, voice: str, index_path: str | None, pitch: int, index_rate: float, f0method: str = "rmvpe"):
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
        self.rvc = rtrvc.RVC(
            int(pitch), 0.0, voice, index_path or "", float(index_rate),
            1, None, None, self.config, None,
        )
        if not hasattr(self.rvc, "tgt_sr"):
            raise RuntimeError("RVC model load failed; check pth/index and runtime log")
        self.sr = int(self.rvc.tgt_sr)
        self.zc = self.sr // 100
        self.block_time = 0.25
        self.block_frame = int(round(self.block_time * self.sr / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(round(0.05 * self.sr / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(1.0 * self.sr / self.zc)) * self.zc
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
        self._librosa = librosa
        self._torch = torch
        self._f0method = f0method

    def callback(self, indata, outdata, frames, time_info, status):
        torch = self._torch
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
            infer = self.rvc.infer(
                self.input_wav_res, self.block_frame_16k, self.skip_head,
                self.return_length, self._f0method,
            )
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
    inputs = []
    outputs = []
    for index, item in enumerate(devices):
        name = str(item.get("name", ""))
        if int(item.get("max_input_channels", 0)) > 0:
            inputs.append({
                "index": index,
                "name": name,
                "hostapi": str(item.get("hostapi", "")),
                "channels": int(item.get("max_input_channels", 0)),
                "samplerate": float(item.get("default_samplerate", 0)),
                "kind": "loopback" if ("Stereo Mix" in name or "立体声混音" in name or "CABLE-A Output" in name or "CABLE Output" in name or "VMic" in name) else "input",
            })
        if int(item.get("max_output_channels", 0)) > 0:
            outputs.append({
                "index": index,
                "name": name,
                "hostapi": str(item.get("hostapi", "")),
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
        )
        samplerate = int(engine.sr)
        blocksize = int(engine.block_frame)
        callback = engine.callback
    else:
        samplerate = int(params.get("samplerate") or sd.query_devices(output_device)["default_samplerate"] or 48000)
        if samplerate <= 0:
            samplerate = 48000
        blocksize = max(256, samplerate // 100)
        callback = _stream_callback
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
        STATE.update({
            "running": True,
            "mode": "rvc" if engine is not None else "passthrough",
            "input_device": input_device,
            "output_device": output_device,
            "samplerate": samplerate,
            "started_at": time.time(),
            "last_error": engine.last_error if engine is not None else "",
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
    data["devices"] = "call /vc/devices"
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
