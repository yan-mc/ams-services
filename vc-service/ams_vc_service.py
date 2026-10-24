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

# Bundled loopback runtime.  It is extracted from service.vc under data; no
# pip install, no system Python site-packages, and no user-profile writes.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
try:
    import pyaudiowpatch as pyaudio_loopback
except Exception:
    pyaudio_loopback = None

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
MONITOR_STREAM = None
MONITOR_QUEUE = None
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
    "input_frames": 0,
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
    # Hide host-API duplicates (MME/DirectSound/WASAPI/WDM-KS) from the main
    # output picker. Prefer WASAPI for each endpoint, and give the user a
    # readable default/virtual label instead of ten near-identical names.
    priority = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2, "Windows WDM-KS": 3}
    default_output_index = int(sd.default.device[1])
    default_output_name = str(sd.query_devices(default_output_index).get("name", ""))
    groups = {}
    for item in outputs:
        groups.setdefault(item["name"], []).append(item)
    output_choices = []
    hidden_names = ("Microsoft 声音映射器", "Microsoft Sound Mapper", "主声音驱动程序", "Primary Sound Driver")
    virtual_items = []
    for name, items in groups.items():
        if any(token in name for token in hidden_names):
            continue
        items = sorted(items, key=lambda x: (priority.get(x.get("hostapi_name", ""), 9), int(x["index"])))
        best = items[0]
        is_virtual = bool(best.get("kind") == "virtual")
        if is_virtual:
            virtual_items.extend(items)
            continue
        is_default = name == default_output_name
        if is_default:
            display = "默认输出（当前耳机/扬声器） · " + name
        elif "耳机" in name or "Headphone" in name:
            display = "耳机 · " + name
        else:
            display = name
        output_choices.append({
            "index": int(best["index"]),
            "name": name,
            "display_name": display,
            "hostapi": best.get("hostapi_name", ""),
            "is_default": is_default,
            "is_virtual": False,
            "alt_indices": [int(x["index"]) for x in items[1:]],
        })
    if virtual_items:
        deduped = []
        seen_virtual = set()
        for item in virtual_items:
            idx = int(item["index"])
            if idx in seen_virtual:
                continue
            seen_virtual.add(idx)
            deduped.append(item)
        virtual_items = sorted(deduped, key=lambda x: (priority.get(x.get("hostapi_name", ""), 9), int(x["index"])))
        best = virtual_items[0]
        output_choices.insert(0, {
            "index": int(best["index"]),
            "name": best["name"],
            "display_name": "VB-Cable 虚拟输出（给 OBS/会议）",
            "hostapi": best.get("hostapi_name", ""),
            "is_default": False,
            "is_virtual": True,
            "alt_indices": [int(x["index"]) for x in virtual_items[1:]],
        })
    output_choices.sort(key=lambda x: (0 if x["is_virtual"] else 1, 0 if x["is_default"] else 1, x["name"]))
    # Keep the main picker intentionally short: VB-Cable plus the current
    # default headphone/speaker. Advance users can still use alt_indices via API.
    output_choices = [x for x in output_choices if x["is_virtual"] or x["is_default"]]
    default_input_index = int(sd.default.device[0])
    default_input_name = str(sd.query_devices(default_input_index).get("name", ""))
    input_choices = []
    seen_input_names = set()
    for item in inputs:
        if item.get("hostapi_name") != "Windows WASAPI":
            continue
        if int(item.get("channels", 0)) <= 0 or float(item.get("samplerate", 0)) < 48000:
            continue
        name = item["name"]
        if any(token in name for token in ("VMic", "CABLE-A Output", "CABLE Output")):
            continue
        if name in seen_input_names:
            continue
        seen_input_names.add(name)
        is_default = name == default_input_name
        display = "默认麦克风（当前设备） · " + name if is_default else "麦克风 · " + name
        input_choices.append({
            "index": int(item["index"]), "name": name, "display_name": display,
            "hostapi": item.get("hostapi_name", ""), "is_default": is_default,
        })
    input_choices.sort(key=lambda x: (0 if x["is_default"] else 1, x["name"]))
    wasapi_loopbacks = []
    if pyaudio_loopback is not None:
        try:
            pa = pyaudio_loopback.PyAudio()
            try:
                default = pa.get_default_wasapi_loopback()
                default_index = int(default.get("index", -1)) if default else -1
                for info in pa.get_loopback_device_info_generator():
                    wasapi_loopbacks.append({
                        "index": int(info["index"]),
                        "value": "wasapi:" + str(info["index"]),
                        "name": str(info["name"]),
                        "hostapi": int(info.get("hostApi", -1)),
                        "channels": int(info.get("maxInputChannels", 0)),
                        "samplerate": float(info.get("defaultSampleRate", 48000)),
                        "is_default": int(info["index"]) == default_index,
                    })
            finally:
                pa.terminate()
        except Exception:
            wasapi_loopbacks = []
    return {
        "inputs": inputs,
        "outputs": outputs,
        "virtual_outputs": cable_in,
        "output_choices": output_choices,
        "input_choices": input_choices,
        "default_input_name": default_input_name,
        "default_output_name": default_output_name,
        "loopback_inputs": cable_out,
        "wasapi_loopbacks": wasapi_loopbacks,
        "has_wasapi_loopback": bool(wasapi_loopbacks),
        "has_vb_cable": bool(cable_in),
        "default_input": int(sd.default.device[0]),
        "default_output": int(sd.default.device[1]),
    }


def _prefer_wasapi(index: int, kind: str) -> int:
    """Map a stale MME/DirectSound/WDM-KS selection to its WASAPI endpoint.

    The UI now only offers WASAPI endpoints, but older pages or saved forms can
    still send MME/WDM-KS indices.  WDM-KS is especially fragile after a rapid
    stop/start, so avoid opening it when an equivalent WASAPI device exists.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    info = devices[index]
    host_name = str(hostapis[int(info.get("hostapi", 0))].get("name", ""))
    if host_name not in ("MME", "Windows DirectSound", "Windows WDM-KS"):
        return index
    name = str(info.get("name", ""))
    for candidate, item in enumerate(devices):
        c_host = str(hostapis[int(item.get("hostapi", 0))].get("name", ""))
        if c_host != "Windows WASAPI":
            continue
        c_name = str(item.get("name", ""))
        if c_name != name:
            continue
        if kind == "input" and int(item.get("max_input_channels", 0)) > 0:
            return candidate
        if kind == "output" and int(item.get("max_output_channels", 0)) > 0:
            return candidate
    if kind == "output" and ("CABLE" in name or "VB-Audio" in name):
        for candidate, item in enumerate(devices):
            c_host = str(hostapis[int(item.get("hostapi", 0))].get("name", ""))
            c_name = str(item.get("name", ""))
            if c_host == "Windows WASAPI" and ("VSpk" in c_name or "CABLE" in c_name) and int(item.get("max_output_channels", 0)) > 0:
                return candidate
    if "Realtek" in name:
        for candidate, item in enumerate(devices):
            c_host = str(hostapis[int(item.get("hostapi", 0))].get("name", ""))
            c_name = str(item.get("name", ""))
            if c_host == "Windows WASAPI" and "Realtek" in c_name:
                if kind == "input" and int(item.get("max_input_channels", 0)) > 0:
                    return candidate
                if kind == "output" and int(item.get("max_output_channels", 0)) > 0:
                    return candidate
    if kind == "input" and ("立体声混音" in name or "Stereo Mix" in name):
        raise RuntimeError("请改用“WASAPI Loopback · 扬声器”作为系统输出；旧版立体声混音 WDM-KS 设备已不再使用")
    return index


def _pick_device(kind: str, preferred: int | None, name_contains: str) -> int:
    devices = sd.query_devices()
    if preferred is not None:
        return _prefer_wasapi(int(preferred), kind)
    for index, item in enumerate(devices):
        name = str(item.get("name", ""))
        if name_contains.lower() in name.lower():
            if kind == "input" and int(item.get("max_input_channels", 0)) > 0:
                return _prefer_wasapi(index, kind)
            if kind == "output" and int(item.get("max_output_channels", 0)) > 0:
                return _prefer_wasapi(index, kind)
    default = sd.default.device[0 if kind == "input" else 1]
    if default is not None and int(default) >= 0:
        return _prefer_wasapi(int(default), kind)
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


class LoopbackStream:
    """Bundled PyAudioWPatch loopback input + sounddevice output.

    PyAudioWPatch loopback runs in a blocking-read worker thread instead of a
    PortAudio callback; a callback from a second PortAudio instance can stay
    silent when sounddevice also has an active output stream.
    """
    def __init__(self, pa, input_stream, output_stream, reader, frames):
        self.pa = pa
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.reader = reader
        self.frames = frames
        self.stop_event = threading.Event()
        self.worker = None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                data = self.input_stream.read(self.frames, exception_on_overflow=False)
            except Exception as exc:
                with STATE_LOCK:
                    STATE["last_error"] = str(exc)
                break
            try:
                self.reader(data)
            except Exception as exc:
                with STATE_LOCK:
                    STATE["last_error"] = str(exc)

    def start(self):
        self.output_stream.start()
        self.input_stream.start_stream()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        try: self.input_stream.stop_stream()
        except Exception: pass
        try: self.output_stream.stop()
        except Exception: pass
        if self.worker is not None:
            self.worker.join(timeout=1.0)

    def close(self):
        try: self.input_stream.close()
        except Exception: pass
        try: self.output_stream.close()
        except Exception: pass
        try: self.pa.terminate()
        except Exception: pass


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


def _push_monitor(data) -> None:
    sink = MONITOR_QUEUE
    if sink is None:
        return
    try:
        sink.put_nowait(np.array(data, copy=True))
    except queue.Full:
        try: sink.get_nowait()
        except queue.Empty: pass
        try: sink.put_nowait(np.array(data, copy=True))
        except queue.Full: pass


def _monitor_callback(outdata, frames, time_info, status):
    sink = MONITOR_QUEUE
    if sink is None:
        outdata.fill(0)
        return
    try:
        data = sink.get_nowait()
    except queue.Empty:
        outdata.fill(0)
        return
    if data.shape[0] < frames:
        data = np.pad(data, ((0, frames - data.shape[0]), (0, 0)))
    outdata[:] = data[:frames]


def _start_monitor_stream(device_index, samplerate: int):
    global MONITOR_STREAM, MONITOR_QUEUE
    if device_index is None:
        return
    MONITOR_QUEUE = queue.Queue(maxsize=8)
    stream = sd.OutputStream(
        device=int(device_index), samplerate=int(samplerate), channels=1,
        dtype="float32", blocksize=max(256, int(round(0.05 * samplerate))),
        callback=_monitor_callback,
    )
    stream.start()
    MONITOR_STREAM = stream


def _stream_callback(indata, outdata, frames, time_info, status):
    if status:
        with STATE_LOCK:
            STATE["overflows"] += 1
    outdata[:] = indata
    _push_monitor(outdata)
    with STATE_LOCK:
        STATE["frames"] += int(frames)


def start_session(params: dict) -> dict:
    global STREAM, RVC_ENGINE
    with STATE_LOCK:
        if STATE["running"]:
            raise RuntimeError("session already running")
    raw_input = str(params.get("input_device") or "")
    input_provider = str(params.get("input_provider") or ("wasapi" if raw_input.startswith("wasapi:") else "")).lower()
    if input_provider == "wasapi":
        input_device = raw_input
        output_device = _pick_device("output", params.get("output_device"), params.get("output_name") or "VSpk")
    else:
        input_device = _pick_device("input", params.get("input_device"), params.get("input_name") or "USB Audio")
        output_device = _pick_device("output", params.get("output_device"), params.get("output_name") or "VSpk")
        input_device, output_device = _match_hostapi(input_device, output_device)
    monitor_index = None
    monitor_raw = str(params.get("monitor_device") or "").strip()
    if monitor_raw:
        monitor_index = _pick_device("output", monitor_raw, "VSpk")
        if int(monitor_index) == int(output_device):
            monitor_index = None
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
            _push_monitor(outdata)
    else:
        samplerate = int(params.get("samplerate") or sd.query_devices(output_device)["default_samplerate"] or 48000)
        if samplerate <= 0:
            samplerate = 48000
        blocksize = max(256, samplerate // 100)
        callback = _stream_callback
    if input_provider == "wasapi":
        if pyaudio_loopback is None:
            raise RuntimeError("内置 WASAPI loopback 运行时不可用；请重装 service.vc 组件")
        pa = pyaudio_loopback.PyAudio()
        try:
            default_info = pa.get_default_wasapi_loopback()
            loopback_index = int(raw_input.split(":")[-1]) if ":" in raw_input and raw_input.split(":")[-1].isdigit() else int(default_info["index"])
            info = pa.get_device_info_by_index(loopback_index)
            channels = max(1, min(2, int(info.get("maxInputChannels", 2))))
            device_sr = 48000
            samplerate = device_sr
            if engine is not None:
                import torch
                import torchaudio.transforms as tat
                in_resampler = tat.Resample(orig_freq=device_sr, new_freq=engine.sr, dtype=torch.float32)
                out_resampler = tat.Resample(orig_freq=engine.sr, new_freq=device_sr, dtype=torch.float32)
                capture_frames = int(round(engine.block_frame * device_sr / engine.sr))
            else:
                in_resampler = None
                out_resampler = None
                capture_frames = 4800
            capture_frames = max(512, capture_frames)
            frames_queue = queue.Queue(maxsize=8)

            def consume_loopback(in_data):
                try:
                    with STATE_LOCK:
                        STATE["input_frames"] += int(len(in_data) // (2 * channels))
                    raw = np.frombuffer(in_data, dtype=np.int16)
                    if channels > 1:
                        raw = raw.reshape(-1, channels).mean(axis=1)
                    mono = raw.astype(np.float32) / 32768.0
                    current = ENGINE_HOLDER.get("engine")
                    if current is not None:
                        res = in_resampler(torch.from_numpy(mono.copy())).numpy()
                        if res.shape[0] < current.block_frame:
                            res = np.pad(res, (0, current.block_frame - res.shape[0]))
                        if res.shape[0] > current.block_frame:
                            res = res[-current.block_frame:]
                        tmp = np.zeros((current.block_frame, 1), dtype="float32")
                        current.callback(res.reshape(-1, 1), tmp, current.block_frame, None, None)
                        data = out_resampler(torch.from_numpy(tmp[:, 0].copy())).numpy().reshape(-1, 1)
                    else:
                        data = mono.reshape(-1, 1)
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

            def loopback_output_cb(outdata, frame_count, time_info, out_status):
                try:
                    data = frames_queue.get_nowait()
                except queue.Empty:
                    outdata.fill(0)
                else:
                    if data.shape[0] < frame_count:
                        data = np.pad(data, ((0, frame_count - data.shape[0]), (0, 0)))
                    outdata[:] = data[:frame_count]
                _push_monitor(outdata)
                with STATE_LOCK:
                    STATE["frames"] += int(frame_count)

            pa_stream = pa.open(
                format=pyaudio_loopback.paInt16,
                channels=channels,
                rate=device_sr,
                input=True,
                input_device_index=loopback_index,
                frames_per_buffer=capture_frames,
                start=False,
            )
            output_stream = sd.OutputStream(
                device=output_device,
                samplerate=device_sr,
                channels=1,
                dtype="float32",
                blocksize=max(256, int(round(0.05 * device_sr))),
                callback=loopback_output_cb,
            )
            stream = LoopbackStream(pa, pa_stream, output_stream, consume_loopback, capture_frames)
            try:
                stream.start()
                if monitor_index is not None:
                    _start_monitor_stream(monitor_index, device_sr)
            except Exception:
                stream.close()
                raise
            with STATE_LOCK:
                STREAM = stream
                RVC_ENGINE = engine
                ENGINE_HOLDER["engine"] = engine
                STATE.update({
                    "running": True,
                    "mode": "rvc" if engine is not None else "passthrough",
                    "input_device": raw_input,
                    "output_device": output_device,
                    "samplerate": device_sr,
                    "started_at": time.time(),
                    "monitor_device": monitor_index,
                    "last_error": (engine.last_error or engine.index_warning) if engine is not None else "",
                    "voice": params.get("voice"),
                    "index": params.get("index_path"),
                    "pitch": int(params.get("pitch") or 0),
                    "index_rate": float(params.get("index_rate") or 0),
                })
            return status_payload()
        except Exception:
            try: pa.terminate()
            except Exception: pass
            raise

    # WDM-KS inputs (notably Stereo Mix) often cannot be paired with a
    # different-host duplex stream. Use independent input/output streams for
    # loopback devices; the microphone path keeps the lower-latency duplex.
    input_name = str(sd.query_devices(input_device).get("name", ""))
    is_loopback = engine is not None or any(token in input_name for token in ("立体声混音", "Stereo Mix", "CABLE-A Output", "CABLE Output", "VMic"))
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
            blocksize = max(256, int(round(engine.block_time * device_sr)))
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
            _push_monitor(outdata)
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
            try: stream.close()
            except Exception: pass
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
    if monitor_index is not None:
        try:
            _start_monitor_stream(monitor_index, samplerate)
        except Exception:
            try: stream.stop()
            except Exception: pass
            try: stream.close()
            except Exception: pass
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
            "monitor_device": monitor_index,
            "last_error": (engine.last_error or engine.index_warning) if engine is not None else "",
            "voice": params.get("voice"),
            "index": params.get("index_path"),
            "pitch": int(params.get("pitch") or 0),
            "index_rate": float(params.get("index_rate") or 0),
        })
    return status_payload()


def start_session_with_retry(params: dict) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            return start_session(params)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            retryable = ("PaErrorCode -9999" in message or "PaErrorCode -9997" in message or "WdmSyncIoctl" in message)
            if attempt >= 2 or not retryable:
                raise
            try:
                stop_session()
            except Exception:
                pass
            time.sleep(0.45 * (attempt + 1))
            try:
                # PortAudio/WASAPI can leave the endpoint in a transient
                # not-ready state after a fast stop/start cycle.
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
    if last_error is not None:
        raise last_error
    raise RuntimeError("VC start failed")


def stop_session() -> dict:
    global STREAM, RVC_ENGINE, MONITOR_STREAM, MONITOR_QUEUE
    with STATE_LOCK:
        stream = STREAM
        STREAM = None
        RVC_ENGINE = None
        ENGINE_HOLDER["engine"] = None
        monitor = MONITOR_STREAM
        MONITOR_STREAM = None
        MONITOR_QUEUE = None
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
    if monitor is not None:
        try:
            monitor.stop()
            monitor.close()
        except Exception as exc:
            with STATE_LOCK:
                STATE["last_error"] = str(exc)
    # Give WDM/WASAPI a moment to release the endpoint before a new start.
    if stream is not None or monitor is not None:
        time.sleep(0.25)
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
        global RVC_ENGINE
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/vc/session/start":
                self._send(200, start_session_with_retry(body))
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
                        current_engine = RVC_ENGINE
                    engine_sr = int(current_engine.sr) if current_engine is not None else int(current_sr or 0)
                    if engine_sr and engine_sr == int(new_engine.sr):
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
                        self._send(200, start_session_with_retry(params))
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
