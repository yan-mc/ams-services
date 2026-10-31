#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Headless GPT-SoVITS TTS service for AMS Gateway.

Run inside the portable GPT-SoVITS runtime:
  runtime\\python.exe -u -B tts_service\\tts_service.py --host 127.0.0.1 --port 0 \
      --sovits <path.pth> --gpt <path.ckpt> --version v5turbo \
      --ref-audio <ref.wav> --ref-text <text> --logo-file <file>

HTTP (loopback only):
  GET  /health
  POST /synthesize  {text, speed, seed, output_path}
  POST /shutdown
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import pathlib
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "GPT_SoVITS", ROOT / "tools", ROOT / "tools" / "asr", ROOT / "GPT_SoVITS" / "BigVGAN"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS  # noqa: E402


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class TtsModel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.version = args.version
        self.ref_audio = pathlib.Path(args.ref_audio).resolve()
        self.ref_text = args.ref_text
        self.language = args.language
        if not self.ref_audio.is_file():
            raise SystemExit(f"reference audio not found: {self.ref_audio}")
        config = {
            "custom": {
                "device": args.device,
                "is_half": args.is_half,
                "version": self.version,
                "t2s_weights_path": str(pathlib.Path(args.gpt).resolve()),
                "vits_weights_path": str(pathlib.Path(args.sovits).resolve()),
                "cnhuhbert_base_path": str(ROOT / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
                "bert_base_path": str(ROOT / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            }
        }
        t0 = time.time()
        self.tts = TTS(config)
        self.load_s = time.time() - t0
        self.sample_rate = int(getattr(self.tts.configs, "sampling_rate", 32000) or 32000)
        print(f"model ready version={self.version} load_s={self.load_s:.2f} sr={self.sample_rate}", flush=True)

    def synthesize(self, text: str, speed: float, seed: int, output_path: str) -> dict:
        if not text.strip():
            raise ValueError("text is empty")
        out = pathlib.Path(output_path)
        if not out.is_absolute():
            raise ValueError("output_path must be absolute")
        out.parent.mkdir(parents=True, exist_ok=True)
        inputs = {
            "text": text,
            "text_lang": self.language,
            "ref_audio_path": str(self.ref_audio),
            "prompt_text": self.ref_text,
            "prompt_lang": self.language,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": max(0.5, min(2.0, float(speed))),
            "seed": int(seed),
            "parallel_infer": True,
            "return_fragment": False,
            "streaming_mode": False,
        }
        t0 = time.time()
        result = list(self.tts.run(inputs))
        if not result:
            raise RuntimeError("GPT-SoVITS returned no audio")
        sr, audio = result[-1]
        audio = np.asarray(audio)
        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        from scipy.io import wavfile
        wavfile.write(str(out), int(sr), audio)
        duration = len(audio) / float(sr) if sr else 0.0
        gen_s = time.time() - t0
        return {
            "ok": True,
            "wav_path": str(out),
            "sample_rate": int(sr),
            "duration_s": round(duration, 3),
            "gen_s": round(gen_s, 3),
            "rtf": round(gen_s / duration, 3) if duration else None,
            "engine": self.version,
            "model_load_s": round(self.load_s, 3),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "AMSGsvTts/1"
    protocol_version = "HTTP/1.1"
    model: TtsModel
    token: str = ""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, payload: dict) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.token:
            return True
        auth = self.headers.get("Authorization", "")
        return hmac.compare_digest(auth, f"Bearer {self.token}")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > 4 * 1024 * 1024:
            raise ValueError("invalid body length")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"ok": False, "error": "not found"})
            return
        self._send(200, {"ok": True, "engine": self.model.version, "sample_rate": self.model.sample_rate,
                         "load_s": round(self.model.load_s, 3)})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if not self._authorized():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            if path == "/synthesize":
                body = self._body()
                result = self.model.synthesize(
                    text=str(body.get("text") or ""),
                    speed=float(body.get("speed") or 1.0),
                    seed=int(body.get("seed") if body.get("seed") is not None else 42),
                    output_path=str(body.get("output_path") or ""),
                )
                self._send(200, result)
                return
            if path == "/shutdown":
                self._send(200, {"ok": True, "message": "shutting down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--sovits", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--is-half", action="store_true", default=True)
    ap.add_argument("--token", default="")
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("only loopback host is allowed")
    model = TtsModel(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    Handler.model = model
    Handler.token = args.token
    host, port = server.server_address[0], int(server.server_address[1])
    print(json.dumps({"event": "ready", "host": host, "port": port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
