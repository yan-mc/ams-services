# -*- coding: utf-8 -*-
"""RVC 换声后端脚本（由 RVC 整合包 runtime python 3.9 执行）。

stdout 每行一个 JSON 消息（progress/log/artifact/result），供 Gateway Runner 解析。
只读 RVC 源目录；输出 wav 写到 --output，旁车 meta json 写到同目录。

用法：
  <AMS_RVC_PYTHON> -B rvc_convert.py --input in.wav --voice manbo --output out.wav ^
      --rvc-root F:\ai歌曲\RVC20260311 --f0-up-key 0 --index-rate 0.75 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

_EMIT_LOCK = threading.Lock()


def emit(payload):
    line = json.dumps(payload, ensure_ascii=False)
    with _EMIT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(level, message):
    emit({"type": "log", "level": level, "message": str(message)})


def progress(percent, stage, message=None):
    item = {"type": "progress", "percent": round(float(percent), 1), "stage": stage}
    if message:
        item["message"] = str(message)[:300]
    emit(item)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="RVC 换声后端")
    parser.add_argument("--input", required=True)
    parser.add_argument("--voice", required=True, help="音源名（assets/weights 下）或 pth 完整路径")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--f0-up-key", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"])
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--index-path", default="")
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-sec", type=float, default=300.0)
    parser.add_argument("--overlap-sec", type=float, default=0.05)
    parser.add_argument("--is-half", action="store_true", default=True)
    return parser.parse_args(argv)


class Heartbeat(threading.Thread):
    """推理期间持续上报进度（RVC 内部无回调，用墙上时间估算 25→85%）。"""

    def __init__(self, start, end, interval=1.0):
        super().__init__(daemon=True)
        self.start_value = float(start)
        self.end_value = float(end)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._current = self.start_value

    def run(self):
        while not self._stop.wait(self.interval):
            self._current = min(self.end_value, self._current + (self.end_value - self.start_value) / 30.0)
            progress(self._current, "infer", "RVC 推理中")

    def stop(self):
        self._stop.set()
        try:
            progress(min(self.end_value, self._current + 1.0), "infer", "RVC 推理结束")
        except Exception:
            pass


def resolve_voice(root, voice):
    candidate = Path(voice).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for name in (voice, voice + ".pth"):
        path = root / "assets" / "weights" / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("音源不存在：%s（也不在 %s/assets/weights 下）" % (voice, root))


def resolve_index(root, model_path):
    stem = model_path.stem
    for path in (model_path.with_suffix(".index"), root / "assets" / "indices" / (stem + ".index")):
        if path.is_file():
            return path.resolve()
    return None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = {"type": "result", "ok": False}
    heartbeat = None
    try:
        rvc_root = Path(args.rvc_root).expanduser().resolve()
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError("输入音频不存在：%s" % input_path)
        if not rvc_root.is_dir():
            raise FileNotFoundError("RVC 根目录不存在：%s" % rvc_root)

        progress(2, "resolve", "解析音源/index")
        voice_path = resolve_voice(rvc_root, args.voice)
        index_path = Path(args.index_path).expanduser() if args.index_path else resolve_index(rvc_root, voice_path)
        if index_path is not None and not index_path.is_file():
            raise FileNotFoundError("指定的 index 不存在：%s" % index_path)
        log("info", "音源：%s；index：%s" % (voice_path.name, index_path.name if index_path else "无"))
        if index_path is None:
            log("warning", "未找到 index：本次 index_rate 不生效（仍可换声，音色相似度略降）")

        os.chdir(str(rvc_root))
        sys.path.insert(0, str(rvc_root))
        sys.argv = sys.argv[:1]

        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(rvc_root / ".env"), override=False)
        for key, value in {
            "weight_root": "assets/weights",
            "index_root": "logs",
            "rmvpe_root": "assets/rmvpe",
            "outside_index_root": "assets/indices",
        }.items():
            os.environ.setdefault(key, value)

        import numpy as np
        import soundfile as sf
        import torch
        from configs.config import Config
        from infer.modules.vc.modules import VC

        config = Config()
        if args.device == "cpu":
            config.device = "cpu"
        elif args.device.startswith("cuda") and torch.cuda.is_available():
            config.device = args.device
        config.is_half = bool(args.is_half and str(config.device).startswith("cuda"))

        # faiss 在非 ASCII 路径下会 fopen 失败（RVC pipeline 会静默忽略 index），这里做兼容
        import faiss

        original_read_index = faiss.read_index

        def _read_index_unicode_safe(path, *call_args, **call_kwargs):
            try:
                return original_read_index(path, *call_args, **call_kwargs)
            except Exception:
                with open(path, "rb") as handle:
                    payload = handle.read()
                return faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))

        faiss.read_index = _read_index_unicode_safe

        progress(6, "load_model", "加载 RVC 模型（Hubert/RMVPE 在首次推理时加载）")
        os.environ["weight_root"] = str(voice_path.parent)
        vc = VC(config)
        vc.get_vc(voice_path.name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        progress(22, "load_audio", "加载并重采样输入音频")
        info = sf.info(str(input_path))
        total_input_sec = float(info.duration or 0.0)
        chunk_sec = float(getattr(args, "chunk_sec", 300.0) or 300.0)
        overlap_sec = float(getattr(args, "overlap_sec", 0.05) or 0.05)
        chunk_paths = [input_path]
        if total_input_sec > max(600.0, chunk_sec * 2):
            source, source_sr = sf.read(str(input_path), always_2d=False)
            source = np.asarray(source)
            if source.ndim > 1:
                source = source.mean(axis=1)
            chunk_dir = output_path.parent / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            step = max(1, int(chunk_sec * source_sr))
            overlap = max(1, int(overlap_sec * source_sr))
            chunk_paths = []
            for index, start in enumerate(range(0, len(source), step)):
                end = min(len(source), start + step + overlap)
                part = chunk_dir / ("part_%03d.wav" % index)
                sf.write(str(part), source[start:end], source_sr, subtype="PCM_24")
                chunk_paths.append(part)
            log("info", "长音频分段：%d 段，总长 %.1f 秒" % (len(chunk_paths), total_input_sec))

        parts = []
        total_infer = 0.0
        for index, chunk in enumerate(chunk_paths):
            progress(24 + 55.0 * index / max(1, len(chunk_paths)), "infer", "分段 %d/%d" % (index + 1, len(chunk_paths)))
            heartbeat = Heartbeat(25.0, 85.0)
            heartbeat.start()
            started = time.perf_counter()
            rvc_info, wav_opt = vc.vc_single(
                0,
                str(chunk),
                int(args.f0_up_key),
                None,
                str(args.f0_method),
                str(index_path) if index_path else "",
                None,
                float(args.index_rate),
                int(args.filter_radius),
                int(args.resample_sr),
                float(args.rms_mix_rate),
                float(args.protect),
            )
            total_infer += time.perf_counter() - started
            heartbeat.stop()
            heartbeat = None
            if wav_opt is None or wav_opt[0] is None:
                raise RuntimeError("RVC 推理未返回音频：%s" % str(rvc_info)[:500])
            parts.append((int(wav_opt[0]), np.asarray(wav_opt[1])))

        infer_sec = total_infer
        if len(parts) == 1:
            sample_rate, array = parts[0]
        else:
            sample_rate = parts[0][0]
            if any(item[0] != sample_rate for item in parts):
                raise RuntimeError("RVC 分段输出采样率不一致")
            fade = max(1, int(overlap_sec * sample_rate))
            merged = parts[0][1]
            for _, nxt in parts[1:]:
                n = min(fade, len(merged), len(nxt))
                if n > 1:
                    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    merged = merged.copy()
                    merged[-n:] = merged[-n:] * (1.0 - t) + nxt[:n] * t
                    merged = np.concatenate([merged, nxt[n:]])
                else:
                    merged = np.concatenate([merged, nxt])
            array = merged
        array = np.asarray(array)
        if array.dtype.kind in ("i", "u"):
            array = array.astype(np.float32) / (float(np.iinfo(array.dtype).max) + 1.0)
        else:
            array = array.astype(np.float32)
        target_len = int(round(total_input_sec * sample_rate)) if sample_rate else 0
        if target_len > 0 and len(array) != target_len:
            if len(array) > target_len:
                array = array[:target_len]
            else:
                array = np.pad(array, (0, target_len - len(array)))
        peak_before = float(np.max(np.abs(array))) if len(array) else 0.0
        normalized = False
        if peak_before > 0.99:
            array = array * (0.98 / peak_before)
            normalized = True
        rms = float(np.sqrt(np.mean(np.square(array)))) if len(array) else 0.0
        duration = len(array) / float(sample_rate) if sample_rate else 0.0

        progress(92, "save", "写出 wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), array, sample_rate, subtype="PCM_24")

        meta = {
            "engine": "rvc",
            "chunks": len(chunk_paths),
            "chunk_sec": float(getattr(args, "chunk_sec", 300.0) or 300.0),
            "rvc_info": str(rvc_info),
            "voice": voice_path.name,
            "index_used": bool(index_path),
            "f0_up_key": int(args.f0_up_key),
            "index_rate": float(args.index_rate),
            "f0_method": str(args.f0_method),
            "filter_radius": int(args.filter_radius),
            "resample_sr": int(args.resample_sr),
            "rms_mix_rate": float(args.rms_mix_rate),
            "protect": float(args.protect),
            "device": str(config.device),
            "infer_sec": round(infer_sec, 3),
            "rtf": round(infer_sec / duration, 4) if duration else None,
            "sample_rate": sample_rate,
            "duration_sec": round(duration, 4),
            "peak": round(float(np.max(np.abs(array))) if len(array) else 0.0, 6),
            "peak_before": round(peak_before, 6),
            "normalized": normalized,
            "rms_dbfs": round(20.0 * np.log10(max(rms, 1e-9)), 3),
            "input_duration_sec": round(total_input_sec, 4),
        }
        if torch.cuda.is_available():
            meta["vram_peak_alloc_mb"] = int(torch.cuda.max_memory_allocated() / 1024 ** 2)
        meta_path = output_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"type": "artifact", "name": output_path.name, "path": str(output_path), "kind": "audio", "meta": meta})
        emit({"type": "artifact", "name": meta_path.name, "path": str(meta_path), "kind": "report", "meta": {}})
        progress(100, "done")
        result = {"type": "result", "ok": True, "meta": meta}
    except Exception as exc:  # noqa: BLE001 - 需要把错误完整送回 Gateway
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except Exception:
                pass
        detail = traceback.format_exc()
        log("error", "换声失败：%s" % exc)
        result = {
            "type": "result",
            "ok": False,
            "error": {
                "code": "RVC_CONVERT_FAILED",
                "message": str(exc),
                "detail": {"traceback": detail[-2000:]},
                "hint": "检查 AMS_RVC_PYTHON/AMS_RVC_ROOT、音源是否存在、音频是否可解码、显存是否足够",
            },
        }
    finally:
        emit(result)
        return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
