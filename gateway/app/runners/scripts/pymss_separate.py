# -*- coding: utf-8 -*-
"""pymss 分离后端脚本（由 pymss 独立环境 python 3.12 执行）。

stdout 每行一个 JSON 消息（progress/log/artifact/result），供 Gateway Runner 解析。
模型目录默认取 --model-dir（Gateway 传 <AMS_PYMSS_ROOT>/models）；
不修改模型文件，产物写入 --output-dir。

用法：
  <AMS_PYMSS_PYTHON> -B pymss_separate.py --input in.wav --model scnet_checkpoint_musdb18 ^
      --device auto --output-dir out --model-dir models [--chunk 485100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_EMIT_LOCK = None


def emit(payload):
    line = json.dumps(payload, ensure_ascii=False)
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
    parser = argparse.ArgumentParser(description="pymss 分离后端")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--stem-batch-size", type=int, default=None)
    parser.add_argument("--normalize", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--standardize", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--stems", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = {"type": "result", "ok": False}
    separator = None
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError("输入音频不存在：%s" % input_path)
        model_dir = Path(args.model_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        progress(2, "load_model", "加载 pymss 引擎与模型：%s" % args.model)
        from pymss import MSSeparator
        from pymss.audio_io import load_audio, save_audio

        inference_params = {}
        if args.chunk:
            inference_params["chunk_size"] = int(args.chunk)
        if args.overlap_size is not None:
            inference_params["overlap_size"] = int(args.overlap_size)
        if args.stem_batch_size is not None:
            inference_params["stem_batch_size"] = int(args.stem_batch_size)
        if args.normalize != "auto":
            inference_params["normalize"] = args.normalize == "true"
        if args.standardize != "auto":
            inference_params["standardize"] = args.standardize == "true"
        stems = [item.strip() for item in str(args.stems or "").split(",") if item.strip()] or None

        def on_progress(done, total, message):
            total = max(1, int(total or 1))
            done = max(0, min(int(done), total))
            percent = 10.0 + (done / float(total)) * 80.0
            progress(percent, "separate", str(message or "separating"))

        started = time.perf_counter()
        separator = MSSeparator.from_model_name(
            args.model,
            model_dir=str(model_dir),
            download=False,
            device=args.device,
            output_format="wav",
            audio_params={"wav_bit_depth": "FLOAT"},
            inference_params=inference_params,
            progress_callback=on_progress,
        )
        progress(12, "load_audio", "加载/重采样输入音频")
        sample_rate = int(separator.config.audio.sample_rate)
        mix, _source_rate = load_audio(str(input_path), sr=sample_rate, mono=False, downmix_stereo=True)
        duration = float(mix.shape[-1]) / float(sample_rate) if getattr(mix, "size", 0) else 0.0
        log("info", "输入 %.2fs @ %dHz，开始分离（device=%s）" % (duration, sample_rate, separator.device))

        results = separator.separate(mix, pbar=False, stems=stems)
        progress(92, "save", "写出 stems")
        base_name = input_path.stem
        artifacts = []
        stems = []
        for stem, audio in results.items():
            output_path = output_dir / ("%s_%s.wav" % (base_name, stem))
            save_audio(str(output_path), audio, sample_rate, "wav", {"wav_bit_depth": "FLOAT"})
            stems.append(stem)
            artifact_meta = {
                "stem": stem,
                "sample_rate": sample_rate,
                "channels": int(audio.shape[1]) if getattr(audio, "ndim", 1) > 1 else 1,
                "duration_sec": round(len(audio) / float(sample_rate), 4),
            }
            emit({"type": "artifact", "name": output_path.name, "path": str(output_path), "kind": "audio", "meta": artifact_meta})
            artifacts.append(str(output_path))

        elapsed = time.perf_counter() - started
        manifest = {
            "engine": "pymss",
            "model": args.model,
            "device": str(separator.device),
            "chunk_size": inference_params.get("chunk_size"),
            "sample_rate": sample_rate,
            "input": str(input_path),
            "duration_sec": round(duration, 4),
            "stems": stems,
            "elapsed_sec": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration else None,
            "artifacts": artifacts,
        }
        manifest_path = output_dir / "separation-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"type": "artifact", "name": manifest_path.name, "path": str(manifest_path), "kind": "report", "meta": {}})
        progress(100, "done")
        result = {"type": "result", "ok": True, "meta": manifest}
    except Exception as exc:  # noqa: BLE001
        log("error", "分离失败：%s" % exc)
        result = {
            "type": "result",
            "ok": False,
            "error": {
                "code": "PYMSS_SEPARATE_FAILED",
                "message": str(exc),
                "detail": {"traceback": traceback.format_exc()[-2000:]},
                "hint": "检查 AMS_PYMSS_PYTHON/AMS_PYMSS_ROOT、模型是否已下载、显存是否足够；CPU 回退可设 device=cpu",
            },
        }
    finally:
        try:
            if separator is not None:
                separator.close()
        except Exception:
            pass
        emit(result)
        return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
