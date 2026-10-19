# -*- coding: utf-8 -*-
"""对比两套 pymss 输出（CPU vs GPU）：逐 stem 相关性/能量/差异/时长。

用法：
  <pymss venv python> -B compare_stems.py --a <cpu_dir> --b <gpu_dir> --result-json out.json
目录结构支持 pymss --save-as-folder 产物（<out>/<name>/<name>_<stem>.wav）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf


def collect(root: str) -> dict:
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    result = {}
    for path in files:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        for prefix in ("input_music_20s_", "input_music_60s_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        result[stem] = path
    return result


def load(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio, sr


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="基准目录（CPU）")
    parser.add_argument("--b", required=True, help="对比目录（GPU）")
    parser.add_argument("--label-a", default="cpu")
    parser.add_argument("--label-b", default="gpu")
    parser.add_argument("--result-json", default="")
    args = parser.parse_args(argv)

    a_files = collect(args.a)
    b_files = collect(args.b)
    stems = sorted(set(a_files) & set(b_files))
    if not stems:
        print(json.dumps({"error": "没有可匹配的 stem", "a": a_files, "b": b_files}, ensure_ascii=False))
        return 1

    result = {
        "schema": "ams.stems_compare/1",
        "dir_a": args.a,
        "dir_b": args.b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "stems": {},
    }
    for stem in stems:
        a_audio, a_sr = load(a_files[stem])
        b_audio, b_sr = load(b_files[stem])
        n = min(len(a_audio), len(b_audio))
        a = a_audio[:n].mean(axis=1)
        b = b_audio[:n].mean(axis=1)
        if a.size == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
        diff = a - b
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        item = {
            "a_file": a_files[stem],
            "b_file": b_files[stem],
            "sample_rate_a": a_sr,
            "sample_rate_b": b_sr,
            "duration_sec_a": round(len(a_audio) / a_sr, 4),
            "duration_sec_b": round(len(b_audio) / b_sr, 4),
            "duration_diff_ms": round(abs(len(a_audio) / a_sr - len(b_audio) / b_sr) * 1000.0, 2),
            "corr": round(corr, 6) if corr is not None else None,
            "rms_a": round(rms_a, 6),
            "rms_b": round(rms_b, 6),
            "rms_ratio_b_over_a": round(rms_b / rms_a, 4) if rms_a > 0 else None,
            "rms_diff": round(float(np.sqrt(np.mean(diff ** 2))), 6),
            "mean_abs_diff": round(float(np.mean(np.abs(diff))), 6),
            "max_abs_diff": round(float(np.max(np.abs(diff))), 6),
        }
        result["stems"][stem] = item
        print(f"{stem:10s} corr={item['corr']} rms_a={item['rms_a']} rms_b={item['rms_b']} "
              f"mean|d|={item['mean_abs_diff']} max|d|={item['max_abs_diff']} dur_diff={item['duration_diff_ms']}ms")

    corrs = [v["corr"] for v in result["stems"].values() if v.get("corr") is not None]
    result["summary"] = {
        "stem_count": len(result["stems"]),
        "corr_min": round(min(corrs), 6) if corrs else None,
        "corr_mean": round(float(np.mean(corrs)), 6) if corrs else None,
        "max_abs_diff_max": max((v["max_abs_diff"] for v in result["stems"].values()), default=None),
    }
    print(json.dumps(result["summary"], ensure_ascii=False))
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
