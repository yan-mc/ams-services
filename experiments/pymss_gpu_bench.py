# -*- coding: utf-8 -*-
"""pymss GPU/CPU 分离基准脚本（M1 / task-23）。

在 experiments/pymss 的独立 venv 中**进程内**执行 pymss CLI，采集：
- 每次运行的墙钟/CPU 时间、pymss 自报 RTF、CUDA 显存峰值（allocated/reserved）；
- 进程峰值工作集（psutil）；
- 设备信息（torch/cuda/GPU 名称/显存总量/CUDNN）；
结果输出 JSON（stdout + --result-json）。

用法：
  <pymss venv python> -B pymss_gpu_bench.py --model-dir <models> --input <wav> ^
     --output <outdir> --device cuda --batch-size 1 --runs 1 ^
     --result-json results/pymss_gpu_scnet.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time

import psutil
import torch

from pymss.cli import main as pymss_main

RTF_RE = re.compile(r"RTF=([0-9.]+)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="pymss 分离基准")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="scnet_checkpoint_musdb18")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--source", default="hf-mirror")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--result-json", default="")
    return parser.parse_args(argv)


def device_info(device: str) -> dict:
    info = {
        "device": device,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "vram_total_mb": int(props.total_memory / 1024 ** 2),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def main(argv=None) -> int:
    args = parse_args(argv)
    result = {
        "schema": "ams.pymss_bench/1",
        "model": args.model,
        "input": args.input,
        "output": args.output,
        "batch_size": args.batch_size,
        "runs": [],
        "device_info": device_info(args.device),
    }
    proc = psutil.Process()
    for run_index in range(1, args.runs + 1):
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        buffer = io.StringIO()
        started = time.perf_counter()
        cpu_started = time.process_time()
        exit_code = 0
        cli_args = [
            "infer",
            args.model,
            "-i",
            args.input,
            "-o",
            args.output,
            "--save-as-folder",
            "--device",
            args.device,
            "--format",
            "wav",
            "--wav-bit-depth",
            "FLOAT",
            "--model-dir",
            args.model_dir,
            "--source",
            args.source,
            "--param",
            f"batch_size={args.batch_size}",
        ]
        if not args.no_download:
            cli_args.append("--download")
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                pymss_main(cli_args)
        except SystemExit as exc:  # argparse/pymss 可能以 SystemExit 结束
            exit_code = int(exc.code or 0)
        wall = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_alloc = int(torch.cuda.max_memory_allocated() / 1024 ** 2)
            vram_reserved = int(torch.cuda.max_memory_reserved() / 1024 ** 2)
        else:
            vram_alloc = None
            vram_reserved = None
        text = buffer.getvalue()
        rtf_matches = RTF_RE.findall(text)
        run_info = {
            "run": run_index,
            "exit_code": exit_code,
            "wall_sec": round(wall, 3),
            "cpu_sec": round(cpu, 3),
            "cpu_ratio": round(cpu / wall, 2) if wall else None,
            "rtf_reported": float(rtf_matches[-1]) if rtf_matches else None,
            "vram_peak_alloc_mb": vram_alloc,
            "vram_peak_reserved_mb": vram_reserved,
            "peak_working_set_mb": round(proc.memory_info().peak_wset / 1024 ** 2, 1),
            "rss_mb": round(proc.memory_info().rss / 1024 ** 2, 1),
        }
        result["runs"].append(run_info)
        print(f"[run {run_index}] wall={run_info['wall_sec']}s cpu={run_info['cpu_sec']}s "
              f"rtf={run_info['rtf_reported']} vram={vram_alloc}MB", flush=True)
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
