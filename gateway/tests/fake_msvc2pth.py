# -*- coding: utf-8 -*-
"""测试用 fake msvc2pth：模拟 CLI 与 --json 报告输出（不依赖 torch/RVC）。

用法与 msvc2pth.py 一致（子集）：
  python fake_msvc2pth.py <input.msvc> --out <dir> [--name X] [--reference-pth P] --json

环境变量 FAKE_MSVC_MODE=fail 时输出错误并返回 1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input")
    parser.add_argument("--out", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--reference-pth", dest="reference_pth", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args, _unknown = parser.parse_known_args()

    if os.environ.get("FAKE_MSVC_MODE") == "fail":
        print("fake msvc2pth: 故意失败", file=sys.stderr)
        return 1

    print("[cache] fake calibration cache hit")
    print("[calibrate] fake 数据分片边界搜索 ...")
    out_dir = os.path.abspath(args.out or os.path.dirname(os.path.abspath(args.input)))
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    voice_dir = os.path.join(out_dir, name)
    os.makedirs(voice_dir, exist_ok=True)
    pth_path = os.path.join(voice_dir, name + ".pth")
    with open(pth_path, "wb") as handle:
        handle.write(b"FAKE_PTH_CONTENT")
    report = {
        "tool": "ams-tools/msvc2pth-fake",
        "tool_version": "fake",
        "source": os.path.abspath(args.input),
        "voice_name": name,
        "output_pth": os.path.abspath(pth_path),
        "output_size": os.path.getsize(pth_path),
        "tensor_count": 457,
        "reference_pth": args.reference_pth,
        "warnings": ["fake 后端：仅用于测试"],
    }
    report_path = args.report or os.path.join(voice_dir, "conversion.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("OK ->", pth_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
