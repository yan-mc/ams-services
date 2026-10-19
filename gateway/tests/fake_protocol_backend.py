# -*- coding: utf-8 -*-
"""测试用 fake 后端：模拟 Gateway Runner 的 JSONL 协议（不依赖真实模型）。

模式（环境变量 FAKE_PROTOCOL_MODE）：
- success：进度 10/50/100，写若干 fake 音频产物，返回 ok=true
- fail：返回 ok=false（code=FAKE_BACKEND_FAILED），退出码 1
- cancel：写 pid 文件后长时间 sleep，用于验证取消杀进程树

Runner 会以不同参数调用本脚本（rvc/pymss 两套 CLI），因此这里用
parse_known_args 忽略未知参数，输出目录取 --output-dir 或 --output 的父目录。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--rvc-root", dest="rvc_root", default=None)
    parser.add_argument("--f0-up-key", dest="f0_up_key", default="0")
    parser.add_argument("--index-rate", dest="index_rate", default="0.75")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunk", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--model-dir", dest="model_dir", default=None)
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    mode = os.environ.get("FAKE_PROTOCOL_MODE", "success")
    stems = int(os.environ.get("FAKE_PROTOCOL_STEMS", "2"))
    output_dir = args.output_dir or (os.path.dirname(os.path.abspath(args.output)) if args.output else os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    if mode == "fail":
        emit({"type": "log", "level": "warning", "message": "fake backend 故意失败"})
        emit(
            {
                "type": "result",
                "ok": False,
                "error": {
                    "code": "FAKE_BACKEND_FAILED",
                    "message": "fake 后端失败（测试用）",
                    "hint": "检查 fake 模式",
                },
            }
        )
        return 1

    if mode == "cancel":
        pid_path = os.path.join(output_dir, "fake.pid")
        with open(pid_path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        emit({"type": "progress", "percent": 10.0, "stage": "fake.start"})
        emit({"type": "log", "level": "info", "message": "fake 后端等待取消"})
        time.sleep(60)
        emit({"type": "result", "ok": True})
        return 0

    emit({"type": "log", "level": "info", "message": "fake backend 启动"})
    emit({"type": "progress", "percent": 10.0, "stage": "fake.start"})
    time.sleep(0.05)
    emit({"type": "progress", "percent": 50.0, "stage": "fake.run"})
    for index in range(1, stems + 1):
        name = "fake-stem-%d.wav" % index
        path = os.path.join(output_dir, name)
        with open(path, "wb") as handle:
            handle.write(b"FAKEAUDIO" + bytes([index]))
        emit(
            {
                "type": "artifact",
                "name": name,
                "path": path,
                "kind": "audio",
                "meta": {"stem": "stem%d" % index},
            }
        )
    emit({"type": "progress", "percent": 100.0, "stage": "fake.done"})
    emit({"type": "result", "ok": True, "meta": {"mode": "success"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
