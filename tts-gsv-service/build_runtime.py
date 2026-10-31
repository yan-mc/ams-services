#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble a portable runtime.tts-gsv-core from an extracted GPT-SoVITS package.

Usage (run from repo, not inside the target runtime):
  python build_runtime.py --source F:\\...\\GPT-SoVITS-v2pro-20261006 \
      --out F:\\AMS-Studio\\data\\components\\runtime.tts-gsv-core --version 0.1.0

The source package must contain runtime/, GPT_SoVITS/, tools/.
This script only copies/creates files; it never deletes existing directories.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".git", ".gitignore",
    "TEMP", "logs", "output", "SoVITS_weights*", "GPT_weights*",
    "*.log",
)
REQUIRED = ("runtime", "GPT_SoVITS", "tools")
PTH_LINES = [
    "..\\..\\..",
    "..\\..\\..\\GPT_SoVITS",
    "..\\..\\..\\GPT_SoVITS\\BigVGAN",
    "..\\..\\..\\tools",
    "..\\..\\..\\tools\\asr",
    "..\\..\\..\\tools\\uvr5",
]


def _copy_dir(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"missing source dir: {src}")
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing target: {dst}")
    shutil.copytree(src, dst, ignore=IGNORE)


def _write_pth(root: pathlib.Path) -> None:
    site = root / "runtime" / "Lib" / "site-packages"
    if not site.is_dir():
        site = root / "runtime" / "lib" / "site-packages"
    if not site.is_dir():
        raise SystemExit(f"site-packages not found under {root}")
    target = site / "ams_gsv_paths.pth"
    target.write_text("\r\n".join(PTH_LINES) + "\r\n", encoding="utf-8")
    print(f"wrote {target}")


def _count(root: pathlib.Path) -> tuple[int, int]:
    files = 0
    size = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            try:
                size += path.stat().st_size
            except OSError:
                pass
    return files, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--engine-default", default="v5turbo")
    ap.add_argument("--gpt-source-root", default="")
    args = ap.parse_args()
    source = pathlib.Path(args.source).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source package not found: {source}")
    for name in REQUIRED:
        if not (source / name).is_dir():
            raise SystemExit(f"source package missing {name}/")
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing out: {out}")
    out.mkdir(parents=True)
    for name in REQUIRED:
        _copy_dir(source / name, out / name)
    service_src = pathlib.Path(__file__).resolve().parent
    service_dst = out / "tts_service"
    service_dst.mkdir(parents=True)
    shutil.copy2(service_src / "tts_service.py", service_dst / "tts_service.py")
    shutil.copy2(service_src / "tts_train_service.py", service_dst / "tts_train_service.py")
    support_src = service_src / "support"
    if support_src.is_dir():
        shutil.copytree(support_src, service_dst / "support")
    _write_pth(out)
    files, size = _count(out)
    manifest = {
        "schema": "ams.runtime.tts-gsv-core/1",
        "id": "runtime.tts-gsv-core",
        "version": args.version,
        "engine_default": args.engine_default,
        "source_package": str(source),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": files,
        "bytes": size,
        "layout": {
            "python": "runtime/python.exe",
            "service": "tts_service/tts_service.py",
            "code": "GPT_SoVITS",
            "tools": "tools",
        },
        "env": {
            "AMS_TTS_GSV_ROOT": "<component root>",
            "AMS_TTS_GSV_HOME": "<AMS_HOME>/tts-gsv",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"runtime built: {out} files={files} bytes={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
