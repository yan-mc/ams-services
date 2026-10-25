#!/usr/bin/env python3
"""AMS D2: controlled RVC training pipeline.
Run with RVC runtime Python from the RVC root; emits AMS JSONL protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def progress(percent, stage, message=None):
    item = {"type": "progress", "percent": round(float(percent), 1), "stage": stage}
    if message:
        item["message"] = str(message)[:300]
    emit(item)


def run(command, cwd, stage, start, end):
    progress(start, stage)
    proc = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    last = start
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        if stage == "train" and "Epoch:" in line:
            try:
                epoch = int(line.split("Epoch:")[1].strip().split()[0])
                total = max(1, int(os.environ.get("AMS_TRAIN_EPOCHS", "100")))
                last = min(end - 1, start + (end - start) * epoch / total)
                progress(last, "train", line[:200])
            except Exception:
                pass
    code = proc.wait()
    if code != 0:
        raise RuntimeError("%s failed with exit code %s" % (stage, code))
    progress(end, stage)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--sample-rate", default="48k", choices=["32k", "40k", "48k"])
    parser.add_argument("--if-f0", type=int, default=1)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--voice-dir", required=True)
    args = parser.parse_args()

    root = Path(args.rvc_root).resolve()
    dataset = Path(args.dataset_dir).resolve()
    voice_dir = Path(args.voice_dir).resolve()
    exp = root / "logs" / args.exp_name
    exp.mkdir(parents=True, exist_ok=True)
    if not dataset.is_dir():
        raise RuntimeError("dataset_dir not found: " + str(dataset))
    if not any(dataset.rglob("*.wav")):
        raise RuntimeError("dataset contains no wav: " + str(dataset))
    os.environ["AMS_TRAIN_EPOCHS"] = str(args.epochs)
    os.chdir(root)
    py = sys.executable
    progress(2, "validate", "dataset ok")

    run([py, "infer/modules/train/preprocess.py", str(dataset), {"32k":"32000","40k":"40000","48k":"48000"}[args.sample_rate], "1", str(exp), "True", "3.7"], root, "preprocess", 5, 20)

    if args.if_f0:
        run([py, "infer/modules/train/extract/extract_f0_print.py", str(exp), "1", args.f0_method], root, "f0", 20, 35)
    run([py, "infer/modules/train/extract_feature_print.py", "cuda:0", "1", "0", str(exp), args.version, "true"], root, "feature", 35, 50)

    config_src = root / "configs" / ("v1" if (args.version == "v1" or args.sample_rate == "40k") else "v2") / (args.sample_rate + ".json")
    if not config_src.is_file():
        raise RuntimeError("RVC config not found: " + str(config_src))
    (exp / "config.json").write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")
    gt = exp / "0_gt_wavs"
    feat = exp / ("3_feature256" if args.version == "v1" else "3_feature768")
    f0 = exp / "2a_f0"
    nsf = exp / "2b-f0nsf"
    names = sorted({p.name.split(".")[0] for p in gt.glob("*.wav")} & {p.name.split(".")[0] for p in feat.glob("*.npy")})
    if args.if_f0:
        names = sorted(set(names) & {p.name.split(".")[0] for p in f0.glob("*.npy")} & {p.name.split(".")[0] for p in nsf.glob("*.npy")})
    if not names:
        raise RuntimeError("no matched training samples")
    lines = []
    for name in names:
        if args.if_f0:
            lines.append("%s|%s|%s|%s|0" % (gt / (name + ".wav"), feat / (name + ".npy"), f0 / (name + ".wav.npy"), nsf / (name + ".wav.npy")))
        else:
            lines.append("%s|%s|0" % (gt / (name + ".wav"), feat / (name + ".npy")))
    mute = root / "logs" / "mute"
    if args.if_f0:
        m = "%s|%s|%s|%s|0" % (mute / "0_gt_wavs" / ("mute" + args.sample_rate + ".wav"), mute / ("3_feature256" if args.version == "v1" else "3_feature768") / "mute.npy", mute / "2a_f0" / "mute.wav.npy", mute / "2b-f0nsf" / "mute.wav.npy")
    else:
        m = "%s|%s|0" % (mute / "0_gt_wavs" / ("mute" + args.sample_rate + ".wav"), mute / ("3_feature256" if args.version == "v1" else "3_feature768") / "mute.npy")
    lines.extend([m, m])
    (exp / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    pretrain_dir = root / "assets" / ("pretrained_v2" if args.version == "v2" else "pretrained")
    pg = pretrain_dir / (("f0G" if args.if_f0 else "G") + args.sample_rate + ".pth")
    pd = pretrain_dir / (("f0D" if args.if_f0 else "D") + args.sample_rate + ".pth")
    effective_batch = args.batch_size
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < 6 and effective_batch > 1:
                effective_batch = 1
            elif free_gb < 10 and effective_batch > 2:
                effective_batch = 2
            if effective_batch != args.batch_size:
                emit({"type": "log", "level": "warning", "message": "GPU free %.1f GB; batch %s -> %s" % (free_gb, args.batch_size, effective_batch)})
    except Exception:
        pass
    command = [py, "infer/modules/train/train.py", "-e", args.exp_name, "-sr", args.sample_rate, "-f0", str(args.if_f0), "-bs", str(effective_batch), "-g", args.gpu, "-te", str(args.epochs), "-se", str(args.save_every), "-l", "1", "-c", "0", "-sw", "1", "-v", args.version]
    if pg.is_file(): command.extend(["-pg", str(pg)])
    if pd.is_file(): command.extend(["-pd", str(pd)])
    run(command, root, "train", 50, 85)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from infer.lib.train.process_ckpt import extract_small_model
    extract_small_model(str(exp / "G_2333333.pth"), args.model_name, args.sample_rate, args.if_f0, "%sepoch" % args.epochs, args.version)
    pth = root / "assets" / "weights" / (args.model_name + ".pth")
    if not pth.is_file():
        candidates = sorted((root / "assets" / "weights").glob(args.model_name + "*.pth"))
        if not candidates:
            raise RuntimeError("trained pth not found")
        pth = candidates[-1]
    progress(88, "index", "building FAISS index")
    import faiss
    import numpy as np
    fs = sorted(feat.glob("*.npy"))
    arr = np.concatenate([np.load(str(f)) for f in fs]).astype("float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    index_path = exp / (args.model_name + ".index")
    faiss.write_index(index, str(index_path))

    voice_dir.mkdir(parents=True, exist_ok=True)
    target = voice_dir / args.model_name
    if target.exists():
        raise RuntimeError("voice already exists: " + str(target))
    target.mkdir(parents=True)
    import shutil
    shutil.copy2(pth, target / (args.model_name + ".pth"))
    shutil.copy2(index_path, target / (args.model_name + ".index"))
    (target / "voice.json").write_text(json.dumps({"name": args.model_name, "source": "dataset:" + str(dataset), "version": args.version, "sample_rate": args.sample_rate, "epochs": args.epochs, "f0": bool(args.if_f0), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(100, "done")
    emit({"type": "artifact", "name": args.model_name + ".pth", "path": str(target / (args.model_name + ".pth")), "kind": "model"})
    emit({"type": "artifact", "name": args.model_name + ".index", "path": str(target / (args.model_name + ".index")), "kind": "index"})
    emit({"type": "result", "ok": True, "meta": {"voice": str(target), "exp": str(exp)}})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"type": "result", "ok": False, "error": {"code": "TRAIN_FAILED", "message": str(exc)[:1000]}})
        raise
