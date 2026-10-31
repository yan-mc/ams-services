#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Headless GPT-SoVITS V5turbo training driver (AMS Gateway -> runtime).

Stages: build train.list -> text/hubert/semantic preprocessing -> SoVITS LoRA ->
GPT training -> copy weights/ref -> write voice.json.

Only v5turbo is enabled in this P2 version; V2ProPlus full fine-tune comes later.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import wave

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


def emit(kind: str, **kwargs) -> None:
    payload = {"type": kind, **kwargs}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def progress(percent: float, stage: str, message: str = "") -> None:
    emit("progress", percent=max(0.0, min(100.0, float(percent))), stage=stage, message=message)


def log(level: str, message: str) -> None:
    emit("log", level=level, message=message)


def wav_duration(path: pathlib.Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:  # noqa: BLE001
        return None


def read_metadata(dataset: pathlib.Path) -> dict[str, str]:
    import csv as _csv
    out: dict[str, str] = {}
    for name in ("metadata.csv", "metadata.cleaned.csv"):
        p = dataset / name
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            if "|" in line:
                parts = line.split("|")
            else:
                parts = next(_csv.reader([line]))
            if len(parts) < 2:
                continue
            key = pathlib.Path(parts[0].strip()).stem
            text = " ".join("|".join(parts[1:]).strip().split())
            if key and text:
                out[key] = text
        return out
    for name in ("xianyun.list", "train.list"):
        p = dataset / name
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                out[pathlib.Path(parts[0]).stem] = parts[3].strip()
        return out
    return out


def build_train_list(dataset: pathlib.Path, wav_dir: pathlib.Path, voice_id: str, out_dir: pathlib.Path) -> tuple[pathlib.Path, int]:
    meta = read_metadata(dataset)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wav under {wav_dir}")
    rows = []
    for wav in wavs:
        text = meta.get(wav.stem) or meta.get(wav.name)
        if not text:
            continue
        rows.append(f"{wav.resolve()}|{voice_id}|zh|{text}")
    if len(rows) < 20:
        raise SystemExit(f"dataset too small: {len(rows)} labeled clips (need >=20 for P2 smoke)")
    list_path = out_dir / "train.list"
    list_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return list_path, len(rows)


def pick_reference(dataset: pathlib.Path, wav_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    meta = read_metadata(dataset)
    for wav in sorted(wav_dir.glob("*.wav")):
        text = meta.get(wav.stem, "") or meta.get(wav.name, "")
        dur = wav_duration(wav)
        if text and dur and 3.0 <= dur <= 10.0:
            return wav, text
    raise SystemExit("no 3-10s reference clip with metadata")


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = pathlib.Path(args.runtime_root).resolve()
        self.home = pathlib.Path(args.home).resolve()
        self.dataset = pathlib.Path(args.dataset_dir).resolve()
        self.voice_id = args.voice_id
        if not SAFE_ID.match(self.voice_id):
            raise SystemExit("invalid voice_id")
        self.wav_dir = self.dataset / "wav" if (self.dataset / "wav").is_dir() else self.dataset
        if not self.dataset.is_dir() or not self.wav_dir.is_dir():
            raise SystemExit(f"dataset not found: {self.dataset}")
        self.python = self.root / "runtime" / "python.exe"
        if not self.python.is_file():
            raise SystemExit(f"runtime python not found: {self.python}")
        self.s2_script = self.root / "tts_service" / "support" / "s2_train_v5_lora_nw0.py"
        self.s1_script = self.root / "tts_service" / "support" / "s1_train_v5_single.py"
        for p in (self.s2_script, self.s1_script):
            if not p.is_file():
                raise SystemExit(f"training support script missing: {p}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out = self.home / "training" / f"{self.voice_id}-{stamp}"
        self.out.mkdir(parents=True, exist_ok=False)
        self.logs = self.out / "logs"
        self.logs.mkdir()
        self.tmp = self.home / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        self.env.update({
            "TEMP": str(self.tmp), "TMP": str(self.tmp),
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        })

    def run_script(self, name: str, script: pathlib.Path, extra_args: list[str], env_extra: dict[str, str] | None = None,
                   cwd: pathlib.Path | None = None) -> None:
        log_path = self.logs / f"{name}.log"
        log("info", f"[train] start {name}")
        env = dict(self.env)
        if env_extra:
            env.update(env_extra)
        with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                [str(self.python), "-u", "-B", str(script), *extra_args],
                cwd=str(cwd or self.root), env=env,
                stdout=fh, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                timeout=self.args.timeout_sec,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.is_file() else ""
            raise SystemExit(f"{name} failed with code {proc.returncode}:\n{tail}")
        log("info", f"[train] done {name}")

    def prep(self) -> None:
        list_path, count = build_train_list(self.dataset, self.wav_dir, self.voice_id, self.out)
        log("info", f"[train] dataset clips={count}")
        exp = self.out / "exp"
        exp.mkdir(exist_ok=True)
        prep_env = {
            "inp_text": str(list_path), "inp_wav_dir": str(self.wav_dir),
            "exp_name": self.voice_id, "opt_dir": str(exp),
            "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "True", "version": "v5turbo",
            "bert_pretrained_dir": str(self.root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhubert_base_dir": str(self.root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
            "pretrained_s2G": str(self.root / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained" / "s2Gv5turbo.pth"),
            "s2config_path": str(self.root / "GPT_SoVITS" / "configs" / "s2.json"),
        }
        progress(10, "prep", "text")
        self.run_script("1-get-text", self.root / "GPT_SoVITS" / "prepare_datasets" / "1-get-text.py", [], prep_env)
        part = exp / "2-name2text-0.txt"
        if part.is_file():
            (exp / "2-name2text.txt").write_text(part.read_text(encoding="utf-8"), encoding="utf-8")
        progress(20, "prep", "hubert")
        self.run_script("2-get-hubert", self.root / "GPT_SoVITS" / "prepare_datasets" / "2-get-hubert-wav32k.py", [], prep_env)
        progress(35, "prep", "semantic")
        self.run_script("3-get-semantic", self.root / "GPT_SoVITS" / "prepare_datasets" / "3-get-semantic.py", [], prep_env)
        part_sem = exp / "6-name2semantic-0.tsv"
        if not part_sem.is_file():
            raise SystemExit("semantic output missing")
        lines = [x for x in part_sem.read_text(encoding="utf-8").splitlines() if x.strip()]
        (exp / "6-name2semantic.tsv").write_text("item_name\tsemantic_audio\n" + "\n".join(lines) + "\n", encoding="utf-8")
        self.exp = exp
        log("info", f"[train] prep done clips={count}")

    def sovits(self) -> pathlib.Path:
        cfg = json.loads((self.root / "GPT_SoVITS" / "configs" / "s2.json").read_text(encoding="utf-8"))
        weights = self.out / "weights_sovits"
        weights.mkdir(exist_ok=True)
        cfg["train"].update({
            "batch_size": self.args.batch_size_sovits,
            "epochs": self.args.epochs_sovits,
            "pretrained_s2G": str(self.root / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained" / "s2Gv5turbo.pth"),
            "pretrained_s2D": "",
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": 1,
            "gpu_numbers": "0",
            "grad_ckpt": False,
            "lora_rank": self.args.lora_rank,
            "text_low_lr_rate": 0.4,
        })
        cfg["model"]["version"] = "v5turbo"
        cfg["data"]["exp_dir"] = str(self.out / "exp")
        cfg["s2_ckpt_dir"] = str(self.out / "exp")
        cfg["save_weight_dir"] = str(weights)
        cfg["name"] = self.voice_id
        cfg["version"] = "v5turbo"
        cfg_path = self.out / "s2.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        progress(45, "sovits", f"{self.args.epochs_sovits} epochs")
        self.run_script("sovits-lora", self.s2_script, ["--config", str(cfg_path)],
                        {"_CUDA_VISIBLE_DEVICES": "0"})
        candidates = sorted(weights.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit("no SoVITS weights produced")
        return candidates[-1]

    def gpt(self) -> pathlib.Path:
        import yaml
        cfg = yaml.safe_load((self.root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").read_text(encoding="utf-8"))
        weights = self.out / "weights_gpt"
        weights.mkdir(exist_ok=True)
        cfg["train"].update({
            "batch_size": self.args.batch_size_gpt,
            "epochs": self.args.epochs_gpt,
            "precision": "16-mixed",
            "save_every_n_epoch": max(1, self.args.epochs_gpt),
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": str(weights),
            "exp_name": self.voice_id,
        })
        cfg["data"]["num_workers"] = 0
        cfg["pretrained_s1"] = str(self.root / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt")
        cfg["train_semantic_path"] = str(self.exp / "6-name2semantic.tsv")
        cfg["train_phoneme_path"] = str(self.exp / "2-name2text.txt")
        cfg["output_dir"] = str(self.out / "logs_s1_v5turbo")
        cfg_path = self.out / "s1.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        progress(75, "gpt", f"{self.args.epochs_gpt} epochs")
        self.run_script("gpt", self.s1_script, ["--config_file", str(cfg_path)],
                        {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"})
        candidates = sorted(weights.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit("no GPT weights produced")
        return candidates[-1]

    def finalize(self, sovits: pathlib.Path, gpt: pathlib.Path) -> pathlib.Path:
        voice_dir = self.home / "voices" / self.voice_id
        if voice_dir.exists():
            raise SystemExit(f"voice already exists (refuse overwrite): {voice_dir}")
        (voice_dir / "weights").mkdir(parents=True)
        (voice_dir / "ref").mkdir(parents=True)
        shutil.copy2(sovits, voice_dir / "weights" / sovits.name)
        shutil.copy2(gpt, voice_dir / "weights" / gpt.name)
        ref_src = pathlib.Path(self.args.ref_audio).resolve() if self.args.ref_audio else pick_reference(self.dataset, self.wav_dir)[0]
        ref_text = str(self.args.ref_text or "").strip() or pick_reference(self.dataset, self.wav_dir)[1]
        shutil.copy2(ref_src, voice_dir / "ref" / "ref.wav")
        manifest = {
            "id": self.voice_id,
            "name": self.args.name or self.voice_id,
            "version": "v5turbo",
            "language": self.args.language,
            "sovits": f"weights/{sovits.name}",
            "gpt": f"weights/{gpt.name}",
            "ref_audio": "ref/ref.wav",
            "ref_text": ref_text,
            "tags": ["gpt-sovits", "v5turbo"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "training": {"dataset": str(self.dataset), "out": str(self.out),
                         "sovits_epochs": self.args.epochs_sovits, "gpt_epochs": self.args.epochs_gpt,
                         "lora_rank": self.args.lora_rank},
        }
        (voice_dir / "voice.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return voice_dir / "voice.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--voice-id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--ref-audio", default="")
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--version", default="v5turbo")
    ap.add_argument("--epochs-sovits", type=int, default=8)
    ap.add_argument("--epochs-gpt", type=int, default=10)
    ap.add_argument("--batch-size-sovits", type=int, default=1)
    ap.add_argument("--batch-size-gpt", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--timeout-sec", type=int, default=21600)
    args = ap.parse_args()
    if args.version != "v5turbo":
        raise SystemExit("P2 training driver supports v5turbo only; v2ProPlus comes later")
    for name, lo, hi in (("epochs_sovits", 1, 16), ("epochs_gpt", 1, 20),
                         ("batch_size_sovits", 1, 4), ("batch_size_gpt", 1, 8),
                         ("lora_rank", 8, 128)):
        value = getattr(args, name)
        if not lo <= value <= hi:
            raise SystemExit(f"{name} out of range {lo}..{hi}")
    trainer = Trainer(args)
    progress(2, "init", f"out={trainer.out}")
    trainer.prep()
    sovits = trainer.sovits()
    gpt = trainer.gpt()
    voice_json = trainer.finalize(sovits, gpt)
    progress(100, "done", str(voice_json))
    emit("result", ok=True, artifacts=[{
        "name": "voice.json", "path": str(voice_json), "kind": "file",
        "meta": {"voice_id": args.voice_id, "version": "v5turbo", "training_dir": str(trainer.out)},
    }])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
