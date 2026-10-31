"""tts.dataset.prepare.gsv: import an existing GPT-SoVITS dataset (wav + metadata)."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import wave
from pathlib import Path
from typing import Any

from ..jobs.runner import ArtifactSpec, JobContext, write_json_artifact
from .config import RunnerError

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


def _gsv_home() -> Path:
    value = os.environ.get("AMS_TTS_GSV_HOME", "").strip()
    if value:
        return Path(value).expanduser()
    ams = os.environ.get("AMS_HOME", "").strip()
    if not ams:
        raise RunnerError("BACKEND_NOT_CONFIGURED", "AMS_HOME 未配置")
    return Path(ams).expanduser() / "tts-gsv"


def _read_metadata(dataset: Path) -> dict[str, str]:
    for name in ("metadata.csv", "metadata.cleaned.csv"):
        p = dataset / name
        if not p.is_file():
            continue
        out: dict[str, str] = {}
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            key = Path(parts[0].strip()).stem
            text = "|".join(parts[1:]).strip()
            if key and text:
                out[key] = text
        return out
    p = dataset / "xianyun.list"
    if p.is_file():
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                out[Path(parts[0]).stem] = parts[3].strip()
        return out
    raise RunnerError("INVALID_PARAMS", f"数据集缺少 metadata.csv / metadata.cleaned.csv / xianyun.list：{dataset}")


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:  # noqa: BLE001
        return None


class TtsDatasetGsvRunner:
    job_type = "tts.dataset.prepare.gsv"
    gpu = False
    description = "GPT-SoVITS 数据集导入/校验（wav + metadata -> data/tts-gsv/datasets）"

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        params = dict(ctx.params or {})
        source = Path(str(params.get("source_path") or "")).expanduser().resolve()
        if not source.is_dir():
            raise RunnerError("INPUT_NOT_FOUND", f"源数据集目录不存在：{source}")
        name = str(params.get("dataset_name") or source.name).strip()
        if not SAFE_NAME.match(name):
            raise RunnerError("INVALID_PARAMS", "dataset_name 只能包含中英文/数字/下划线/连字符，1-64 字符")
        wav_dir = source / "wav" if (source / "wav").is_dir() else source
        wavs = sorted(wav_dir.glob("*.wav"))
        if not wavs:
            raise RunnerError("INVALID_PARAMS", f"源目录没有 wav：{wav_dir}")
        meta = _read_metadata(source)
        clips = [w for w in wavs if w.stem in meta]
        if not clips:
            raise RunnerError("INVALID_PARAMS", "metadata 中没有任何 wav 文件名匹配")
        target = _gsv_home() / "datasets" / name
        if target.exists():
            raise RunnerError("INPUT_EXISTS", f"目标数据集已存在（拒绝覆盖）：{target}")
        (target / "wav").mkdir(parents=True)
        ctx.log("info", f"[tts.dataset.prepare.gsv] import {len(clips)} clips -> {target}")
        await ctx.progress(10, "import", name)
        total = 0.0
        rates = set()
        for index, wav in enumerate(clips, start=1):
            shutil.copy2(wav, target / "wav" / wav.name)
            dur = _wav_duration(wav)
            if dur:
                total += dur
            try:
                with wave.open(str(wav), "rb") as wf:
                    rates.add(wf.getframerate())
            except Exception:  # noqa: BLE001
                pass
            if index % 50 == 0:
                await ctx.progress(10 + 70 * index / len(clips), "import", f"{index}/{len(clips)}")
        with (target / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            for wav in clips:
                writer.writerow([wav.name, meta[wav.stem]])
        report = {
            "ok": True,
            "dataset": str(target),
            "source": str(source),
            "clips": len(clips),
            "total_seconds": round(total, 3),
            "sample_rates": sorted(rates),
            "metadata": "metadata.csv",
        }
        report_path = target / "dataset_report.json"
        write_json_artifact(report_path, report)
        await ctx.progress(100, "done", name)
        ctx.log("info", f"[tts.dataset.prepare.gsv] done clips={len(clips)} dur={total:.1f}s")
        return [ArtifactSpec(name="dataset_report.json", path=report_path, kind="report", meta=report)]
