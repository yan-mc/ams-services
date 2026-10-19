# -*- coding: utf-8 -*-
r"""RVC 换声冒烟脚本（M0.7）。

用途：在 RVC20260311 整合包上，用指定音源（默认曼波 40k）对一段干声做 RVC v2 换声，
记录阶段耗时、GPU 显存峰值、输入/输出时长与峰值，输出 wav + JSON 结果。

只读约束：
- 本脚本只读取 RVC 源目录（模型/hubert/rmvpe/index/ffmpeg）；
- 必须用 RVC 自带 runtime python 且加 -B；建议同时设置 PYTHONDONTWRITEBYTECODE=1；
- TMP/TEMP 由调用方指向实验目录（experiments/tmp），避免写系统临时目录；
- 输出 wav 与会话日志写入实验/验证目录（由参数指定）。

用法：
  F:\ai歌曲\RVC20260311\runtime\python.exe -B rvc_smoke.py ^
    --rvc-root "F:\ai歌曲\RVC20260311" ^
    --input "experiments\work\input_vocal_35s.wav" ^
    --output "07_执行计划\验证\audio\rvc_manbo40k_35s.wav" ^
    --model manbo.pth --index assets/indices/manbo.index ^
    --f0-method rmvpe --index-rate 0.75 --f0up-key 0 --protect 0.33 ^
    --rms-mix-rate 0.25 --filter-radius 3 --is-half --device cuda:0 ^
    --result-json "experiments\results\rvc_smoke.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(description="RVC 换声冒烟")
    parser.add_argument("--rvc-root", required=True, help="RVC20260311 整合包目录")
    parser.add_argument("--input", required=True, help="输入干声（wav/mp3/flac 等）")
    parser.add_argument("--output", required=True, help="输出 wav 路径")
    parser.add_argument("--model", default="manbo.pth", help="assets/weights 下的模型文件名")
    parser.add_argument("--speaker-id", type=int, default=0, help="说话人 id（单说话人模型=0，与 RVC WebUI 一致）")
    parser.add_argument("--index", default="", help="index 路径（相对 RVC 根目录或绝对路径；留空=不用 index）")
    parser.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"])
    parser.add_argument("--f0up-key", type=int, default=0, help="变调半音数")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--is-half", action="store_true", help="CUDA 上使用 fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-json", default="", help="结果 JSON 输出路径")
    parser.add_argument(
        "--no-index-compat",
        action="store_false",
        dest="index_compat",
        help="关闭 faiss.read_index 的非 ASCII 路径兼容补丁",
    )
    return parser.parse_args(argv)


def _peak_dbfs(audio) -> float:
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return -999.0
    return round(20.0 * float(np.log10(peak)), 2)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = {
        "status": "unknown",
        "params": {key: value for key, value in vars(args).items()},
        "env": {},
        "timings": {},
    }
    started_all = time.perf_counter()

    rvc_root = Path(args.rvc_root).resolve()
    result["env"]["rvc_root"] = str(rvc_root)
    result["env"]["python"] = sys.version.split()[0]

    # 先解析本脚本参数，再进入 RVC 目录并清空 sys.argv（避免被 RVC 的 Config.arg_parse 解析）
    os.chdir(str(rvc_root))
    sys.path.insert(0, str(rvc_root))
    sys.argv = sys.argv[:1]

    from dotenv import load_dotenv

    # 显式指定 .env 路径：python-dotenv 的 find_dotenv 在脚本模式下会从脚本所在目录
    # 向上查找，而不是当前工作目录，因此这里必须给绝对路径。
    load_dotenv(dotenv_path=str(rvc_root / ".env"), override=False)
    # .env 若缺失/解析失败，用 RVC 默认相对路径兜底（相对 cwd=RVC 根）
    for _key, _value in {
        "weight_root": "assets/weights",
        "weight_uvr5_root": "assets/uvr5_weights",
        "index_root": "logs",
        "outside_index_root": "assets/indices",
        "rmvpe_root": "assets/rmvpe",
    }.items():
        if not os.environ.get(_key):
            os.environ[_key] = _value
    result["env"]["rvc_env"] = {
        key: os.environ.get(key)
        for key in ("weight_root", "index_root", "rmvpe_root", "outside_index_root")
    }

    import numpy as np
    import soundfile as sf
    import torch

    try:
        from configs.config import Config
        from infer.modules.vc.modules import VC

        config = Config()
        if args.device and torch.cuda.is_available() and args.device.startswith("cuda"):
            config.device = args.device
        config.is_half = bool(args.is_half and str(config.device).startswith("cuda"))

        result["env"].update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(config.device),
                "is_half": bool(config.is_half),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)
                if torch.cuda.is_available()
                else 0,
            }
        )

        index_arg = args.index.strip()
        if index_arg and not os.path.isabs(index_arg):
            index_path = str(rvc_root / index_arg)
        else:
            index_path = index_arg
        result["params"]["index_resolved"] = index_path

        # Windows 非 ASCII 路径下 faiss.read_index 会 fopen 失败（RVC pipeline 会静默忽略 index，
        # 相当于 index_rate 失效）。这里做兼容：失败时用 Python 读字节 + faiss.deserialize_index。
        import faiss as _faiss
        import numpy as _np

        _index_compat = {"fallback_used": 0, "errors": []}
        _orig_read_index = _faiss.read_index

        def _read_index_unicode_safe(index_path, *args, **kwargs):
            try:
                return _orig_read_index(index_path, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                try:
                    with open(index_path, "rb") as handle:
                        payload = handle.read()
                    _index_compat["fallback_used"] += 1
                    return _faiss.deserialize_index(_np.frombuffer(payload, dtype=_np.uint8))
                except Exception as inner:  # noqa: BLE001
                    _index_compat["errors"].append(f"{type(inner).__name__}: {inner}")
                    raise exc

        if args.index_compat:
            _faiss.read_index = _read_index_unicode_safe
            result["index_compat"] = "deserialize-fallback"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        vc = VC(config)

        time_load = time.perf_counter()
        vc.get_vc(args.model)
        load_sec = time.perf_counter() - time_load

        input_path = str(Path(args.input).resolve())
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_infer = time.perf_counter()
        rvc_info, wav_opt = vc.vc_single(
            int(args.speaker_id),
            input_path,
            int(args.f0up_key),
            None,
            args.f0_method,
            index_path,
            None,
            float(args.index_rate),
            int(args.filter_radius),
            int(args.resample_sr),
            float(args.rms_mix_rate),
            float(args.protect),
        )
        infer_sec = time.perf_counter() - time_infer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result["rvc_info"] = str(rvc_info)
        result["index_compat_stats"] = _index_compat
        if wav_opt is None or wav_opt[0] is None:
            result["status"] = "failed"
            result["error"] = "vc_single 未返回音频（详见 rvc_info/traceback）"
        else:
            tgt_sr, audio = int(wav_opt[0]), wav_opt[1]
            audio_arr = np.asarray(audio)
            # RVC pipeline 返回 int16（pipeline.py 末尾 *32768 转 int16）；写文件前归一到 float32，
            # 避免把 int16 数值当 float 直接写导致爆音/削波。
            if audio_arr.dtype.kind in {"i", "u"}:
                scale = float(np.iinfo(audio_arr.dtype).max) + 1.0
                audio_float = audio_arr.astype(np.float32) / scale
            else:
                audio_float = audio_arr.astype(np.float32)
            peak = float(np.max(np.abs(audio_float))) if audio_float.size else 0.0
            clipped = int(np.sum(np.abs(audio_float) >= 0.999969)) if audio_float.size else 0

            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_float, tgt_sr, subtype="PCM_24")

            input_info = sf.info(input_path)
            output_info = sf.info(str(output_path))
            input_sec = input_info.frames / float(input_info.samplerate)
            output_sec = output_info.frames / float(output_info.samplerate)

            result.update(
                {
                    "status": "ok",
                    "output": str(output_path),
                    "output_bytes": output_path.stat().st_size,
                    "output_sample_rate": tgt_sr,
                    "output_channels": int(output_info.channels),
                    "output_source_dtype": str(audio_arr.dtype),
                    "input_duration_sec": round(input_sec, 4),
                    "output_duration_sec": round(output_sec, 4),
                    "duration_diff_ms": round(abs(output_sec - input_sec) * 1000.0, 2),
                    "output_peak": round(peak, 6),
                    "output_peak_dbfs": _peak_dbfs(audio_float),
                    "output_clipped_samples": clipped,
                    "output_format": "wav/pcm_24",
                }
            )
            result["timings"] = {
                "load_model_sec": round(load_sec, 3),
                "infer_sec": round(infer_sec, 3),
                "total_sec": round(time.perf_counter() - started_all, 3),
                "rtf_infer": round(infer_sec / input_sec, 4) if input_sec else None,
            }
            if torch.cuda.is_available():
                result["vram_peak_alloc_mb"] = int(torch.cuda.max_memory_allocated() / 1024 ** 2)
                result["vram_peak_reserved_mb"] = int(torch.cuda.max_memory_reserved() / 1024 ** 2)
    except Exception as exc:  # noqa: BLE001 - 冒烟脚本需要把失败完整落盘
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        elapsed = round(time.perf_counter() - started_all, 3)
        result.setdefault("timings", {})["wall_sec"] = elapsed
        if args.result_json:
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        printable = {key: value for key, value in result.items() if key != "traceback"}
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        if result.get("traceback"):
            print(result["traceback"], file=sys.stderr)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
