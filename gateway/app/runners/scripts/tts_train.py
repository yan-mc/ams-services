"""AMS TTS voice training pipeline (Windows native Piper/VITS)."""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, time
from pathlib import Path

def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)

def progress(p, stage, message=None):
    item={"type":"progress","percent":round(float(p),1),"stage":stage}
    if message: item["message"]=str(message)[:300]
    emit(item)

def log(level, message):
    emit({"type":"log","level":level,"message":str(message)[:1000]})

def preprocess_single(dataset, out, language, sample_rate, quality):
    """Single-process Piper preprocess (Windows spawn workers are unreliable)."""
    import piper_phonemize as pp
    from piper_train.norm_audio import cache_norm_audio, make_silence_detector
    template = Path(os.environ["AMS_TTS_CONFIG"])
    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["audio"] = {"sample_rate": int(sample_rate), "quality": quality}
    cfg["espeak"] = {"voice": language}
    cfg["language"] = {"code": language}
    cfg["phoneme_type"] = "espeak"
    cfg["phoneme_map"] = {}
    cfg["phoneme_id_map"] = pp.get_espeak_map()
    cfg["num_symbols"] = pp.get_max_phonemes()
    cfg["num_speakers"] = 1
    cfg["speaker_id_map"] = {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    detector = make_silence_detector()
    cache = out / "cache" / str(sample_rate)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = (dataset / "metadata.csv").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        name, text = line.split("|", 1)
        wav = dataset / "wav" / (name + ".wav")
        if not wav.is_file():
            continue
        phones = pp.phonemize_espeak(text, language)
        flat = [x for sentence in phones for x in sentence]
        if not flat:
            continue
        ids = pp.phoneme_ids_espeak(flat)
        norm, spec = cache_norm_audio(wav, cache, detector, int(sample_rate))
        rows.append({"text": text, "audio_path": str(wav), "speaker": None,
                     "speaker_id": None, "phonemes": flat, "phoneme_ids": ids,
                     "audio_norm_path": str(norm), "audio_spec_path": str(spec)})
        if len(rows) % 50 == 0:
            progress(8 + 20 * min(1.0, len(rows) / max(1, len(lines))), "preprocess", f"{len(rows)} rows")
    if not rows:
        raise RuntimeError("preprocess produced no rows")
    with (out / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ams-home", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--base-checkpoint", default="")
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--language", default="cmn")
    ap.add_argument("--sample-rate", type=int, default=22050)
    args=ap.parse_args()

    root=Path(args.root).resolve()
    ams=Path(args.ams_home).resolve()
    dataset=Path(args.dataset_dir).resolve()
    work=ams/"tts_train"/("job_"+time.strftime("%Y%m%d_%H%M%S"))
    work.mkdir(parents=True, exist_ok=True)
    pre=work/"preprocessed"
    out_voice=ams/"tts_voices"/args.model_name
    if out_voice.exists():
        raise RuntimeError("voice already exists: "+str(out_voice))

    sys.path.insert(0,str(root/"shims"))
    sys.path.insert(0,str(root/"piper"/"src"/"python"))
    sys.path.insert(0,str(root/"deps"))
    import pathlib
    pathlib.PosixPath=pathlib.WindowsPath

    os.environ["PYTHONIOENCODING"]="utf-8"
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    os.environ["AMS_TTS_CONFIG"]=str(ams/"tts_voices"/"zh_CN-huayan-medium"/"zh_CN-huayan-medium.onnx.json")
    os.environ["AMS_ESPEAK_ROOT"]=str(root/"espeak-ng")

    progress(3,"dataset","validate dataset")
    if not (dataset/"metadata.csv").is_file():
        raise RuntimeError("metadata.csv missing: "+str(dataset))
    if not (dataset/"wav").is_dir():
        raise RuntimeError("wav/ missing: "+str(dataset))

    progress(8,"preprocess","single-process Piper preprocess")
    count = preprocess_single(dataset, pre, args.language, args.sample_rate, args.quality)
    if count <= 0:
        raise RuntimeError("preprocess produced no rows")
    progress(30,"preprocess",str(count)+" rows ready")

    base=Path(args.base_checkpoint).expanduser() if args.base_checkpoint else (root/"checkpoints"/"epoch=3269-step=2460540.ckpt")
    if not base.is_file():
        raise RuntimeError("base checkpoint missing: "+str(base))
    m=re.search(r"epoch=(\d+)",base.name)
    base_epoch=int(m.group(1)) if m else 0
    max_epochs=base_epoch+max(1,args.epochs)+1
    progress(32,"train","fine-tune from "+base.name)
    from piper_train.__main__ import main as train_main
    sys.argv=["piper_train","--dataset-dir",str(pre),"--max_epochs",str(max_epochs),
              "--accelerator","gpu","--devices","1","--batch-size",str(args.batch_size),
              "--checkpoint-epochs","1","--default_root_dir",str(work),
              "--resume_from_checkpoint",str(base),"--num_sanity_val_steps","0"]
    train_main()
    progress(78,"train","training returned")

    ckpts=sorted((work/"lightning_logs").glob("version_*/checkpoints/*.ckpt"), key=lambda p:p.stat().st_mtime)
    if not ckpts:
        raise RuntimeError("checkpoint not found")
    ckpt=ckpts[-1]
    onnx=work/(args.model_name+".onnx")
    progress(82,"export","export ONNX")
    from piper_train.export_onnx import main as export_main
    sys.argv=["export",str(ckpt),str(onnx)]
    export_main()
    if not onnx.is_file():
        raise RuntimeError("ONNX not generated")
    config=pre/"config.json"
    if not config.is_file():
        raise RuntimeError("preprocess config missing")
    cfg=work/(args.model_name+".onnx.json")
    shutil.copy2(config,cfg)

    progress(92,"register","register voice")
    out_voice.mkdir(parents=True, exist_ok=False)
    shutil.copy2(onnx,out_voice/(args.model_name+".onnx"))
    shutil.copy2(cfg,out_voice/(args.model_name+".onnx.json"))
    (out_voice/"voice.json").write_text(json.dumps({
        "name":args.model_name,"engine":"piper","sample_rate":args.sample_rate,
        "language":args.language,"quality":args.quality,"dataset":str(dataset),
        "epochs":args.epochs,"base_checkpoint":str(base),
        "created_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
    },ensure_ascii=False,indent=2),encoding="utf-8")
    progress(100,"done")
    emit({"type":"artifact","name":args.model_name+".onnx","path":str(out_voice/(args.model_name+".onnx")),"kind":"model","meta":{"sample_rate":args.sample_rate}})
    emit({"type":"artifact","name":args.model_name+".onnx.json","path":str(out_voice/(args.model_name+".onnx.json")),"kind":"config"})
    emit({"type":"result","ok":True,"meta":{"voice":str(out_voice),"work":str(work)}})

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        import traceback; traceback.print_exc()
        emit({"type":"result","ok":False,"error":{"code":"TTS_TRAIN_FAILED","message":str(exc)[:1000]}})
        raise
