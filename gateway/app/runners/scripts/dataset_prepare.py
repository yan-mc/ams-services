#!/usr/bin/env python3
"""AMS D1: dataset import / mono resample / clip / validate."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path

def emit(payload): print(json.dumps(payload, ensure_ascii=False), flush=True)
def progress(p, stage, message=None):
    item={"type":"progress","percent":round(float(p),1),"stage":stage}
    if message: item["message"]=str(message)[:300]
    emit(item)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-path",required=True)
    parser.add_argument("--dataset-name",required=True)
    parser.add_argument("--home",required=True)
    parser.add_argument("--target-sr",type=int,default=48000)
    parser.add_argument("--min-sec",type=float,default=3.0)
    parser.add_argument("--max-sec",type=float,default=10.0)
    args=parser.parse_args()
    src=Path(args.source_path).expanduser().resolve()
    home=Path(args.home).expanduser().resolve()
    name="".join(c for c in args.dataset_name if c not in '<>:"/\\|?*').strip()
    if not name: raise RuntimeError("dataset name empty")
    base=home/"datasets"/name
    if base.exists(): raise RuntimeError("dataset already exists: "+str(base))
    sources=base/"sources"; processed=base/"processed"
    sources.mkdir(parents=True); processed.mkdir(parents=True)
    files=[]
    if src.is_file() and src.suffix.lower()==".wav": files=[src]
    elif src.is_dir(): files=sorted([p for p in src.rglob("*.wav") if p.is_file()])
    if not files: raise RuntimeError("no wav files in source: "+str(src))
    for f in files:
        dst=sources/f.name
        if not dst.exists(): shutil.copy2(f,dst)
    progress(10,"copy",f"{len(files)} source wav")
    import numpy as np
    import soundfile as sf
    import librosa
    clips=[]; total=0.0; issues=[]
    target=int(args.target_sr)
    for index, f in enumerate(files):
        try:
            wav, sr = sf.read(str(f), always_2d=False)
            if getattr(wav,"ndim",1)>1: wav=wav.mean(axis=1)
            if sr != target:
                wav=librosa.resample(np.asarray(wav,dtype=np.float32),orig_sr=sr,target_sr=target)
            wav=np.asarray(wav,dtype=np.float32)
            step=int(target*args.max_sec)
            min_len=int(target*args.min_sec)
            for offset in range(0,len(wav),step):
                clip=wav[offset:offset+step]
                if len(clip)<min_len: continue
                out=processed/(f.stem+"_%03d.wav"%len(clips))
                sf.write(str(out),clip,target,subtype="PCM_16")
                clips.append({"file":str(out),"seconds":round(len(clip)/target,3)})
                total += len(clip)/target
        except Exception as exc:
            issues.append({"file":str(f),"error":str(exc)[:300]})
        progress(10+80*(index+1)/len(files),"slice",f"{index+1}/{len(files)}")
    meta={"schema":"ams.dataset/1","name":name,"created_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"source":str(src),"target_sr":target,"min_sec":args.min_sec,"max_sec":args.max_sec,"source_files":len(files),"clip_count":len(clips),"total_seconds":round(total,2),"clips":clips,"issues":issues}
    (base/"dataset.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    progress(98,"done","dataset ready")
    emit({"type":"artifact","name":"dataset.json","path":str(base/"dataset.json"),"kind":"report"})
    emit({"type":"result","ok":True,"meta":{"dataset":str(base),"clips":len(clips),"seconds":round(total,2)}})

if __name__=="__main__":
    try: main()
    except Exception as exc:
        emit({"type":"result","ok":False,"error":{"code":"DATASET_FAILED","message":str(exc)[:1000]}})
        raise
