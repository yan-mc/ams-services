"""tts.dataset.prepare: validate/import a TTS dataset into data/datasets/tts."""
from __future__ import annotations
import argparse, json, shutil, wave
from pathlib import Path

def emit(p):
    print(json.dumps(p,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ams-home",required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--name",required=True)
    a=ap.parse_args()
    src=Path(a.source).resolve(); dst=Path(a.ams_home).resolve()/"datasets"/"tts"/a.name
    if not src.is_dir(): raise RuntimeError("source not dir")
    meta=src/"metadata.csv"
    wavdir=src/"wav" if (src/"wav").is_dir() else src/"wavs"
    if not meta.is_file() or not wavdir.is_dir(): raise RuntimeError("需要 metadata.csv 和 wav/")
    if dst.exists(): raise RuntimeError("dataset already exists: "+str(dst))
    dst.mkdir(parents=True,exist_ok=False)
    (dst/"wav").mkdir()
    rows=[]; total=0.0; missing=0
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        parts=line.split("|")
        if len(parts)<2: continue
        name=parts[0]; text=parts[-1]
        w=wavdir/name
        if not w.exists(): w=wavdir/(name+".wav")
        if not w.exists(): missing+=1; continue
        shutil.copy2(w,dst/"wav"/(name+".wav"))
        try:
            with wave.open(str(w),"rb") as f: total += f.getnframes()/f.getframerate()
        except Exception: pass
        rows.append(name+"|"+text.replace("|"," "))
    if not rows: raise RuntimeError("no valid rows")
    (dst/"metadata.csv").write_text("\n".join(rows)+"\n",encoding="utf-8")
    (dst/"dataset.json").write_text(json.dumps({"name":a.name,"clips":len(rows),"duration_sec":round(total,2),"missing":missing},ensure_ascii=False,indent=2),encoding="utf-8")
    emit({"type":"artifact","name":"dataset.json","path":str(dst/"dataset.json"),"kind":"report","meta":{"clips":len(rows),"duration_sec":round(total,2),"missing":missing}})
    emit({"type":"result","ok":True,"meta":{"dataset":str(dst),"clips":len(rows)}})

if __name__=="__main__":
    try: main()
    except BaseException as e:
        import traceback; traceback.print_exc()
        emit({"type":"result","ok":False,"error":{"code":"TTS_DATASET_FAILED","message":str(e)[:1000]}})
        raise
