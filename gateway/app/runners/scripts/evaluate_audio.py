#!/usr/bin/env python3
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf

def emit(x): print(json.dumps(x, ensure_ascii=False), flush=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--reference',default=''); p.add_argument('--output-dir',required=True)
    a=p.parse_args(); src=Path(a.input).expanduser(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not src.is_file(): raise RuntimeError('input not found: '+str(src))
    data,sr=sf.read(str(src),always_2d=True); ch=data.shape[1]; duration=len(data)/sr
    peak=float(np.max(np.abs(data))) if data.size else 0.0
    clipping=int(np.sum(np.abs(data)>=0.999)); silent=float(np.mean(np.max(np.abs(data),axis=1)<1e-4)) if data.size else 1.0
    rms=float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    metrics={'duration_sec':round(duration,4),'sample_rate':int(sr),'channels':int(ch),'peak':round(peak,6),'clipping_samples':clipping,'silence_ratio':round(silent,6),'rms_dbfs':round(20*math.log10(max(rms,1e-9)),3)}
    ref=Path(a.reference).expanduser() if a.reference else None
    if ref and ref.is_file():
        rdata,rsr=sf.read(str(ref),always_2d=True)
        if rsr==sr:
            n=min(len(data),len(rdata)); x=data[:n].mean(axis=1); y=rdata[:n].mean(axis=1)
            noise=x-y; sig=float(np.mean(x*x)); noi=float(np.mean(noise*noise)); metrics['snr_db']=round(10*math.log10(max(sig,1e-12)/max(noi,1e-12)),3)
        else: metrics['snr_db']=None; metrics['snr_note']='reference sample rate differs'
    report={'schema':'ams.evaluate/1','input':str(src),'reference':str(ref) if ref else '',**metrics}
    path=out/'evaluation.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    emit({'type':'artifact','name':'evaluation.json','path':str(path),'kind':'report','meta':metrics})
    emit({'type':'result','ok':True,'meta':metrics})
if __name__=='__main__':
    try: main()
    except Exception as e:
        emit({'type':'result','ok':False,'error':{'code':'EVALUATE_FAILED','message':str(e)[:800]}}); raise
