#!/usr/bin/env python3
import argparse, json, subprocess, sys
from pathlib import Path

def emit(x): print(json.dumps(x, ensure_ascii=False), flush=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputs',required=True); p.add_argument('--output',required=True); p.add_argument('--ffmpeg',required=True); p.add_argument('--loudness',default='-14'); p.add_argument('--true-peak',default='-1'); p.add_argument('--gains',default=''); p.add_argument('--pans',default=''); p.add_argument('--eq-low',default=''); p.add_argument('--eq-mid',default=''); p.add_argument('--eq-high',default='')
    a=p.parse_args(); paths=[x.strip() for x in a.inputs.replace('\n',';').split(';') if x.strip()]
    if not paths: raise RuntimeError('no input tracks')
    for x in paths:
        if not Path(x).is_file(): raise RuntimeError('missing input: '+x)
    cmd=[a.ffmpeg,'-y']
    for x in paths: cmd += ['-i',x]
    def nums(text):
        return [x.strip() for x in str(text or '').split(',')]
    gains=nums(a.gains); pans=nums(a.pans); lows=nums(a.eq_low); mids=nums(a.eq_mid); highs=nums(a.eq_high)
    chains=[]; labels=[]
    for i in range(len(paths)):
        gain=gains[i] if i<len(gains) and gains[i] else '0'
        pan=float(pans[i]) if i<len(pans) and pans[i] else 0.0
        filters=['volume='+gain+'dB']
        for band,value in ((100,lows),(1000,mids),(8000,highs)):
            if i<len(value) and value[i]: filters.append('equalizer=f=%d:t=q:w=1:g=%s'%(band,value[i]))
        left=max(0.0,min(1.0,1.0-pan)); right=max(0.0,min(1.0,1.0+pan)); filters.append('aformat=channel_layouts=stereo'); filters.append('stereotools=balance_in=%s'%pan)
        labels.append('[a%d]'%i); chains.append('[%d:a]%s[a%d]'%(i,','.join(filters),i))
    filt=';'.join(chains)+';'+''.join(labels)+'amix=inputs=%d,loudnorm=I=%s:TP=%s:LRA=11,alimiter=limit=0.98[out]'%(len(paths),a.loudness,a.true_peak)
    cmd += ['-filter_complex',filt,'-map','[out]','-c:a','pcm_s24le',a.output]
    up=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if up.returncode!=0: raise RuntimeError('ffmpeg failed: '+up.stderr[-800:])
    out=Path(a.output)
    emit({'type':'artifact','name':out.name,'path':str(out),'kind':'audio','meta':{'tracks':len(paths),'loudness_target':a.loudness,'true_peak':a.true_peak}})
    emit({'type':'result','ok':True,'meta':{'tracks':len(paths),'output':str(out)}})
if __name__=='__main__':
    try: main()
    except Exception as e:
        emit({'type':'result','ok':False,'error':{'code':'MIX_FAILED','message':str(e)[:800]}}); raise
