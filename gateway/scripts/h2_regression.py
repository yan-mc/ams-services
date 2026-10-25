#!/usr/bin/env python3
import json, os, sys, time, urllib.request
from pathlib import Path

HOME = Path(r'F:\AMS-Studio\data')
ports = json.loads((HOME/'runtime/ports.json').read_text(encoding='utf-8'))
BASE = 'http://127.0.0.1:%s/api/v1' % ports['gateway']['port']
TOKEN = (HOME/'runtime/gateway.token').read_text(encoding='utf-8').strip()

def req(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    r = urllib.request.Request(BASE+path, data=data, method=method, headers={'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode('utf-8'))

def submit(kind, params):
    r = req('POST','/jobs',{'type':kind,'params':params})
    return r.get('data') or r

def wait(job_id, timeout=1800):
    start=time.time()
    while time.time()-start<timeout:
        d=req('GET','/jobs/'+job_id)
        j=d.get('data') or d
        st=j.get('status')
        print('%s %s %.1f %s' % (job_id, st, j.get('progress') or 0, j.get('stage')), flush=True)
        if st in ('succeeded','failed','cancelled'): return j
        time.sleep(3)
    return req('GET','/jobs/'+job_id)['data']

def artifacts(job_id):
    return req('GET','/jobs/'+job_id+'/artifacts').get('data') or []

def wav(job_id):
    for a in artifacts(job_id):
        if a.get('kind')=='audio' and str(a.get('name','')).lower().endswith('.wav'):
            return a['path']
    return None

def run(kind, params, timeout=1800):
    print('SUBMIT', kind, json.dumps(params,ensure_ascii=False), flush=True)
    j=submit(kind,params)
    print('JOB', j.get('id'), flush=True)
    j=wait(j['id'], timeout)
    out={'kind':kind,'job_id':j.get('id'),'status':j.get('status'),'error':j.get('error'),'artifacts':artifacts(j.get('id')),'job':j}
    return out

def main():
    suite=sys.argv[1] if len(sys.argv)>1 else 'core'
    results=[]
    voice=r'F:\AMS-Studio\data\voices\曼波\曼波.pth'
    src3=r'F:\AMS-Studio\data\projects\imports\e_audit_3min.wav'
    zh=r'F:\AMS-Studio\data\tts_voices\zh_CN-huayan-medium\zh_CN-huayan-medium.onnx'
    en=r'F:\AMS-Studio\data\tts_voices\en_US-lessac-medium\en_US-lessac-medium.onnx'
    if suite in ('core','all'):
        results.append(run('ping',{},120))
        t1=run('tts.synthesize',{'text':'AMS Studio 核心功能回归测试，中文语音合成。','model':zh},300); results.append(t1)
        t2=run('tts.synthesize',{'text':'AMS Studio core regression test for English speech synthesis.','model':en},300); results.append(t2)
        a1=wav(t1['job_id']); a2=wav(t2['job_id'])
        if a1: results.append(run('evaluate.audio',{'input_path':a1},300))
        if a1 and a2: results.append(run('mix.master',{'input_paths':a1+';'+a2,'loudness':'-14','true_peak':'-1'},600))
    if suite in ('rvc','all'):
        results.append(run('convert.rvc',{'input_path':src3,'voice_name':voice,'f0_up_key':0,'index_rate':0.75,'device':'cuda:0'},1800))
    if suite in ('rvc512','all'):
        results.append(run('convert.rvc',{'input_path':r'F:\AMS-Studio\data\artifacts\h2_fixtures\e_audit_5min.wav','voice_name':voice,'f0_up_key':0,'index_rate':0.75,'device':'cuda:0'},1800))
        results.append(run('convert.rvc',{'input_path':r'F:\AMS-Studio\data\projects\imports\e_audit_12min_long.wav','voice_name':voice,'f0_up_key':0,'index_rate':0.75,'device':'cuda:0'},3600))
    if suite in ('pymss','all'):
        results.append(run('separate.pymss',{'input_path':src3,'model':'scnet_checkpoint_musdb18','device':'auto'},1800))
    if suite in ('pymss10','all'):
        results.append(run('separate.pymss',{'input_path':r'F:\AMS-Studio\data\artifacts\h2_fixtures\e_audit_10min.wav','model':'scnet_checkpoint_musdb18','device':'auto'},1800))
    if suite in ('train','all'):
        results.append(run('train.rvc',{'dataset_dir':r'F:\AMS-Studio\data\datasets\ljspeech15\processed','model_name':'h2_train_test','version':'v2','sample_rate':'48k','if_f0':1,'f0_method':'rmvpe','epochs':10,'batch_size':4,'save_every':5,'gpu':'0'},3600))
    out=HOME/'logs'/('h2_'+suite+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('WROTE',out, flush=True)
    for r in results: print(r['kind'], r['status'], r['job_id'], (r.get('error') or {}).get('message',''), flush=True)
if __name__=='__main__': main()
