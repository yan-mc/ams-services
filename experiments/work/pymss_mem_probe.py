import json, sys, time, psutil
from pymss.cli import main as pymss_main
base = psutil.Process()
args = ["infer","scnet_checkpoint_musdb18","-i",sys.argv[1],"-o",sys.argv[2],"--save-as-folder","--device","cpu","--format","wav","--wav-bit-depth","FLOAT","--model-dir",sys.argv[3],"--source","hf-mirror","--param","batch_size=1"]
t0=time.perf_counter(); c0=time.process_time(); code=0
try: code = pymss_main(args)
except SystemExit as exc: code = int(exc.code or 0)
wall=time.perf_counter()-t0; cpu=time.process_time()-c0
info = base.memory_info()
print(json.dumps({"exit":code,"wall_sec":round(wall,2),"cpu_sec":round(cpu,2),
  "peak_working_set_mb":round(info.peak_wset/1024**2,1) if hasattr(info,"peak_wset") else None,
  "rss_mb":round(info.rss/1024**2,1),"cpu_ratio":round(cpu/wall,2)}, ensure_ascii=False))
