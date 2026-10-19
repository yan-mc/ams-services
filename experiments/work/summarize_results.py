import json, glob, os
for p in sorted(glob.glob(os.path.join(r"F:\\Deepseek_harness\\AI音乐处理\\08_项目代码\\ams-services\\experiments\\results", "*.json"))):
    name = os.path.basename(p)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        print(name, "PARSE_ERR", exc); continue
    if "runs" in d:
        r = d["runs"]
        print("%-28s model=%-28s batch=%s runs=%s" % (name, d.get("model"), d.get("batch_size"), len(r)))
        for x in r:
            print("   run%s wall=%ss rtf=%s vram_alloc=%sMB vram_reserved=%sMB ws=%sMB cpu_ratio=%s" % (
                x["run"], x["wall_sec"], x["rtf_reported"], x["vram_peak_alloc_mb"], x["vram_peak_reserved_mb"], x["peak_working_set_mb"], x["cpu_ratio"]))
    elif "summary" in d:
        print("%-28s %s" % (name, json.dumps(d["summary"], ensure_ascii=False)))
    else:
        print(name, "keys=", list(d)[:6])
