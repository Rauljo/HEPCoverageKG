import glob, json, os, time, collections
for p in sorted([f for f in glob.glob("eval/runs/*.jsonl")
                 if time.time()-os.path.getmtime(f) < 3600], key=os.path.getmtime, reverse=True)[:4]:
    meta, errs, ok = {}, collections.Counter(), 0
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            continue
        if "_meta" in o:
            meta = o["_meta"]; continue
        e = o["answer"]["error"]
        errs[e[:110]] += 1 if e else 0
        ok += 0 if e else 1
    name = meta.get("questions_path", p).split("/")[-1][:38]
    print("%-40s ok=%d" % (name, ok))
    for msg, n in errs.most_common(2):
        if n:
            print("    x%-4d %s" % (n, msg))
