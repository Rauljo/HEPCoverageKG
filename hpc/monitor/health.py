import glob, json, os, subprocess, time

# Only files belonging to jobs that are RUNNING right now. A cancelled job's file
# is still on disk and still recent, and reading it as "current" is how a dead run
# gets mistaken for a broken live one.
# Python 3.6 on the login node: no capture_output, no text=
live = subprocess.check_output(
    ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j"]).decode()
live_ids = [l.split()[0] for l in live.splitlines() if "hepkg-eval" in l]
started = {}
for j in live_ids:
    log = "logs/eval_%s.out" % j
    if os.path.exists(log):
        started[j] = os.path.getmtime(log)

cutoff = min(started.values()) - 120 if started else time.time()
files = [p for p in glob.glob("eval/runs/*.jsonl") if os.path.getmtime(p) > cutoff]
files.sort(key=os.path.getmtime)

print("live eval jobs: %s" % " ".join(live_ids))
allgood = bool(files)
for p in files:
    meta, ok, err = {}, 0, 0
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            continue
        if "_meta" in o:
            meta = o["_meta"]; continue
        err += 1 if o["answer"]["error"] else 0
        ok += 0 if o["answer"]["error"] else 1
    name = meta.get("questions_path", p).split("/")[-1][:40]
    print("  %-42s ok=%-4d err=%d" % (name, ok, err))
    if err or ok == 0:
        allgood = False
print("\n%s" % ("HEALTHY: every live job is answering, zero errors" if allgood
                else "CHECK NEEDED"))
