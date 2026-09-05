#!/bin/bash
# 8B arms -> swap to TP=2 -> verify the count-step fix cheaply -> only then the
# 20-hour combined arm. The verification gate exists because the SAME mechanism
# has been miswired twice, and both times only the data showed it.
set -u
cd /home/xucabrjs/HEPCoverageKG

say() { echo "[$(date +%H:%M)] $*"; }

say "waiting for the 8B arms"
while [ "$(squeue -h -u "$USER" -n arm8b | wc -l)" -gt 0 ]; do sleep 180; done
say "8B arms done"

say "swapping to TP=2 for full answerer throughput"
NEW=$(sbatch --parsable --gres=gpu:a100:2 --export=ALL,VLLM_TP=2,VLLM_MAX_LEN=32768 serve_vllm_70b.sh)
scancel 48569
L=/home/xucabrjs/hepcoveragekg_setup/logs/vllm72b_${NEW}.out
for _ in $(seq 1 60); do
  grep -q "Application startup complete" "$L" 2>/dev/null && break
  grep -q "REFUSING TO START" "$L" 2>/dev/null && { say "server refused a bad GPU -- stopping"; exit 1; }
  sleep 20
done
grep -q "Application startup complete" "$L" || { say "server never came up -- stopping"; exit 1; }
say "server $NEW up"

say "verifying the count-step citation on the cheap off arm"
V=""
for q in shards/conceptB-00.jsonl dev-2026-08-03-paperA-200.jsonl; do
  id=$(sbatch --parsable --export=ALL,VLLM_PORT=8000,ARM=off,CONTRACT=v2 \
        hpc/critic_arm_job.sh "eval/questions/$q")
  V="$V $id"
done
while [ "$(squeue -h -j $(echo $V | tr ' ' ',') -o %i 2>/dev/null | wc -l)" -gt 0 ]; do sleep 180; done
say "verification arm finished:$V"

# Did the mechanism actually fire? A cited value must be present and must have
# come from a `count` step. Near-zero means it is miswired again, and the
# 20-hour arm must not run on it.
CITED=$(module load Python/3.9.6-GCCcore-11.2.0 >/dev/null 2>&1; .venv/bin/python - <<'PY'
import glob, json
tot = good = 0
for j in __import__("os").environ.get("VJOBS", "").split():
    for f in glob.glob("eval/runs/*-hepkg-%s.jsonl" % j):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if "_meta" in r: continue
            tot += 1
            a = r.get("answer") or {}
            if a.get("value") is not None and "value=" in (a.get("cited") or ""):
                good += 1
print(int(100 * good / tot) if tot else 0)
PY
)
export VJOBS="$V"
say "answers carrying a cited count: ${CITED}%"
if [ "${CITED:-0}" -lt 10 ]; then
  say "the citation mechanism is not firing -- NOT starting the 20-hour arm"
  exit 1
fi

say "starting the combined arm: critic + the fixed contract"
for q in shards/conceptB-00.jsonl shards/conceptB-01.jsonl shards/conceptB-02.jsonl dev-2026-08-03-paperA-200.jsonl; do
  sbatch --export=ALL,VLLM_PORT=8000,ARM=ranked,CONTRACT=v2 \
    hpc/critic_arm_job.sh "eval/questions/$q" >/dev/null
done
say "queued 4 jobs: critic=ranked answer=v2(fixed)"
