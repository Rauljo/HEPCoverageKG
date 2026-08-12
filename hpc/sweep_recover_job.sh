#!/bin/bash
# =============================================================================
# HEPCoverageKG: recover the 60-paper sweep from its own stored replies.
#
# 475 of the sweep's 19,251 calls were discarded by the JSON parser, and every
# single one of them said YES:
#
#     recovered YES: 475      recovered NO: 0
#
# That is not a coincidence, it is the mechanism. A "no" reply carries no quote,
# so there is no LaTeX to choke on and it always parsed. Only a "yes" quotes the
# paper. The bug could therefore only ever delete evidence, never invent it --
# one-directional recall loss, and the harness reported it as "not found".
#
# Re-parsing moves 48 of 480 paper-reads. 26 become yes, 22 become split, which
# is a disagreement we were previously blind to rather than a new answer.
#
# The recovered yeses have never been past the support judge, so they are raw
# recall, not a score. That is what this job is for.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=sweep-recover
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/sweep_recover_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem 24G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
module load Python/3.9.6-GCCcore-11.2.0
set -a; [ -f .env ] && . ./.env; set +a

BASE="${LLM_BASE_URL%/v1}"; HOST="${BASE%:*}"
B_URL="${HOST}:8001/v1"; B_MODEL="Qwen/QwQ-32B-AWQ"

SRC="eval/reader/20260809T202844-sweep-48330.jsonl"
[ -f "$SRC" ] || { echo "ERROR: $SRC missing"; exit 1; }
OUT="${SRC%.jsonl}.reparsed.jsonl"

echo "=== REPARSE (no model calls) ==="
.venv/bin/python - "$SRC" "$OUT" <<'PY'
import sqlite3, sys
from hepcoveragekg.kg import store
from hepcoveragekg.eval import reader as R
conn = sqlite3.connect(f"file:{store.DEFAULT_DB_PATH}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
info = R.reparse(sys.argv[1], sys.argv[2], conn=conn, mode=R.EXISTENCE)
print(f"  {info['recovered']} calls recovered of {info['calls']}, "
      f"{info['still_unparsed']} still unparsed, {info['hard_errors']} hard errors")
PY

echo "--- waiting for the judge endpoint ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-7200} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${B_URL}/models" >/tmp/j_$$.json 2>/dev/null && grep -q '"id"' /tmp/j_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: no judge endpoint"; exit 1; }
  sleep 30
done
SERVED=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/j_$$.json)
rm -f /tmp/j_$$.json
echo "serving: ${SERVED}"
[ "${SERVED#*"$B_MODEL"}" = "$SERVED" ] && { echo "ERROR: expected $B_MODEL"; exit 1; }

echo "=== JUDGE ==="
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" \
  .venv/bin/python -m hepcoveragekg.cli reader --check-support "$OUT" \
  --judge-url "$B_URL" --judge-model "$B_MODEL" --judge-max-tokens 2500 \
  --concurrency "${LLM_CONCURRENCY:-10}"

echo "=== SCORES: before the parser fix vs after, both judged ==="
.venv/bin/python - "eval/reader/20260809T202844-sweep-48330.rechecked.supported.jsonl" \
                   "${OUT%.jsonl}.supported.jsonl" <<'PY'
import io, json, sys, os
from hepcoveragekg.eval import supervisor as S
gold = {q["qid"]: set(q["provenance"]["gabriel_gold"]["papers"])
        for q in S.build_records()
        if (q["provenance"].get("gabriel_gold") or {}).get("papers")}

def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["qid"], {})[r["paper_id"]] = r["answer"]
    return out

before, after = load(sys.argv[1]), load(sys.argv[2])
print(f"{'qid':7} {'gold':>4} | {'before: found':>13} {'hit':>4} {'FP':>4}"
      f" | {'after: found':>12} {'hit':>4} {'FP':>4}")
th = ta = 0
for qid in sorted(gold):
    if qid not in after:
        continue
    g = gold[qid]
    yb = {p for p, v in before.get(qid, {}).items() if v is True}
    ya = {p for p, v in after[qid].items() if v is True}
    th += len(yb & g); ta += len(ya & g)
    print(f"{qid:7} {len(g):4} | {len(yb):13} {len(yb & g):4} {len(yb - g):4}"
          f" | {len(ya):12} {len(ya & g):4} {len(ya - g):4}")
print(f"\n  gold papers found: {th} -> {ta}")
PY
echo "--- done ---"
