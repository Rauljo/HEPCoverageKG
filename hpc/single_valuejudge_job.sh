#!/bin/bash
# =============================================================================
# HEPCoverageKG: re-judge the single-paper answers on their VALUE.
#
# All seven came back supported. gf-10 shows why that is not a score: gold is
# "approximately 80%", we answered with a 1.5% systematic uncertainty, the
# correct sentence WAS in the evidence, and the judge said yes -- correctly, on
# its own terms. It was asked "do these sentences show the paper does this?",
# and the paper does. It was being asked the wrong question.
#
# check_extraction asks instead whether the evidence supports THE ANSWER WE
# GAVE, and to name a better one if the evidence holds it. The correction is
# kept even when the answer is upheld.
#
# No re-reading: the gathered union already exists.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=single-valuejudge
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/single_valuejudge_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
module load Python/3.9.6-GCCcore-11.2.0
set -a; [ -f .env ] && . ./.env; set +a

BASE="${LLM_BASE_URL%/v1}"; HOST="${BASE%:*}"
B_URL="${HOST}:8001/v1"; B_MODEL="Qwen/QwQ-32B-AWQ"

SRC=$(ls -t eval/reader/*single-merged-*.jsonl 2>/dev/null | grep -v supported | head -1)
[ -z "$SRC" ] && { echo "ERROR: no merged single-paper run"; exit 1; }
echo "input: $SRC"

echo "--- waiting for the judge endpoint ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-7200} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${B_URL}/models" >/tmp/v_$$.json 2>/dev/null && grep -q '"id"' /tmp/v_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: no judge endpoint"; exit 1; }
  sleep 30
done
SERVED=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/v_$$.json)
rm -f /tmp/v_$$.json
echo "serving: ${SERVED}"
[ "${SERVED#*"$B_MODEL"}" = "$SERVED" ] && { echo "ERROR: expected $B_MODEL"; exit 1; }

VJ="${SRC%.jsonl}.valuejudged.jsonl"
cp "$SRC" "$VJ"
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" \
  .venv/bin/python -m hepcoveragekg.cli reader --check-support "$VJ" \
  --judge-url "$B_URL" --judge-model "$B_MODEL" --judge-max-tokens 3000 \
  --concurrency "${LLM_CONCURRENCY:-8}"

echo "=== RESULT: judged on the value, not the topic ==="
.venv/bin/python - "${VJ%.jsonl}.supported.jsonl" <<'PY'
import io, json, sys
from hepcoveragekg.eval import supervisor as S
recs = {q["qid"]: q for q in S.build_records()}
for r in sorted((json.loads(l) for l in io.open(sys.argv[1], encoding="utf-8")
                 if l.strip()), key=lambda x: x["qid"]):
    p = recs[r["qid"]]["provenance"]
    g = (p.get("gabriel_gold") or {}).get("value")
    print("=" * 76)
    print(f"{r['qid']}  {r['paper_id']}   supported={r.get('quote_supports')}"
          + ("   [HIS PREMISE: not in graph]" if p.get("premise_disputed") else ""))
    print(f"  Q        : {recs[r['qid']]['text'][:130]}")
    print(f"  HIS GOLD : {str(g)[:130]}")
    for a in (r.get("answers") or [])[:1]:
        print(f"  WE READ  : {a[:130]}")
    if r.get("judge_best_answer"):
        print(f"  JUDGE SAYS THE EVIDENCE SUPPORTS: {r['judge_best_answer'][:130]}")
    if r.get("support_why"):
        print(f"  why      : {r['support_why'][:200]}")
PY
echo "--- done ---"
