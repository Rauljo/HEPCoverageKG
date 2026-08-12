#!/bin/bash
# =============================================================================
# HEPCoverageKG: the single-paper questions, through the same cascade as the rest.
#
# gf-06 and gf-10..gf-15 have been running ONE model, ONE pass, no judge, while
# the sweep questions got 24B -> QwQ -> judge. That is why they have been the
# flakiest results in the set, and it was never a deliberate choice.
#
#   gather A   Mistral-24B  reads every window, returns every sentence bearing
#              on ANY part of the question
#   gather B   QwQ-32B      the same, independently
#   merge      union the two. Recall is a union, not a vote: the 24B needs a
#              concept named in the paper's own words and QwQ reasons to weaker
#              connections, so demanding agreement would discard what the second
#              model was added to find.
#   judge      sees the question and everything gathered, and decides whether the
#              parts add up.
#
# Neither reader is asked whether the answer is complete. Asking that made gf-11
# and gf-14 go from a verified quote each to ZERO found across 32 and 54 calls --
# a recall stage told to assess completeness withholds evidence.
#
# Both servers must be up: 24B on :8000, QwQ on :8001.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=single-cascade
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/single_cascade_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
module load Python/3.9.6-GCCcore-11.2.0
set -a; [ -f .env ] && . ./.env; set +a

BASE="${LLM_BASE_URL%/v1}"
HOST="${BASE%:*}"
A_URL="${HOST}:8000/v1"; A_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
B_URL="${HOST}:8001/v1"; B_MODEL="Qwen/QwQ-32B-AWQ"
STAMP="$(date +%Y%m%dT%H%M%S)"; JOB="${SLURM_JOB_ID:-$$}"

wait_for () {   # url, expected model
  echo "--- waiting for $1 ($2) ---"
  local deadline=$(( $(date +%s) + ${READER_WAIT:-2400} ))
  until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
          "$1/models" >/tmp/w_$$.json 2>/dev/null && grep -q '"id"' /tmp/w_$$.json; do
    [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: $1 never came up"; return 1; }
    sleep 20
  done
  local served
  served=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/w_$$.json)
  rm -f /tmp/w_$$.json
  echo "    serving: $served"
  # Refusing a mismatch has already saved 436 wasted calls once (D-049).
  if [ "${served#*"$2"}" = "$served" ]; then echo "ERROR: expected $2"; return 1; fi
}

A_OUT="eval/reader/${STAMP}-single-gatherA-${JOB}.jsonl"
B_OUT="eval/reader/${STAMP}-single-gatherB-${JOB}.jsonl"
M_OUT="eval/reader/${STAMP}-single-merged-${JOB}.jsonl"

# --no-cascade on BOTH arms. With the cascade on, the router reads only the
# sections it thinks are relevant and, in EXTRACTION mode, a hit there means the
# rest of the paper is never read. That is a precision device, and this is a
# recall stage: the parts of a multi-part answer sit in different sections by
# construction (the algorithm in object selection, its working point in a table,
# the comparison in the results). Read every window of every paper. It costs
# ~300 calls per model across 7 reads, which is nothing.
#
# QwQ first, deliberately. It is the scarce resource -- a 4h GPU reservation
# that is already part-spent -- while the 24B server is cheap to restart. Doing
# the cheap arm first would burn the expensive arm's remaining clock waiting.
wait_for "$B_URL" "$B_MODEL"
echo "=== GATHER B: ${B_MODEL} ==="
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" READER_MAX_TOKENS=3000 \
  .venv/bin/python -m hepcoveragekg.cli reader --scope single --out "$B_OUT" \
  --repeats 3 --no-cascade --concurrency "${LLM_CONCURRENCY:-8}"

# If the 24B never arrives -- it has one GPU to draw from and one of the three
# on this node fails on first inference -- do NOT take QwQ's work down with it.
# A union of one is still a better evidence set than what these questions had.
if wait_for "$A_URL" "$A_MODEL"; then
  echo "=== GATHER A: ${A_MODEL} ==="
  LLM_BASE_URL="$A_URL" LLM_MODEL_NAME="$A_MODEL" READER_MAX_TOKENS=500 \
    .venv/bin/python -m hepcoveragekg.cli reader --scope single --out "$A_OUT" \
    --repeats 3 --no-cascade --concurrency "${LLM_CONCURRENCY:-12}"
else
  echo "!!! no 24B server; merging QwQ's gather alone"
fi

echo "=== MERGE ==="
.venv/bin/python - "$A_OUT" "$B_OUT" "$M_OUT" <<'PY'
import sys
from hepcoveragekg.eval import reader as R
info = R.merge_gathered([sys.argv[1], sys.argv[2]], sys.argv[3])
for k, v in info.items():
    print(f"  {k}: {v}")
PY

# QwQ may have expired while the 24B was loading. Block for a replacement
# rather than judging on the literal reader -- weighing whether a set of
# sentences adds up to a multi-part claim is the one thing the 24B is measurably
# bad at, and it is the entire job here.
wait_for "$B_URL" "$B_MODEL"
echo "=== JUDGE (QwQ, on everything gathered) ==="
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" \
  .venv/bin/python -m hepcoveragekg.cli reader --check-support "$M_OUT" \
  --judge-url "$B_URL" --judge-model "$B_MODEL" --judge-max-tokens 2500 \
  --concurrency "${LLM_CONCURRENCY:-8}"

echo "=== RESULT ==="
.venv/bin/python - "${M_OUT%.jsonl}.supported.jsonl" <<'PY'
import io, json, sys
from hepcoveragekg.eval import supervisor as S
recs = {q["qid"]: q for q in S.build_records()}
OLD = {"gf-06": "True", "gf-10": "False", "gf-11": "True", "gf-12": "False",
       "gf-13": "False", "gf-14": "True", "gf-15": "False"}
for r in sorted((json.loads(l) for l in io.open(sys.argv[1], encoding="utf-8")
                 if l.strip()), key=lambda x: x["qid"]):
    p = recs[r["qid"]]["provenance"]
    tag = "  [HIS PREMISE: not in graph]" if p.get("premise_disputed") else ""
    print(f"{r['qid']} {r['paper_id']}  was={OLD.get(r['qid'],'?'):5} "
          f"now={str(r['answer']):5}  quotes={len(r.get('all_quotes') or [])}"
          f"  judge={r.get('quote_supports')}{tag}")
    print(f"   gold: {str((p.get('gabriel_gold') or {}).get('value'))[:72]}")
    for a in (r.get("answers") or [])[:1]:
        print(f"   ours: {a[:72]}")
PY
echo "--- done ---"
