#!/bin/bash
# =============================================================================
# HEPCoverageKG: recover run 48367 from its own stored replies, then re-judge.
#
# 149 of that run's 732 calls were recorded as unparseable. 124 of them were
# complete, well-formed answers thrown away by two parser faults, both of which
# selected for replies quoting LaTeX -- which is to say replies quoting the cut
# values, masses and efficiencies these questions ask for. gf-10 and gf-11 came
# back with ZERO evidence across 60 calls each and were reported as "the paper
# does not say".
#
# Nothing needs re-reading to fix that: every raw reply is stored on its verdict.
# `reparse` re-reads them with the corrected parser and re-verifies each
# recovered quote against the paper, at no GPU cost -- about 40 QwQ-minutes not
# spent learning the same thing twice.
#
# The 24B arm IS re-run, for a different reason: its remaining 20 failures are
# genuine truncation. 500 completion tokens no longer fits a reply asked for
# EVERY relevant sentence when a single CMS sentence carries 400 characters of
# markup. At 1500 it fits, and with the per-call concurrency fix the whole arm
# is minutes rather than the 20 it took before.
#
# Needs the QwQ endpoint for the judge only. Both servers: hpc/serve_both.sh
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=single-rejudge
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/single_rejudge_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
module load Python/3.9.6-GCCcore-11.2.0
set -a; [ -f .env ] && . ./.env; set +a

BASE="${LLM_BASE_URL%/v1}"; HOST="${BASE%:*}"
A_URL="${HOST}:8000/v1"; A_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
B_URL="${HOST}:8001/v1"; B_MODEL="Qwen/QwQ-32B-AWQ"
STAMP="$(date +%Y%m%dT%H%M%S)"; JOB="${SLURM_JOB_ID:-$$}"

SRC_A=$(ls -t eval/reader/*single-gatherA-*.jsonl 2>/dev/null | grep -v reparsed | head -1)
SRC_B=$(ls -t eval/reader/*single-gatherB-*.jsonl 2>/dev/null | grep -v reparsed | head -1)
[ -z "$SRC_A" ] || [ -z "$SRC_B" ] && { echo "ERROR: no gather run to recover"; exit 1; }
echo "recovering: $SRC_A"
echo "            $SRC_B"

wait_for () {
  echo "--- waiting for $1 ($2) ---"
  local deadline=$(( $(date +%s) + ${READER_WAIT:-2400} ))
  until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
          "$1/models" >/tmp/w_$$.json 2>/dev/null && grep -q '"id"' /tmp/w_$$.json; do
    [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: $1 never came up"; rm -f /tmp/w_$$.json; return 1; }
    sleep 20
  done
  local served
  served=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/w_$$.json)
  rm -f /tmp/w_$$.json
  echo "    serving: $served"
  if [ "${served#*"$2"}" = "$served" ]; then echo "ERROR: expected $2"; return 1; fi
}

echo "=== REPARSE (no model calls) ==="
.venv/bin/python - "$SRC_A" "$SRC_B" <<'PY'
import sqlite3, sys
from hepcoveragekg.kg import store
from hepcoveragekg.eval import reader as R
conn = sqlite3.connect(f"file:{store.DEFAULT_DB_PATH}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
for path in sys.argv[1:]:
    info = R.reparse(path, conn=conn)
    print(f"  {path}")
    print(f"    {info['recovered']} recovered, {info['still_unparsed']} still "
          f"unparsed, {info['hard_errors']} hard errors, of {info['calls']} calls")
PY

# Re-run the cheap arm with room for a full reply. If the endpoint is gone the
# recovered arms still merge -- do not lose the recovery to a missing server.
A_NEW="eval/reader/${STAMP}-single-gatherA2-${JOB}.jsonl"
if wait_for "$A_URL" "$A_MODEL"; then
  echo "=== GATHER A2: ${A_MODEL}, 1500 completion tokens ==="
  LLM_BASE_URL="$A_URL" LLM_MODEL_NAME="$A_MODEL" READER_MAX_TOKENS=1500 \
    .venv/bin/python -m hepcoveragekg.cli reader --scope single --out "$A_NEW" \
    --repeats 3 --no-cascade --concurrency "${LLM_CONCURRENCY:-16}" || A_NEW=""
else
  echo "!!! no 24B server; merging the recovered arms alone"; A_NEW=""
fi

M_OUT="eval/reader/${STAMP}-single-merged-${JOB}.jsonl"
echo "=== MERGE ==="
.venv/bin/python - "$M_OUT" "${SRC_A%.jsonl}.reparsed.jsonl" \
                   "${SRC_B%.jsonl}.reparsed.jsonl" ${A_NEW:+"$A_NEW"} <<'PY'
import sys
from hepcoveragekg.eval import reader as R
info = R.merge_gathered(sys.argv[2:], sys.argv[1])
for k, v in info.items():
    print(f"  {k}: {v}")
PY

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
# What the same questions said before the parser was fixed (run 48367).
OLD = {"gf-06": "True/3q", "gf-10": "False/0q", "gf-11": "False/0q",
       "gf-12": "False/3q", "gf-13": "True/1q", "gf-14": "False/2q",
       "gf-15": "True/4q"}
for r in sorted((json.loads(l) for l in io.open(sys.argv[1], encoding="utf-8")
                 if l.strip()), key=lambda x: x["qid"]):
    p = recs[r["qid"]]["provenance"]
    tag = "   [HIS PREMISE: not in graph]" if p.get("premise_disputed") else ""
    print(f"\n{r['qid']} {r['paper_id']}   was {OLD.get(r['qid'],'?')}"
          f"   ->  now {r['answer']}/{len(r.get('all_quotes') or [])}q"
          f"   judge={r.get('quote_supports')}{tag}")
    print(f"   his gold: {str((p.get('gabriel_gold') or {}).get('value'))[:100]}")
    for a in (r.get("answers") or [])[:2]:
        print(f"   we read : {a[:100]}")
    if r.get("support_why"):
        print(f"   judge   : {r['support_why'][:160]}")
PY
echo "--- done ---"
