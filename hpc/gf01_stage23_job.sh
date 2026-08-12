#!/bin/bash
# =============================================================================
# HEPCoverageKG: stages 2 and 3 for gf-01, after the 24B conditions pass.
#
# The gf-01 job runs the 24B ONLY. This finishes the pipeline:
#
#   stage 2  QwQ re-reads the papers the 24B rejected. On the sweep it recovered
#            37 of 317 -- including both ABCD papers the 24B missed, neither of
#            which contains the word "ABCD". The 24B needs the term; QwQ
#            recognises the method.
#   stage 3  a judge checks every YES from either model. Not optional: the
#            aggregation is OR-over-windows, so one window with a verified quote
#            flips a paper. At a 7% per-window error rate over 14 windows that is
#            a 64% chance of a wrong paper-level answer -- and a BETTER reader
#            raises it, because it finds more per window.
#
# Runs on the QwQ server, which needs its own allocation. Submitted with
# --dependency=afterany on the 24B job so it starts whether or not that one
# ended cleanly; there is no point holding a GPU while waiting for perfection.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=gf01-s23
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/gf01_s23_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
REPO="${REPO:-$HOME/HEPCoverageKG}"
cd "$REPO"
module load Python/3.9.6-GCCcore-11.2.0

_OM="${LLM_MODEL_NAME:-}"; _OU="${LLM_BASE_URL:-}"
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$_OM" ] && LLM_MODEL_NAME="$_OM"
[ -n "$_OU" ] && LLM_BASE_URL="$_OU"

# The newest full gf-01 run. Taken by mtime rather than passed in, so a rerun of
# the 24B stage does not need this script edited to match.
FULL=$(ls -t eval/reader/*gf01-full-*.jsonl 2>/dev/null | head -1)
[ -z "$FULL" ] && { echo "ERROR: no gf-01 full run found -- did stage 1 finish?"; exit 1; }
echo "input   : $FULL   ($(wc -l < "$FULL") papers)"
echo "endpoint: ${LLM_BASE_URL}"
echo "model   : ${LLM_MODEL_NAME}"

echo "--- waiting for the QwQ server ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-2400} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${LLM_BASE_URL%/v1}/v1/models" >/tmp/q_$$.json 2>/dev/null \
      && grep -q '"id"' /tmp/q_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: QwQ server never came up"; exit 1; }
  sleep 20
done
SERVED=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/q_$$.json)
rm -f /tmp/q_$$.json
echo "serving : ${SERVED}"
[ "${SERVED#*"$LLM_MODEL_NAME"}" = "$SERVED" ] && {
  echo "ERROR: endpoint serves '${SERVED}', configured for '${LLM_MODEL_NAME}'"; exit 1; }

echo "=== STAGE 2: QwQ re-reads what the 24B rejected ==="
READER_MAX_TOKENS=3000 .venv/bin/python -m hepcoveragekg.cli reader \
  --recheck "$FULL" --concurrency "${LLM_CONCURRENCY:-10}"

RECHECKED="${FULL%.jsonl}.rechecked.jsonl"
[ -f "$RECHECKED" ] || { echo "ERROR: stage 2 produced no output"; exit 1; }

echo "=== STAGE 3: judge every YES ==="
.venv/bin/python -m hepcoveragekg.cli reader \
  --check-support "$RECHECKED" \
  --judge-url "$LLM_BASE_URL" --judge-model "$LLM_MODEL_NAME" \
  --judge-max-tokens 2500 --concurrency "${LLM_CONCURRENCY:-10}"

echo "=== FINAL: gf-01 against his gold, through the whole pipeline ==="
.venv/bin/python - "${RECHECKED%.jsonl}.supported.jsonl" <<'PY'
import json, sys
from hepcoveragekg.eval import supervisor as S
gold = set(next(q for q in S.SUPERVISOR_QUESTIONS if q["qid"] == "gf-01")["gabriel_gold"]["papers"])
rows = list(map(json.loads, open(sys.argv[1], encoding="utf-8")))
yes = {r["paper_id"] for r in rows if r["answer"] is True}
print(f"  found {len(yes)}   gold {len(gold)}   overlap {len(yes & gold)}/{len(gold)}   (was 2/18)")
print(f"  false positives {len(yes - gold)}")
rec = [r for r in rows if r.get("recovered_by")]
print(f"  recovered by QwQ: {len(rec)}")
print(f"  still missing   : {sorted(gold - yes)}")
PY
echo "--- done ---"
