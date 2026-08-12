#!/bin/bash
# =============================================================================
# HEPCoverageKG: re-run gf-01 with its three conditions asked separately.
#
# gf-01 scored 2/18. Every single-condition question scored well (gf-04 9/10,
# gf-02 6/6, gf-03 3/4), so the hypothesis is the SHAPE, not the subject: a
# ~12,000-character window almost never establishes "a search" AND "b-tagged
# jets" AND "missing transverse momentum" at once, so every window honestly says
# no and the paper comes back no.
#
# TWO PHASES, and the first gates the second.
#
#   diagnostic  the 16 papers in his gold we wrongly said no to, PLUS 6 papers
#               NOT in his gold. Recovering the 16 alone would prove nothing --
#               a change that makes everything say yes would do that too, which
#               is exactly the trap QwQ fell into. The 6 controls are the check.
#   full        all 60 papers. Runs REGARDLESS -- the diagnostic explains the
#               number, it does not withhold it.
#
# Runs unattended overnight, so it must fail loudly rather than quietly produce
# a number. Every check below exists because something silently didn't.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=gf01
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/gf01_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
REPO="${REPO:-$HOME/HEPCoverageKG}"
cd "$REPO"
module load Python/3.9.6-GCCcore-11.2.0

_OVERRIDE_MODEL="${LLM_MODEL_NAME:-}"
_OVERRIDE_URL="${LLM_BASE_URL:-}"
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$_OVERRIDE_MODEL" ] && LLM_MODEL_NAME="$_OVERRIDE_MODEL"
[ -n "$_OVERRIDE_URL" ] && LLM_BASE_URL="$_OVERRIDE_URL"

STAMP="$(date +%Y%m%dT%H%M%S)"
JOB="${SLURM_JOB_ID:-$$}"
mkdir -p eval/reader

echo "node    : $(hostname)"
echo "endpoint: ${LLM_BASE_URL}"
echo "model   : ${LLM_MODEL_NAME}"

# Wait for the server, then verify it serves what we asked for. Both checks have
# earned their place: a job that exits on arrival wastes a queue slot that took
# hours, and a mismatched model once cost 436 calls of 404s (D-049).
echo "--- waiting for the server ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-2400} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${LLM_BASE_URL%/v1}/v1/models" >/tmp/m_$$.json 2>/dev/null \
      && grep -q '"id"' /tmp/m_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: server never came up"; exit 1; }
  sleep 20
done
SERVED=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/m_$$.json)
rm -f /tmp/m_$$.json
echo "serving : ${SERVED}"
if [ "${SERVED#*"$LLM_MODEL_NAME"}" = "$SERVED" ]; then
  echo "ERROR: endpoint serves '${SERVED}', configured for '${LLM_MODEL_NAME}'"; exit 1
fi

# 16 gold papers we missed + 6 controls that are NOT in his gold.
MISSED="2004.14060,2006.05880,2010.14293,2012.03799,2012.08600,2102.01444,2106.01676,2106.14246,2201.11585,2202.08676,2211.08028,2302.05225,2307.01094,2403.01556,2506.13565,2508.13900"
CONTROLS="2001.06899,2110.11231,2207.12246,2208.12095,2308.02285,2309.14442"

DIAG="eval/reader/${STAMP}-gf01-diagnostic-${JOB}.jsonl"
echo "=== PHASE 1: diagnostic (22 papers) ==="
.venv/bin/python -m hepcoveragekg.cli reader --conditions gf-01 \
  --only-papers "${MISSED},${CONTROLS}" --out "$DIAG" \
  --repeats "${READER_REPEATS:-2}" --concurrency "${LLM_CONCURRENCY:-16}"

.venv/bin/python - "$DIAG" "$MISSED" "$CONTROLS" <<'PY'
import json, sys
rows = {r["paper_id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
missed, controls = sys.argv[2].split(","), sys.argv[3].split(",")
rec = sum(1 for p in missed if rows.get(p, {}).get("answer") is True)
fp  = sum(1 for p in controls if rows.get(p, {}).get("answer") is True)
print(f"  recovered {rec}/{len(missed)} of the papers we wrongly missed")
print(f"  said yes to {fp}/{len(controls)} papers NOT in his gold")
# Both halves matter. Recovering the misses is the point; but a change that
# simply says yes more often would do that too, so the controls decide whether
# the improvement is real or just a looser threshold.
# ADVISORY, not blocking. The full 60 runs tonight regardless -- the diagnostic
# is here to tell us WHY the number came out as it did, not to withhold it.
if rec < len(missed) * 0.4:
    print("DIAGNOSTIC: weak recovery -- the question shape may not be the problem")
elif fp > len(controls) * 0.5:
    print("DIAGNOSTIC: recovers the misses but also fires on controls -- looser, not better")
else:
    print("DIAGNOSTIC: recovers the misses without firing on the controls -- the shape was the problem")
PY

FULL="eval/reader/${STAMP}-gf01-full-${JOB}.jsonl"
echo "=== PHASE 2: all 60 papers (runs regardless of the diagnostic) ==="
.venv/bin/python -m hepcoveragekg.cli reader --conditions gf-01 \
  --out "$FULL" --repeats "${READER_REPEATS:-2}" \
  --concurrency "${LLM_CONCURRENCY:-16}"

echo "=== scored against his gold ==="
.venv/bin/python - "$FULL" <<'PY'
import json, sys
from hepcoveragekg.eval import supervisor as S
gold = set(next(q for q in S.SUPERVISOR_QUESTIONS if q["qid"] == "gf-01")["gabriel_gold"]["papers"])
rows = list(map(json.loads, open(sys.argv[1], encoding="utf-8")))
yes = {r["paper_id"] for r in rows if r["answer"] is True}
print(f"  found {len(yes)}   gold {len(gold)}   overlap {len(yes & gold)}/{len(gold)}   (was 2/18)")
print(f"  false positives: {len(yes - gold)}")
missing = sorted(gold - yes)
if missing:
    print(f"  still missing: {missing}")
    for p in missing[:5]:
        row = next((r for r in rows if r["paper_id"] == p), None)
        if row:
            print(f"    {p}: missing condition(s) -> {row['missing']}")
PY
echo "--- done ---"
