#!/bin/bash
# =============================================================================
# HEPCoverageKG: gf-01 again, because this is the one run that could not be
# recovered from storage.
#
# gf-01 is the worst-scoring question in the set -- 18 gold papers, 6 found --
# and it is a three-part conjunction: a SEARCH, and b-tagged jets, and missing
# transverse momentum. It is asked one condition at a time with the AND computed
# over the whole paper (D-058).
#
# When the JSON parser was fixed, the sweep and the single-paper runs gave back
# 599 discarded calls for free, because they store every raw reply. The
# conditions path did not store them. So gf-01 -- the question with the most to
# gain, since a lost YES on ANY one condition fails the whole conjunction -- is
# the only one that has to be read again from scratch.
#
# The path now keeps its raw replies. This will not happen twice.
#
# Two things are different from run 48358 besides the parser: the concurrency
# gate bounds CALLS rather than reads, so the arm no longer drains to one call
# in flight; and the judge sees every condition's quote at once.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=gf01-rerun
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/gf01_rerun_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem 24G
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
OUT="eval/reader/${STAMP}-gf01-rerun-${JOB}.jsonl"

wait_for () {
  echo "--- waiting for $1 ($2) ---"
  local deadline=$(( $(date +%s) + ${READER_WAIT:-7200} ))
  until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
          "$1/models" >/tmp/w_$$.json 2>/dev/null && grep -q '"id"' /tmp/w_$$.json; do
    [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: $1 never came up"; rm -f /tmp/w_$$.json; return 1; }
    sleep 30
  done
  local served
  served=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/w_$$.json)
  rm -f /tmp/w_$$.json
  echo "    serving: $served"
  if [ "${served#*"$2"}" = "$served" ]; then echo "ERROR: expected $2"; return 1; fi
}

wait_for "$A_URL" "$A_MODEL"
echo "=== gf-01 CONDITIONS: ${A_MODEL} ==="
LLM_BASE_URL="$A_URL" LLM_MODEL_NAME="$A_MODEL" READER_MAX_TOKENS=500 \
  .venv/bin/python -m hepcoveragekg.cli reader --conditions gf-01 --out "$OUT" \
  --repeats 3 --concurrency "${LLM_CONCURRENCY:-16}"

wait_for "$B_URL" "$B_MODEL"
echo "=== JUDGE (QwQ, on the whole condition set) ==="
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" \
  .venv/bin/python -m hepcoveragekg.cli reader --check-support "$OUT" \
  --judge-url "$B_URL" --judge-model "$B_MODEL" --judge-max-tokens 2500 \
  --concurrency "${LLM_CONCURRENCY:-10}"

echo "=== RESULT ==="
.venv/bin/python - "${OUT%.jsonl}.supported.jsonl" <<'PY'
import io, json, sys
from collections import Counter
from hepcoveragekg.eval import supervisor as S
gold = set(next(q for q in S.build_records()
                if q["qid"] == "gf-01")["provenance"]["gabriel_gold"]["papers"])
rows = [json.loads(l) for l in io.open(sys.argv[1], encoding="utf-8") if l.strip()]
yes = {r["paper_id"] for r in rows if r["answer"] is True}
up = sum(1 for r in rows if r.get("quote_supports") is True)
dn = sum(1 for r in rows if r.get("quote_supports") is False)
print(f"  judge: {up} upheld, {dn} downgraded"
      + (f", precision {up/(up+dn):.0%}" if up + dn else ""))
print(f"  found {len(yes)}   gold {len(gold)}   overlap {len(yes & gold)}/{len(gold)}")
print("  history: 2/18 combined -> 4/18 judged -> 6/18 judged on full evidence")
print(f"  false positives {len(yes - gold)}")
print(f"  still missing   {sorted(gold - yes)}")
missing = Counter()
for r in rows:
    if r["paper_id"] in gold and r["answer"] is not True:
        for c in (r.get("missing") or []):
            missing[c[:60]] += 1
print("\n  WHICH condition loses the gold papers:")
for c, n in missing.most_common():
    print(f"    {n:3}x  {c}")
PY
echo "--- done ---"
