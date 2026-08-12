#!/bin/bash
# =============================================================================
# HEPCoverageKG: re-judge gf-01 with ALL of each paper's evidence.
#
# Last night the judge was handed ONE sentence and the whole three-part question.
# One sentence rarely shows a search AND b-tagged jets AND missing transverse
# momentum, so it rejected four papers whose conditions were all confirmed --
# narrating the evidence as it did so:
#
#   "The sentence mentions a search (not a measurement) and explicitly includes
#    missing transverse momentum..."                       -> downgraded
#
# No re-reading is needed: the per-condition quotes are already in the stage-1
# output, they were only flattened to the first one. This reconstructs the set
# and judges the conjunction against all of it.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=gf01-rejudge
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/gf01_rejudge_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
module load Python/3.9.6-GCCcore-11.2.0

_OM="${LLM_MODEL_NAME:-}"; _OU="${LLM_BASE_URL:-}"
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$_OM" ] && LLM_MODEL_NAME="$_OM"
[ -n "$_OU" ] && LLM_BASE_URL="$_OU"

SRC=$(ls -t eval/reader/*gf01-full-*.rechecked.jsonl 2>/dev/null | head -1)
[ -z "$SRC" ] && { echo "ERROR: no rechecked gf-01 run found"; exit 1; }
FIXED="${SRC%.jsonl}.withquotes.jsonl"
echo "input : $SRC"

# Rebuild the per-condition quote set from what stage 1 already recorded.
.venv/bin/python - "$SRC" "$FIXED" <<'PY'
import io, json, sys
n = 0
with io.open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in io.open(sys.argv[1], encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        conds = r.get("conditions") or {}
        quotes = {c: v.get("quote") for c, v in conds.items() if v.get("quote")}
        if quotes:
            r["quotes"] = quotes
            n += 1
        # A judged row from the previous pass must be re-judged, not skipped.
        r.pop("quote_supports", None); r.pop("support_why", None)
        r.pop("downgraded", None)
        if r.get("recheck_answer") is True or (conds and all(
                v.get("answer") is True for v in conds.values())):
            r["answer"] = True
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  rebuilt the quote set for {n} papers")
PY

echo "--- waiting for the judge endpoint ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-2400} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${LLM_BASE_URL%/v1}/v1/models" >/tmp/j_$$.json 2>/dev/null \
      && grep -q '"id"' /tmp/j_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: server never came up"; exit 1; }
  sleep 20
done
SERVED=$(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/j_$$.json)
rm -f /tmp/j_$$.json
echo "serving: ${SERVED}"
[ "${SERVED#*"$LLM_MODEL_NAME"}" = "$SERVED" ] && { echo "ERROR: wrong model"; exit 1; }

.venv/bin/python -m hepcoveragekg.cli reader --check-support "$FIXED" \
  --judge-url "$LLM_BASE_URL" --judge-model "$LLM_MODEL_NAME" \
  --judge-max-tokens 2500 --concurrency "${LLM_CONCURRENCY:-10}"

echo "=== gf-01 after judging on the full evidence ==="
.venv/bin/python - "${FIXED%.jsonl}.supported.jsonl" <<'PY'
import io, json, sys
from hepcoveragekg.eval import supervisor as S
gold = set(next(q for q in S.SUPERVISOR_QUESTIONS if q["qid"] == "gf-01")["gabriel_gold"]["papers"])
rows = [json.loads(l) for l in io.open(sys.argv[1], encoding="utf-8") if l.strip()]
yes = {r["paper_id"] for r in rows if r["answer"] is True}
up = sum(1 for r in rows if r.get("quote_supports") is True)
dn = sum(1 for r in rows if r.get("quote_supports") is False)
print(f"  judge: {up} upheld, {dn} downgraded"
      + (f", precision {up/(up+dn):.0%}" if up + dn else ""))
print(f"  found {len(yes)}   gold {len(gold)}   overlap {len(yes & gold)}/{len(gold)}")
print(f"  (was 2/18 combined, 4/18 after last night's judge)")
print(f"  false positives {len(yes - gold)}")
print(f"  still missing   {sorted(gold - yes)}")
PY
echo "--- done ---"
