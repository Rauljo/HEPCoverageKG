#!/bin/bash
# =============================================================================
# HEPCoverageKG: isolate the parser fix. One variable, not three.
#
# The first comparison of "before vs after the parser fix" was worthless, and
# both confounds were self-inflicted:
#
#   1. The BEFORE file was judged on 08-09 by the 24B; the AFTER file by QwQ.
#      Two different judges, and they disagree by design -- on 2004.04545 the
#      24B said "mentions the use of b-tagged jets, which is part of the event
#      selection" and passed it, while QwQ said "explicitly mentions b-tagged
#      jets ... but it does not mention missing transverse momentum" and failed
#      it. QwQ is right: gf-01 is a three-part conjunction and that sentence
#      shows one part. The apparent regression is a better judge, not a worse
#      parser.
#
#   2. The gf-01 conditions re-run used repeats=3 where the original used 2.
#      Consensus needs the repeats within a window to AGREE, so a third sample
#      can only ever make a yes harder to reach. That alone could produce the
#      6/18 -> 4/18 drop, with no other cause.
#
# So: judge the ORIGINAL stage-1 sweep with QwQ, on identical settings to the
# reparsed run. Same stage, same judge, same everything -- only the parser
# differs. And re-run gf-01 at repeats=2, the value the baseline used.
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=parser-control
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/parser_control_%j.out
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

echo "--- waiting for the judge endpoint ---"
deadline=$(( $(date +%s) + ${READER_WAIT:-7200} ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${B_URL}/models" >/tmp/p_$$.json 2>/dev/null && grep -q '"id"' /tmp/p_$$.json; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: no judge endpoint"; exit 1; }
  sleep 30
done
rm -f /tmp/p_$$.json

SRC="eval/reader/20260809T202844-sweep-48330.jsonl"
CTL="eval/reader/sweep-48330.oldparser-qwqjudge.jsonl"
cp "$SRC" "$CTL"
echo "=== CONTROL: the ORIGINAL parse, judged by QwQ ==="
LLM_BASE_URL="$B_URL" LLM_MODEL_NAME="$B_MODEL" \
  .venv/bin/python -m hepcoveragekg.cli reader --check-support "$CTL" \
  --judge-url "$B_URL" --judge-model "$B_MODEL" --judge-max-tokens 2500 \
  --concurrency "${LLM_CONCURRENCY:-10}"

echo "=== PARSER EFFECT, everything else held fixed ==="
.venv/bin/python - "${CTL%.jsonl}.supported.jsonl" \
   "eval/reader/20260809T202844-sweep-48330.reparsed.supported.jsonl" <<'PY'
import io, json, os, sys
from hepcoveragekg.eval import supervisor as S
gold = {q["qid"]: set(q["provenance"]["gabriel_gold"]["papers"])
        for q in S.build_records()
        if (q["provenance"].get("gabriel_gold") or {}).get("papers")}
def load(p):
    out = {}
    if not os.path.exists(p): return out
    for l in io.open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out.setdefault(r["qid"], {})[r["paper_id"]] = r["answer"]
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"{'qid':7} {'gold':>4} | {'old parse':>9} {'hit':>4} {'FP':>4}"
      f" | {'fixed':>6} {'hit':>4} {'FP':>4}")
ta = tb = 0
for qid in sorted(gold):
    if qid not in b: continue
    g = gold[qid]
    ya = {p for p, v in a.get(qid, {}).items() if v is True}
    yb = {p for p, v in b[qid].items() if v is True}
    ta += len(ya & g); tb += len(yb & g)
    print(f"{qid:7} {len(g):4} | {len(ya):9} {len(ya&g):4} {len(ya-g):4}"
          f" | {len(yb):6} {len(yb&g):4} {len(yb-g):4}")
print(f"\n  gold found: {ta} -> {tb}   (same stage, same judge, only the parser differs)")
PY
echo "--- done ---"
