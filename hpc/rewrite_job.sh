#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=hepkg-rewrite
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/rewrite_%j.out
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# =============================================================================
# Rewrite the generated questions with the REWRITER model (S-67).
#
# Points at port 8000, which is the Mistral job -- the answerer runs on 8001, so
# both are served at once and nothing has to be killed and reloaded between the
# rewriting and the evaluation.
#
# Runs on a compute node rather than the login node (capped, shared) and rather
# than a laptop (sleeps when the lid closes, taking the SSH tunnel with it).
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG

# ORDER MATTERS. `.env` is sourced FIRST, then the overrides -- sourcing it
# afterwards silently put LLM_MODEL_NAME back to the ANSWERER (Qwen), so every
# rewrite call asked the Mistral server for a model it does not serve and got a
# 404. All 436 calls "errored" with no clue why.
#
# The key lives in .env and nothing else here loads it: the eval path works only
# because cli.py calls load_dotenv(), while this script uses the library directly.
set -a; . ./.env; set +a

export LLM_BASE_URL="http://compute-gpu-0-1:8000/v1"
export LLM_MODEL_NAME="mistralai/Mistral-Small-24B-Instruct-2501"

echo "host=$(hostname)  rewriter=${LLM_MODEL_NAME}"
.venv/bin/python - <<'PY'
import os
from hepcoveragekg.eval import questions as Q, reword as R

MODEL = os.environ["LLM_MODEL_NAME"]

tb = list(Q.load("eval/questions/dev-2026-08-03-conceptB.jsonl"))
out, st = R.reword(tb, mode="faithful", model=MODEL, attempts=2)
Q.save(out, "eval/questions/dev-2026-08-03-conceptB-reworded.jsonl")
print("FAITHFUL (Tier B):", st, flush=True)

rt = list(Q.load("eval/questions/dev-2026-08-03-retrieval.jsonl"))
out2, st2 = R.reword(rt, mode="vague", model=MODEL, attempts=3)
Q.save(out2, "eval/questions/dev-2026-08-03-retrieval-reworded.jsonl")
print("VAGUE (retrieval):", st2, flush=True)
for o in out2[:6]:
    print(f"   target: {o.provenance['label'][:34]}")
    print(f"   asked:  {o.text[:78]}")

# Fail loudly if nothing survived. A job that produces no questions must NOT
# satisfy an `afterok` dependency -- otherwise the next job runs on an empty file
# and reports a clean, meaningless result.
if not out or not out2:
    raise SystemExit(f"rewrite produced nothing: faithful={len(out)} vague={len(out2)}")
PY
