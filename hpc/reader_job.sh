#!/bin/bash
# =============================================================================
# HEPCoverageKG: the reference reader on DIAS.
#
# Reads the PAPERS to answer the supervisor's evaluation questions, independent
# of the knowledge graph. His gold is graph-agreement gold — computed by
# filtering his own pipeline's export — so scoring our graph against it measures
# whether two pipelines built from one extraction agree with each other. Where
# the extraction dropped something, both drop it and both score 100%. This is
# the check that can actually see that.
#
# COMPUTE partition (the CPU one; "CPU" is not a partition name here). The model is served by a SEPARATE job (hpc/serve_vllm*.sh). This one
# just makes HTTP calls, so asking for a GPU here would idle one for hours.
#
# Submit AFTER the server is up and answering:
#   sbatch --dependency=after:<server_jobid> hpc/reader_job.sh
#   # or, once /v1/models responds:
#   READER_SCOPE=sweep sbatch hpc/reader_job.sh
#
# Two runs, because they have very different shapes (D-056):
#   sweep   7 questions x 60 papers, section-cascade  -> ~420 paper-reads
#   single  7 questions x 1-2 named papers, no cascade -> ~9 paper-reads
#
# compute-0-1 is excluded: it accepts jobs and runs nothing (D-049).
# =============================================================================
#SBATCH -p COMPUTE
#SBATCH --job-name=reader-hepckg
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/reader_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem 16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=compute-0-1

set -euo pipefail

REPO="${REPO:-$HOME/HEPCoverageKG}"
cd "$REPO"

# The venv was built against this module's interpreter; without it .venv/bin/python
# dies with "libpython3.9.so.1.0: cannot open shared object file", which reads like
# a broken venv rather than a missing module.
module load Python/3.9.6-GCCcore-11.2.0

# .env FIRST, then the overrides RE-APPLIED. Sourcing .env alone is not enough:
# `sbatch --export=ALL,LLM_MODEL_NAME=...` puts the override in the environment
# BEFORE this script runs, and `set -a; . .env` then overwrites it with whatever
# .env says. That is D-049 happening again -- the first run of this job was
# configured for Qwen-72B while the server served Mistral-24B, purely because
# .env still named the old model.
#
# So capture what was passed in, source .env for everything unset, then put the
# explicit values back on top. The endpoint check below is the backstop.
_OVERRIDE_MODEL="${LLM_MODEL_NAME:-}"
_OVERRIDE_URL="${LLM_BASE_URL:-}"
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$_OVERRIDE_MODEL" ] && LLM_MODEL_NAME="$_OVERRIDE_MODEL"
[ -n "$_OVERRIDE_URL" ] && LLM_BASE_URL="$_OVERRIDE_URL"

SCOPE="${READER_SCOPE:-sweep}"
REPEATS="${READER_REPEATS:-3}"
TEMPERATURE="${READER_TEMPERATURE:-0.3}"
CONCURRENCY="${LLM_CONCURRENCY:-16}"
STAMP="$(date +%Y%m%dT%H%M%S)"
# SLURM_JOB_ID in the name: four jobs starting in the same second once wrote to
# one file and truncated each other (D-049).
OUT="eval/reader/${STAMP}-${SCOPE}-${SLURM_JOB_ID:-$$}.jsonl"

echo "node       : $(hostname)"
echo "endpoint   : ${LLM_BASE_URL:-unset}"
echo "model      : ${LLM_MODEL_NAME:-unset}"
echo "scope      : ${SCOPE}   repeats=${REPEATS}  temp=${TEMPERATURE}  conc=${CONCURRENCY}"
echo "out        : ${OUT}"

# Wait for the server rather than failing on arrival. Submitted with
# `--dependency=after:<server>`, this job starts the moment the server job is
# ALLOCATED, but vLLM then spends several minutes loading weights. Exiting
# immediately would waste the queue slot we just waited hours for.
WAIT_SECONDS="${READER_WAIT:-1800}"
echo "--- waiting up to ${WAIT_SECONDS}s for ${LLM_BASE_URL} ---"
deadline=$(( $(date +%s) + WAIT_SECONDS ))
until curl -sS --max-time 10 -H "Authorization: Bearer ${LLM_API_KEY}" \
        "${LLM_BASE_URL%/v1}/v1/models" >/tmp/models_$$.json 2>/dev/null \
      && grep -q '"id"' /tmp/models_$$.json; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: server never came up within ${WAIT_SECONDS}s"; exit 1
  fi
  sleep 20
done

# Say what is ACTUALLY being served. Every long DIAS failure so far has been an
# endpoint that was not what it claimed: a model name reverted by env ordering,
# two servers sharing a port via SO_REUSEPORT, a node that accepts work and runs
# none (D-049).
SERVED=$(python3 -c \
  'import json,sys; print(",".join(m["id"] for m in json.load(open(sys.argv[1])).get("data",[])))' \
  /tmp/models_$$.json)
rm -f /tmp/models_$$.json
echo "serving    : ${SERVED}"
if [ -n "${LLM_MODEL_NAME:-}" ] && [ "${SERVED#*"$LLM_MODEL_NAME"}" = "$SERVED" ]; then
  echo "ERROR: endpoint serves '${SERVED}' but we are configured for '${LLM_MODEL_NAME}'."
  echo "       Refusing to run — this is how 436 calls were wasted on 404s before."
  exit 1
fi

mkdir -p eval/reader

# `single` reads whole papers: for 1-2 named papers the cascade saves nothing
# and risks missing an answer that lives outside the routed sections.
CASCADE_FLAG=""
[ "$SCOPE" = "single" ] && CASCADE_FLAG="--no-cascade"

EXTRA=""
[ -n "${READER_QID:-}" ] && EXTRA="$EXTRA --qid ${READER_QID}"
[ -n "${READER_LIMIT_PAPERS:-}" ] && EXTRA="$EXTRA --limit-papers ${READER_LIMIT_PAPERS}"

# shellcheck disable=SC2086
.venv/bin/python -m hepcoveragekg.cli reader \
  --scope "$SCOPE" \
  --out "$OUT" \
  --repeats "$REPEATS" \
  --temperature "$TEMPERATURE" \
  --concurrency "$CONCURRENCY" \
  $CASCADE_FLAG $EXTRA

echo "--- done ---"
wc -l "$OUT" || true
cat "${OUT%.jsonl}.meta.json" || true

# A smoke run GATES the full sweep: chained with `--dependency=afterok`, exiting
# non-zero here stops 420 reads from being spent on an endpoint that produces
# nothing usable. The two failures worth catching are both silent — replies that
# never parse, and yes-answers whose quotes never verify (a model paraphrasing
# instead of copying). Either makes every downstream number meaningless while
# the run still "succeeds".
if [ "${READER_SMOKE:-0}" = "1" ]; then
  echo "--- smoke gate ---"
  .venv/bin/python - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
verdicts = [v for r in rows for v in r["verdicts"]]
parsed = [v for v in verdicts if v["answer"] is not None]
yes = [v for v in verdicts if v["answer"]]
verified = [v for v in yes if v["quote_verified"]]
print(f"  reads          {len(rows)}")
print(f"  verdicts       {len(verdicts)}")
print(f"  parsed         {len(parsed)}  ({len(parsed)/max(len(verdicts),1):.0%})")
print(f"  said yes       {len(yes)}")
print(f"  quote verified {len(verified)}  of {len(yes)} yes-answers")
fail = []
if not verdicts:
    fail.append("no verdicts at all")
if len(parsed) < 0.5 * len(verdicts):
    fail.append("more than half the replies did not parse")
if yes and not verified:
    fail.append("every yes had an unverifiable quote — the model is paraphrasing, not copying")
if fail:
    print("SMOKE FAILED: " + "; ".join(fail))
    sys.exit(1)
print("SMOKE OK")
PY
fi
