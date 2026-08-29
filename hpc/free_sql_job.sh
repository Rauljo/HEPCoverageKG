#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=freesql
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/freesql_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# The control: the same model, the same database, one tool that takes SQL.
#
# Nothing else measured in this project can falsify "typed tools over a typed
# graph beat a model turned loose on the data", because every other arm varies
# something INSIDE the typed layer. This varies the layer itself.
#
# It uses the SAME 72B planner on port 8000 as every planner arm, and no critic
# at all -- there is nothing for a critic to judge when there are no search
# candidates, only rows. So there is no 8B endpoint to wait for.
#
# WHAT MAKES IT A FAIR CONTROL, and not a straw man:
#   - it sees every table, including entity_canonical and same_as. The claim
#     under test is that the tools and vocabularies help, not that we hold data
#     nobody else has.
#   - six rounds, the same as the planner, so it can look, fail and recover.
#   - the same scorers on the same questions, with papers collected from
#     paper_id columns exactly as the planner collects them from entities.
#
# The one asymmetry that cannot be removed: it has no retrieval index, because
# that is not in the database. On concept questions the planner has a real
# advantage the control cannot match, and that is part of what is being
# measured rather than a flaw to apologise for.
#
# A PREDICTION ALREADY IN TROUBLE. The design doc expected free SQL to lose
# badly on "how many analyses used Pythia?" -- three spellings, one generator.
# But every spelling CONTAINS "pythia", so a one-line LIKE returns 58, which is
# the same answer the whole typed layer produces. The claim has to be won on
# synonymy WITHOUT a shared substring: MET vs missing transverse momentum vs
# ETmiss. See vault/ideas/free-sql-control.md.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG
if [ -f .env ]; then set -a; . ./.env; set +a; fi

GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export LLM_BASE_URL="http://${GPU_HOST}:${VLLM_PORT:-8000}/v1"

QUESTIONS="${1:-eval/questions/dev-fast-mixed.jsonl}"
REPEATS="${REPEATS:-3}"

# Wait for the answerer, same as the planner arms. Authenticated, because vLLM
# is served with --api-key and an unauthorised /v1/models returns 401, which
# `curl -f` reads as a dead server. The key goes over stdin via `curl -K -` so
# it cannot appear in `ps` or in the job log.
waited=0
until printf 'header = "Authorization: Bearer %s"\nurl = "%s/models"\n' \
        "$LLM_API_KEY" "$LLM_BASE_URL" | curl -sf --max-time 10 -o /dev/null -K -; do
  [ "$waited" -ge "${ENDPOINT_WAIT:-1800}" ] && {
    echo "FATAL: $LLM_BASE_URL never answered within ${ENDPOINT_WAIT:-1800}s"; exit 3; }
  sleep 30; waited=$((waited + 30))
done
echo "answerer ready at $LLM_BASE_URL (waited ${waited}s)"

echo "host=$(hostname)  system=free-sql  questions=$QUESTIONS  repeats=$REPEATS"

.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system free-sql --repeats "$REPEATS"
