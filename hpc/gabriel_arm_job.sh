#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --exclude=compute-0-1
#SBATCH --job-name=garm
# 24h, not 12: eight questions finish in minutes, but the same script runs the
# 200-question sets, where a critic arm writes ~0.8 records a minute under
# six-way contention -- 600 records is thirteen hours before any tail.
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/garm_%j.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# =============================================================================
# One arm of the critic ablation, scored against the SUPERVISOR'S gold.
#
# FIVE ARMS, NOT EIGHT. The obvious design is 2x2x2 over {critic, ordering,
# judge}, but with the critic OFF there is no ordering and no judge -- those
# knobs do not exist on that path. Running them would produce three pairs of
# byte-identical results and cost 3/8 of the compute to learn nothing.
#
#   ARM=off             the control: no critic
#   ARM=ranked-72b      critic on, retrieval order,  judge = the answerer
#   ARM=ranked-8b       critic on, retrieval order,  judge = Llama-3.1-8B
#   ARM=shuffled-72b    critic on, shuffled,         judge = the answerer
#   ARM=shuffled-8b     critic on, shuffled,         judge = Llama-3.1-8B
#   ARM=shuffled-72b-forced  as shuffled-72b, but the critic's kept set is
#                       substituted wherever a raw set is passed to a tool
#
# EVERYTHING ELSE IS HELD AT ITS RECORDED DEFAULT. One axis at a time, or the
# result is uninterpretable -- D-062 lost a week to a one-line prompt change
# that moved the control while an arm was being read.
#
#   contract v3          ON.  Measured as an arm: +0.059 set F1, 27% faster.
#                        (Its two separately-measured mechanisms -- citing a
#                        count, and `refine` -- were both DROPPED. What is left
#                        is kept on principle and unattributed. The arm-level
#                        number is real; the attribution is not.)
#   force-critic-set     OFF. 91aeb1c: "off by default -- it takes a decision
#                        away from the model, and that has to be earned by the
#                        measurement it makes possible." It measured flat on
#                        quality and merely faster, so switching it on here
#                        would hold a knob at a value the record does not
#                        support, and would do it inside the arms meant to
#                        judge the critic.
#
# That leaves a known cost, stated rather than hidden: with force off, 39% of
# counts in a critic-on arm run over the RAW set and discard the critic's
# verdicts -- four counts in ten. It is the likeliest reason the critic's
# counting gain (+0.063) is so much smaller than its set-F1 gain (x2.3). So
# these arms measure the critic AS IT CURRENTLY SHIPS, not the critic at its
# best. If a critic arm wins, one follow-up arm with force on is the next
# question, and it is one arm, not a redesign.
#
# THE GOLD IS PARTIAL, BY CONSTRUCTION. Gabriel only ever saw papers our system
# surfaced, so `judged_set_f1` scores inside the judged universe -- a paper he
# never looked at is unknown, not wrong. Precision under that restriction is
# clean; recall is optimistic and cannot be otherwise. See D-072.
#
# EIGHT QUESTIONS IS THIN. The power analysis (D-063) wanted 200 for a 0.0008
# effect. Nothing here will resolve a small difference, and it is not meant to:
# it is the only measurement in the project against labels a physicist wrote.
# Run QUESTIONS=eval/questions/dev-2026-08-03-paperA-200.jsonl for the powered
# version, and read the two together.
#
# The judge endpoint is chosen by CRITIC_BASE_URL: unset means the critic talks
# to the planner's own client (the 72B), set points it at the small model. This
# is why serve_split.sh must be up for the -8b arms and why they will fail fast,
# not silently, if it is not.
# =============================================================================
set -euo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG

# The planner loads .env itself; the gate below needs the key too.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export LLM_BASE_URL="http://${GPU_HOST}:${VLLM_PORT:-8000}/v1"

QUESTIONS="${1:-eval/questions/gabriel-gold-2026-08-25.jsonl}"
ARM="${ARM:-off}"
REPEATS="${REPEATS:-3}"

# Held fixed across every arm.
FLAGS="--contract v3"

case "$ARM" in
  off)
    ;;
  ranked-72b)
    FLAGS="$FLAGS --critic" ;;
  shuffled-72b)
    FLAGS="$FLAGS --critic --critic-seed 20260815" ;;
  ranked-8b)
    FLAGS="$FLAGS --critic"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  shuffled-72b-forced)
    # The sixth arm, added 2026-08-28 to settle a contradiction rather than
    # argue it. Mechanism and measurement disagree about --force-critic-set:
    #
    #   mechanism  the kept set reaches the SAME gold papers as the raw set in
    #              96.7% of 396 stored search steps, losing 1.54% of gold-paper
    #              reachings -- while shrinking what gets counted 4-7x
    #              (68 -> 9 entities, 61 -> 17). Counting over the raw set means
    #              counting entities the critic just called unrelated.
    #   measurement the arm came out flat on quality, only faster.
    #
    # The likeliest reconciliation is dilution: only 39% of counts used the raw
    # set, so an effect on those was averaged over every question. Pairing this
    # against shuffled-72b isolates it on one axis.
    FLAGS="$FLAGS --critic --critic-seed 20260815 --force-critic-set" ;;
  shuffled-8b)
    FLAGS="$FLAGS --critic --critic-seed 20260815"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

# Fail before spending a GPU hour rather than after. A missing 8B endpoint on a
# -8b arm silently falls back to the 72B in `_critic_client`, which would put
# two different arms under one label -- the exact failure D-070 exists to stop.
#
# The probe MUST authenticate. vLLM is served with --api-key, so an unauthorised
# /v1/models returns 401, `curl -f` treats that as failure, and the gate would
# refuse to start against a perfectly healthy server. The key goes in over
# stdin via `curl -K -`, never as an argument, so it cannot appear in `ps` or
# in the job log.
case "$ARM" in
  *-8b)
    if ! printf 'header = "Authorization: Bearer %s"\nurl = "%s/models"\n' \
           "$LLM_API_KEY" "$CRITIC_BASE_URL" \
         | curl -sf --max-time 10 -o /dev/null -K -; then
      echo "FATAL: judge endpoint ${CRITIC_BASE_URL} is not answering."
      echo "       serve_split.sh must be running, or this arm would quietly"
      echo "       run the 72B and be recorded as the 8B."
      exit 3
    fi ;;
esac

echo "host=$(hostname)  arm=$ARM  questions=$QUESTIONS  repeats=$REPEATS"
echo "flags='$FLAGS'  critic_endpoint='${CRITIC_BASE_URL:-<planner client, 72B>}'"

.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats "$REPEATS" $FLAGS
