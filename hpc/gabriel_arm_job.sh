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
#   ARM=shuffled-8b-persist  as shuffled-8b, plus the widening ladder (D-076)
#
# THE 8B JUDGE IS NOW THE STANDARD for new experiments. It is not measurably
# worse -- -0.0035 paired on the supervisor's gold, well inside the noise -- and
# it is roughly 3x faster under sustained load. The powered version of that
# comparison arrives as a by-product of the 72B pair still running, and belongs
# in the write-up even though the decision is already taken.
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
# THE SUBMITTED MODEL MUST WIN OVER .env (D-082). `set -a; . ./.env` re-exports
# LLM_MODEL_NAME, which is pinned to the 72B, so `sbatch --export=...` was
# silently discarded and the client asked a QwQ server for 72B weights. The
# serve scripts were fixed; these eval jobs were not, and they are the half
# that picks the model NAME sent in the request.
_OVERRIDE_MODEL="${LLM_MODEL_NAME:-}"
# SAME PROTECTION FOR THE JUDGE, and it is a separate variable for a reason.
# 2026-09-01: the guard above was applied to LLM_MODEL_NAME and never to
# CRITIC_MODEL, so the `*-8b` arms below hardcoded the Llama name over whatever
# was submitted. Port 8001 was serving Qwen3.5-9B, the client asked for
# NousResearch/Meta-Llama-3.1-8B-Instruct, and every judge call 404'd. Six QwQ
# arms (54044-54049) ran a full hour each with `--critic` set and NO CRITIC
# ALIVE -- 118 failed calls and 0 successful ones in 54046. The critic is the
# largest single effect measured on this project (+0.231), so those runs are
# not "slightly off", they are critic-off runs wearing a critic-on label.
# The .env override was fixed once (D-082) for one variable; this is the same
# bug in the same file, one variable over.
_OVERRIDE_CRITIC="${CRITIC_MODEL:-}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
# Restored after .env had its say.
if [ -n "$_OVERRIDE_MODEL" ]; then export LLM_MODEL_NAME="$_OVERRIDE_MODEL"; fi
echo "client will request model: ${LLM_MODEL_NAME:-<unset>}"


GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export LLM_BASE_URL="http://${GPU_HOST}:${VLLM_PORT:-8000}/v1"

QUESTIONS="${1:-eval/questions/gabriel-gold-2026-08-25.jsonl}"
ARM="${ARM:-off}"
REPEATS="${REPEATS:-3}"

# Held fixed across every arm.
FLAGS="--contract v3"
# Appended verbatim, so a new arm can be tried without touching the table
# above. Recorded in the run config either way, via effective_config.
EXTRA="${EXTRA_FLAGS:-}"

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
  shuffled-8b-persist)
    # The persistence arm (D-076). Identical to shuffled-8b except that a run
    # about to abstain while holding rows, with rounds to spare, is offered one
    # concrete untried route. Paired against shuffled-8b on the same code, which
    # is why the baseline has to be re-run rather than taken from 2026-08-28:
    # three fixes have landed since (D-073 empty-hop notes, D-074 the v3
    # abstention challenge, D-075 the duplicate guard).
    FLAGS="$FLAGS --critic --critic-seed 20260815 --persist"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  shuffled-8b)
    FLAGS="$FLAGS --critic --critic-seed 20260815"
    export CRITIC_BASE_URL="http://${GPU_HOST}:8001/v1"
    export CRITIC_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

# WAIT for the endpoints, do not merely test them.
#
# The first version failed immediately if the judge was not answering, which is
# right when a server is already up and wrong the moment anything is chained
# behind a server launch: a 72B takes minutes to load, so an arm submitted with
# a Slurm dependency arrives before the model is ready and kills itself. That
# turns an unattended overnight queue into a queue that quietly does nothing.
#
# Both ports matter now. Every arm answers on 8000, and with the 8B judge as the
# standard configuration every critic arm also judges on 8001.
#
# The probe MUST authenticate: vLLM is served with --api-key, an unauthorised
# /v1/models returns 401, and `curl -f` reads that as a dead server. The key
# goes over stdin via `curl -K -`, never as an argument, so it cannot appear in
# `ps` or in the job log.
wait_for() {
  local url="$1" name="$2" waited=0
  while [ "$waited" -lt "${ENDPOINT_WAIT:-1800}" ]; do
    if printf 'header = "Authorization: Bearer %s"\nurl = "%s/models"\n' \
         "$LLM_API_KEY" "$url" | curl -sf --max-time 10 -o /dev/null -K -; then
      echo "  $name ready at $url (waited ${waited}s)"
      return 0
    fi
    sleep 30; waited=$((waited + 30))
  done
  echo "FATAL: $name at $url never answered within ${ENDPOINT_WAIT:-1800}s."
  echo "       Refusing to start: without it this arm would silently run a"
  echo "       different configuration from the one it is labelled with."
  return 1
}

# The submitted judge wins over the arm table's default, for the same reason
# the submitted answerer does. Applied HERE, after the case block above has had
# its say, or the hardcoded Llama name would win again.
if [ -n "$_OVERRIDE_CRITIC" ]; then export CRITIC_MODEL="$_OVERRIDE_CRITIC"; fi

# A LIVE PORT IS NOT A LIVE MODEL. wait_for only proves something answers at
# the URL; it says nothing about the model NAME in the request body. On
# 2026-09-01 that gap cost six hour-long QwQ arms: 8001 was up and healthy
# serving Qwen3.5-9B, the probe passed, and then every single judge call 404'd
# because the client asked for a Llama that server had never heard of. The run
# completed, scored, and reported -- as a critic-on arm with no critic.
# So: assert the exact id we will send is in the served list, and refuse if not.
model_served() {
  local url="$1" want="$2" name="$3"
  local ids
  ids=$(printf 'header = "Authorization: Bearer %s"\nurl = "%s/models"\n' \
          "$LLM_API_KEY" "$url" | curl -sf --max-time 15 -K -) || return 1
  case "$ids" in
    *"\"id\":\"$want\""*|*"\"id\": \"$want\""*) echo "  $name serves '$want'"; return 0 ;;
  esac
  echo "FATAL: $name at $url does NOT serve '$want'."
  echo "       It serves: $(printf '%s' "$ids" | tr ',' '\n' | grep -o '"id":[^,}]*' | head -5)"
  echo "       Refusing to start: this arm would report critic-on results with"
  echo "       a critic that 404s on every call (2026-09-01, jobs 54044-54049)."
  return 1
}

wait_for "$LLM_BASE_URL" "answerer" || exit 3
model_served "$LLM_BASE_URL" "$LLM_MODEL_NAME" "answerer" || exit 4
case "$ARM" in
  *-8b|*-8b-persist)
    wait_for "$CRITIC_BASE_URL" "judge" || exit 3
    model_served "$CRITIC_BASE_URL" "$CRITIC_MODEL" "judge" || exit 4 ;;
esac

echo "host=$(hostname)  arm=$ARM  questions=$QUESTIONS  repeats=$REPEATS"
echo "flags='$FLAGS $EXTRA'  critic_endpoint='${CRITIC_BASE_URL:-<planner client, 72B>}'"

# SYSTEM and WORKERS are parameters now. The script hardcoded `--system
# planner`, so the free-SQL control could not be run against a local vLLM at
# all -- and the runner was serial until D-094. vLLM BATCHES concurrent
# requests, so workers help more here than against a hosted endpoint: the GPU
# is idle between one request finishing and the next arriving.
.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system "${SYSTEM:-planner}" --repeats "$REPEATS" \
    --workers "${WORKERS:-16}" $FLAGS $EXTRA
