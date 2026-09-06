#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=critic
# Requeueable ON PURPOSE: the ECC check below puts the job back in the
# queue when Slurm draws this node's faulty card, and a job that is not
# marked requeueable cannot do that.
#SBATCH --requeue
#SBATCH --output=/home/xucabrjs/hepcoveragekg_setup/logs/critic_%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# =============================================================================
# THE CRITIC ALONE, ON ONE CARD, ON PORT 8001.
#
# WHY A SEPARATE SCRIPT. serve_pair.sh, serve_split.sh and serve_both.sh all
# begin with `pkill -u $(id -u) -f "vllm serve"` to clear their own stale
# servers. That pattern matches EVERY vllm process this user owns, including a
# healthy planner someone else's job is serving right now. Submitting any of
# them to add a judge alongside a live 24h run would kill the run it was meant
# to complement. This script therefore has NO pkill, deliberately, and must
# never grow one: it exists to be started next to something already working.
#
# THE MODEL IS A REQUIRED ARGUMENT, NOT A DEFAULT. serve_pair.sh reads
#   BIG="${_OVERRIDE_BIG:-${LLM_MODEL_NAME:-Qwen/QwQ-32B-AWQ}}"
# whose final fallback is unreachable, because .env always sets LLM_MODEL_NAME.
# A job submitted without --export therefore serves whatever .env pins while
# appearing to default to something else. That cost job 54028: it asked for QwQ,
# loaded a 72B, and died on a memory figure sized for the model it did not load.
# Here there is no default to be wrong about -- no argument, no job.
#
#   sbatch hpc/serve_critic.sh Qwen/Qwen3.5-9B
#
# WHY ONE CARD AND NOT TWO. serve_pair.sh asks for two purely so it can refuse
# the faulty CA:00.0 and use the spare. That trick only pays when two cards are
# free; when one is, it queues instead of running. The judge is the thing we
# want up NOW, beside a planner that already holds the other cards, so it takes
# the single free card and lets the ECC gate refuse rather than gamble.
# =============================================================================
set -euo pipefail

MODEL="${1:?usage: serve_critic.sh <hf-repo-id>   (no default -- see header)}"

cd "${REPO:-$HOME/HEPCoverageKG}"
# NEVER cd to $HOME and never run python from it: a stray ~/inspect.py shadows
# the standard library there and breaks `import torch` inside the container.
if [ -f .env ]; then
    # Only the key is wanted from .env. Sourcing the whole file is what let a
    # pinned LLM_MODEL_NAME override an explicit request, so read the one line.
    LLM_API_KEY="$(grep -E '^LLM_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi
: "${LLM_API_KEY:?not set -- add it to .env on the cluster}"
export HF_HUB_OFFLINE=1

IMG="${VLLM_IMAGE:-$HOME/hepcoveragekg_setup/images/vllm-openai-v0.18.0}"
PORT="${CRITIC_PORT:-8001}"
# 0.90 OOMs ON THIS MODEL, and not for the reason it looks like. vLLM 0.18.0
# sizes the KV cache from a profiling run that does NOT account for the memory
# CUDA graph capture will need afterwards. At 0.90 it handed 51.53 GiB / 421,872
# tokens to KV -- vastly more than a judge will ever use -- and then died trying
# to allocate 6.44 GiB of graphs. 0.19 fixes this by default; here the estimator
# is switched on below AND the budget is cut, because a critic reading one window
# and one candidate answer does not need a third of a million tokens of cache.
UTIL="${CRITIC_UTIL:-0.45}"
# 32k is far more than a judge needs -- it reads one paper window and a candidate
# answer. Qwen's own guidance is to keep 128k+ "to preserve thinking", which is
# advice for long-context reasoning tasks, not for grading a short claim. Raise
# it here if a judgement ever needs the whole paper.
LEN="${CRITIC_LEN:-8192}"
#
# !! THE CALLER MUST SEND  chat_template_kwargs={"enable_thinking": false}  !!
#
# Qwen3.5 is a reasoning model and there is no server-side switch for this in
# 0.18.0 -- it is a per-request field, so it lives in the critic client, not
# here. Measured on this exact endpoint, same question, temperature 0:
#
#   thinking left on   800 tokens, finish_reason=length, content EMPTY
#   thinking turned off 19 tokens, finish_reason=stop,   correct verdict
#
# The first row is the D-059 failure wearing a new hat: the judge burned its
# whole budget thinking, returned an empty string, and reported no error. A
# caller that trusts `content` records that as "no objection" and the run looks
# healthy. Do not serve this model to the critic path without the flag.
# Qwen3.5 emits its chain of thought before the answer. Left unparsed those
# tokens land in the reply body, which is exactly how 99 of 366 QwQ replies were
# discarded (D-059). vLLM splits them into reasoning_content when told the
# format; set CRITIC_REASONING_PARSER='' for a model that does not think.
PARSER="${CRITIC_REASONING_PARSER-qwen3}"

echo "node:  $(hostname)"
echo "model: ${MODEL}"
echo "image: ${IMG}"
echo "port:  ${PORT}   util: ${UTIL}   max-model-len: ${LEN}   reasoning-parser: ${PARSER:-<none>}"

if nvidia-smi -L 2>/dev/null | grep -q "MIG "; then
    echo "ERROR: $(hostname) is MIG-partitioned -- slices are too small."; exit 76
fi

# THE ECC GATE. compute-gpu-0-1 has one A100 (bus CA:00.0, index 2) carrying
# 1413 uncorrected ECC errors: it loads a model happily and dies on the first
# inference (D-040, D-057). With a single card there is no spare to fall back
# to, so refuse in a second rather than discover it an hour into a run.
gpu="${CUDA_VISIBLE_DEVICES%%,*}"
[ -z "$gpu" ] && { echo "ERROR: no CUDA_VISIBLE_DEVICES"; exit 1; }
errs=$(nvidia-smi -i "$gpu" --query-gpu=ecc.errors.uncorrected.aggregate.total \
        --format=csv,noheader,nounits 2>/dev/null || echo unknown)
bus=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || echo "?")
echo "  gpu $gpu ($bus): uncorrected ECC = $errs"
[ "$errs" = "0" ] || {
    # REQUEUE, DO NOT DIE. compute-gpu-0-1 has three A100s and one of them
    # (bus CA:00.0) carries 1413 uncorrected ECC errors. Which card Slurm hands
    # out is luck, so refusing outright throws away the QUEUE WAIT as well as
    # the job: on 2026-09-06 both servers waited 8h29m for a GPU, drew the bad
    # card, and died one second after starting. The arms behind them never ran.
    #
    # A requeue puts the job back in the queue and Slurm may place it on a
    # different card. Bounded by SLURM_RESTART_COUNT so a node whose cards are
    # ALL bad stops rather than spinning -- an infinite requeue would hide the
    # hardware fault, which is the thing this check exists to surface.
    echo "ERROR: card is not ECC-clean ($errs uncorrected)."
    tries="${SLURM_RESTART_COUNT:-0}"
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$tries" -lt "${MAX_ECC_REQUEUE:-4}" ]; then
        echo "       requeueing (attempt $((tries + 1)) of ${MAX_ECC_REQUEUE:-4})"
        echo "       -- another card on this node may be clean."
        scontrol requeue "$SLURM_JOB_ID"
        sleep 30          # let the requeue land before the shell exits
        exit 0
    fi
    echo "       giving up after $tries requeues: every card drawn was faulty."
    echo "       Report the node -- this is hardware, not configuration."
    exit 75
}

# NO pkill HERE. See the header. Anything already serving stays serving.
ARGS=(
  --host 0.0.0.0 --port "$PORT"
  --api-key "$LLM_API_KEY"
  --served-model-name "$MODEL"
  --max-model-len "$LEN"
  --gpu-memory-utilization "$UTIL"
)
[ -n "$PARSER" ] && ARGS+=(--reasoning-parser "$PARSER")

# --env, not the ambient environment: apptainer's passthrough rules are version
# dependent and a silently-dropped variable here reads as "the fix did nothing".
exec apptainer exec --nv --pwd /tmp \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  "$IMG" vllm serve "$MODEL" "${ARGS[@]}"
