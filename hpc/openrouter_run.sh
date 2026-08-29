#!/bin/bash
# =============================================================================
# One arm against a hosted frontier model, through OpenRouter.
#
# Runs on the LOGIN NODE or any machine with internet -- it needs no GPU, since
# the planner is remote. The critic stays on the LOCAL 8B, which is both the
# point and the economy:
#
#   the point   -- only the planner changes, so a difference is attributable to
#                  the planner. Moving both at once would confound them.
#   the economy -- the critic is 48% of all calls. Sending it to a metered
#                  endpoint would roughly double the bill for a component we
#                  have already measured to death.
#
# COST, from the measured 18,899 prompt tokens per record. Gabriel's 8 questions
# x 3 repeats is ~466k input, ~12k output:
#
#     openai/gpt-5.6-luna           ~$0.10   (the default; $0.20/$1.20 per M)
#     openai/gpt-4o                 ~$1.28
#     anthropic/claude-sonnet-4     ~$1.57
#     anthropic/claude-opus-4       ~$7.86   <- a third of the budget, once
#
# Those completion figures come from Qwen, which does not emit reasoning
# tokens. A reasoning model bills its thinking as output, so treat them as a
# FLOOR: gpt-5.6-luna at 4k reasoning tokens per record is ~$0.21, not $0.10.
#
# The fast set is 8.6x bigger. On luna that is still under a dollar; on a
# frontier model it is $10-13, so do not start there.
#
# LLM_BUDGET_USD is a HARD STOP, checked before every call. Set it. A malformed
# loop retrying a 19k-token prompt is the whole account, unattended.
#
# THE MODEL MUST SUPPORT TOOL CALLING. The planner is nothing but tool calls; a
# model without them produces an empty run and a real bill.
# =============================================================================
set -euo pipefail
cd "${REPO:-$HOME/HEPCoverageKG}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Capture the LOCAL vLLM key BEFORE LLM_API_KEY is overwritten with OpenRouter's.
# The critic talks to the local server and needs this one; inheriting the
# OpenRouter key gives it 401 on every chunk, and `judge_candidates` then does
# what it is built to do with a failure -- default every candidate to KEPT. The
# run continues, labelled critic-on, having run critic-off.
LOCAL_LLM_KEY="${LLM_API_KEY:-}"

: "${OPENROUTER_API_KEY:?not set -- put it in .env, never on the command line}"
MODEL="${OPENROUTER_MODEL:-openai/gpt-5.6-luna}"
QUESTIONS="${1:-eval/questions/gabriel-gold-2026-08-25.jsonl}"
REPEATS="${REPEATS:-3}"

export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL_NAME="$MODEL"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export LLM_BUDGET_USD="${LLM_BUDGET_USD:-2.00}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"

# A REASONING model spends output tokens thinking before it answers, and those
# come out of the same allowance. The planner's default is 800, tuned on Qwen,
# which does not think: hand that to a reasoning model and it can spend the lot
# reasoning and return an empty answer -- a full bill for no data. Raised here,
# and the run records the value so a short-completion arm is distinguishable
# from a broken one.
case "$MODEL" in
  *gpt-5.6-*|*o1*|*o3*|*reasoning*)
    export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-8000}"
    echo "reasoning model detected -- completion allowance raised to $LLM_MAX_COMPLETION_TOKENS" ;;
esac

# The critic stays local and free. Unset it to put the critic on the remote
# model too -- and expect the bill to roughly double.
GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export CRITIC_BASE_URL="${CRITIC_BASE_URL:-http://${GPU_HOST}:8001/v1}"
export CRITIC_MODEL="${CRITIC_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
# The critic talks to the LOCAL vLLM and needs the LOCAL key. Without this it
# inherits LLM_API_KEY -- now the OpenRouter key -- gets 401 on every chunk, and
# defaults every candidate to kept: a run labelled critic-on that ran critic-off.
export CRITIC_API_KEY="${CRITIC_API_KEY:-$LOCAL_LLM_KEY}"
: "${CRITIC_API_KEY:?no local vLLM key found in .env -- the critic would 401 and silently no-op}"

echo "planner : $MODEL  (via OpenRouter)"
echo "critic  : $CRITIC_MODEL  (local, free)"
echo "budget  : \$$LLM_BUDGET_USD hard stop"
echo "questions: $QUESTIONS x $REPEATS"
echo

# SYSTEM=planner (default) or SYSTEM=free-sql, the control. free-sql takes no
# critic flags -- there are no search candidates to judge, only rows -- so the
# planner-only arguments are omitted rather than passed and ignored.
case "${SYSTEM:-planner}" in
  free-sql)
    echo "system  : free-sql (SQL + search, no critic)"
    .venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
        --system free-sql --repeats "$REPEATS" ;;
  planner)
    echo "system  : planner (typed tools, contract v3, critic on)"
    .venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
        --system planner --repeats "$REPEATS" \
        --contract v3 --critic --critic-seed 20260815 ;;
  *) echo "unknown SYSTEM=${SYSTEM}"; exit 2 ;;
esac
