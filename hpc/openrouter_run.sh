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
#     openai/gpt-4o                 ~$1.28
#     anthropic/claude-sonnet-4     ~$1.57
#     google/gemini-2.5-pro         ~$0.70
#     deepseek/deepseek-chat        ~$0.14
#     anthropic/claude-opus-4       ~$7.86   <- a third of the budget, once
#
# The fast set is 8.6x bigger: $10-13 on a frontier model. Do not start there.
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

: "${OPENROUTER_API_KEY:?not set -- put it in .env, never on the command line}"
MODEL="${OPENROUTER_MODEL:-openai/gpt-4o}"
QUESTIONS="${1:-eval/questions/gabriel-gold-2026-08-25.jsonl}"
REPEATS="${REPEATS:-3}"

export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL_NAME="$MODEL"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export LLM_BUDGET_USD="${LLM_BUDGET_USD:-2.00}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-180}"

# The critic stays local and free. Unset it to put the critic on the remote
# model too -- and expect the bill to roughly double.
GPU_HOST="${GPU_HOST:-compute-gpu-0-1}"
export CRITIC_BASE_URL="${CRITIC_BASE_URL:-http://${GPU_HOST}:8001/v1}"
export CRITIC_MODEL="${CRITIC_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"

echo "planner : $MODEL  (via OpenRouter)"
echo "critic  : $CRITIC_MODEL  (local, free)"
echo "budget  : \$$LLM_BUDGET_USD hard stop"
echo "questions: $QUESTIONS x $REPEATS"
echo

.venv/bin/python -m hepcoveragekg.cli eval run "$QUESTIONS" \
    --system planner --repeats "$REPEATS" \
    --contract v3 --critic --critic-seed 20260815
