#!/usr/bin/env bash
#SBATCH --job-name=frozen
#SBATCH --output=/home/xucabrjs/HEPCoverageKG/logs/frozen_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# The clean batch (D-089), moved to the cluster so it survives a closed laptop.
# Answers come from OpenRouter, not a local vLLM -- compute nodes have outbound
# HTTPS (verified 2026-09-02, HTTP 200 to openrouter.ai from a compute node).
set -uo pipefail
module load Python/3.9.6-GCCcore-11.2.0
cd /home/xucabrjs/HEPCoverageKG

# The key comes from .env on disk, never an argument and never echoed.
set -a; . ./.env; set +a
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_API_KEY="$OPENROUTER_API_KEY"
export CRITIC_BASE_URL="https://openrouter.ai/api/v1"
export CRITIC_API_KEY="$OPENROUTER_API_KEY"
export LLM_MODEL_NAME="qwen/qwen3-32b"
export CRITIC_MODEL="meta-llama/llama-3.1-8b-instruct"
unset LLM_MAX_COMPLETION_TOKENS
export KIND_SEMANTICS=0
[ -n "${LLM_API_KEY:-}" ] || { echo "FATAL: no OPENROUTER_API_KEY in .env"; exit 2; }
echo "key present: yes (${#LLM_API_KEY} chars)"

CHATLAS="kipark/all-mpnet-base-v2-combined_4400-400vs1000"
Q=eval/questions/gabriel-gold-2026-08-25.jsonl
run () { local label="$1"; shift
  echo "##### $label #####"
  echo "      encoder=${ALIASES_EMBED_MODEL:-BAAI/bge-base-en-v1.5 (default)}  kind_semantics=$KIND_SEMANTICS"
  .venv/bin/python -m hepcoveragekg.cli eval run "$Q" "$@" \
      --repeats 3 --critic-seed 20260815 --timeout 1200 2>&1 \
    | grep -E "judged_f1|judged_precision|judged_recall|answered |errored|seconds" | head -6
  echo; }

echo "############ BATCH START $(date +%F\ %H:%M) on $(hostname) ############"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)"
unset ALIASES_EMBED_MODEL
run "A1  free-SQL  CONTROL (frozen baseline)"      --system free-sql
KIND_SEMANTICS=1 run "A2  free-SQL  + kind-semantics" --system free-sql
run "A3  free-SQL  + search-sets"                  --system free-sql --search-sets
run "A4  free-SQL  + concept-prompt"               --system free-sql --concept-prompt
run "A5  free-SQL  + index-quotes"                 --system free-sql --index-quotes
run "A6  free-SQL  + index-values"                 --system free-sql --index-values
export ALIASES_EMBED_MODEL="$CHATLAS"
run "A7  free-SQL  chATLAS, no search-sets"        --system free-sql
run "A8  free-SQL  chATLAS + search-sets  <-KEY"   --system free-sql --search-sets
unset ALIASES_EMBED_MODEL
run "B1  typed     CONTROL (frozen baseline)"      --system planner --critic
KIND_SEMANTICS=1 run "B2  typed     + kind-semantics" --system planner --critic
run "B3  typed     + subgoal-status"               --system planner --critic --subgoal-status
run "B4  typed     + reviewer"                     --system planner --critic --reviewer
run "B5  typed     + index-values"                 --system planner --critic --index-values
run "A1prime free-SQL CONTROL REPEAT (drift)"      --system free-sql
run "B1prime typed    CONTROL REPEAT (drift)"      --system planner --critic
echo "############ BATCH END $(date +%F\ %H:%M) ############"
echo "READ A1 vs A1prime AND B1 vs B1prime FIRST. If either differs by >0.06 the batch drifted."
