#!/usr/bin/env bash
# =============================================================================
# THE CLEAN BATCH. Every arm measured against a control that ran on the SAME
# CODE, in the SAME script, with the SAME seed.
#
# WHY THIS EXISTS. On 2026-09-01 roughly forty arms were run across fourteen
# scripts between 11:00 and 23:57, while the shared code changed four times
# (schema card 17:39, retrieve+free_sql 18:34, templates 18:47, planner 18:48).
# The free-SQL baseline was measured four times and read 0.592, 0.526, 0.473,
# 0.465 -- a 0.127 spread in the REFERENCE LINE, larger than almost every
# effect being chased. Most of that night is uninterpretable. See D-086/D-089.
#
# THE FIVE RULES, each earned by a specific failure:
#
#   1. ONE SCRIPT, NOTHING ELSE RUNNING. Every arm here shares one code state.
#      Do not start anything else while this runs, and do not edit any file
#      under hepcoveragekg/ -- each arm is a fresh Python process that
#      re-imports, so an edit mid-batch changes later arms only.
#   2. THE CONTROL RUNS FIRST AND LAST. If the two disagree by more than the
#      noise floor, the batch is not trustworthy and no arm in it should be
#      believed. Drift is measured, not assumed away.
#   3. ONE SEED EVERYWHERE. `--critic-seed 20260815` on every arm. Mixing
#      seeded and unseeded runs is what made sets_v2 incomparable.
#   4. KIND_SEMANTICS DEFAULTS OFF. It went into the frozen path unflagged on
#      2026-09-01 and silently moved every later run. Here it is an arm.
#   5. THE ENCODER IS RECORDED. ALIASES_EMBED_MODEL is now in _ENV_CONFIG, so
#      a run file can say which encoder produced it. It could not before, which
#      is why the encoder arms could not be reconstructed afterwards.
#
# THE HEADLINE EXPERIMENT, and it is the reason the encoder arms are here at
# all. `--search-sets` sets SET_CAP=400: retrieval goes 400 deep and the whole
# match set is materialised as a temp table the model filters in SQL. WITHOUT
# it, `limit=20` and the model pastes ~3 ids. So encoder quality below rank ~3
# is INVISIBLE without search-sets and DECISIVE with it -- and every encoder
# arm run on 2026-09-01 was plain `--system free-sql`, i.e. the one setting
# where a better encoder cannot help. Arms 1/3/8/9 are the 2x2 that fixes this.
#
#   Prediction, recorded before running so it cannot be adjusted afterwards:
#   chATLAS should beat bge-base by MORE with search-sets than without. The
#   mechanism is precision. search-sets v1 cost precision 0.704 -> 0.602
#   because the model joined the ~400-row table with no WHERE and "could not
#   see that the set was heterogeneous". A 400-deep set built by bge-base
#   contains `lepton candidate`, `proton candidate`, `Photon candidate` at
#   ranks 5-9 on the Higgs query; built by chATLAS it contains NONE of those.
#   Same query, measured 2026-09-02. If the encoder matters anywhere, it is
#   here, and it should show up as less precision damage rather than as recall.
# =============================================================================
set -uo pipefail
cd "/Users/raulsal/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Dissertation/HEPCoverageKG"
. /private/tmp/claude-501/-Users-raulsal-Library-CloudStorage-OneDrive-UniversityCollegeLondon-Dissertation-HEPCoverageKG/0946ea08-c854-4158-a4c1-7dce0ce186f6/scratchpad/openrouter_env.sh

unset LLM_MAX_COMPLETION_TOKENS
export LLM_MODEL_NAME="qwen/qwen3-32b"
export CRITIC_MODEL="meta-llama/llama-3.1-8b-instruct"
export KIND_SEMANTICS=0          # rule 4: off unless an arm turns it on
CHATLAS="kipark/all-mpnet-base-v2-combined_4400-400vs1000"
Q=eval/questions/gabriel-gold-2026-08-25.jsonl

run () {
  local label="$1"; shift
  echo "##### $label #####"
  echo "      encoder=${ALIASES_EMBED_MODEL:-BAAI/bge-base-en-v1.5 (default)}  kind_semantics=$KIND_SEMANTICS"
  .venv/bin/python -m hepcoveragekg.cli eval run "$Q" "$@" \
      --repeats 3 --critic-seed 20260815 --timeout 1200 2>&1 \
    | grep -E "judged_f1|judged_precision|judged_recall|answered |errored|seconds" | head -6
  echo
}

echo "############ BATCH START $(date '+%Y-%m-%d %H:%M') ############"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo

# ---- free-SQL -------------------------------------------------------------
unset ALIASES_EMBED_MODEL
run "A1  free-SQL  CONTROL (frozen baseline)"        --system free-sql
KIND_SEMANTICS=1 \
run "A2  free-SQL  + kind-semantics"                 --system free-sql
run "A3  free-SQL  + search-sets"                    --system free-sql --search-sets
run "A4  free-SQL  + concept-prompt"                 --system free-sql --concept-prompt
run "A5  free-SQL  + index-quotes"                   --system free-sql --index-quotes
run "A6  free-SQL  + index-values"                   --system free-sql --index-values

# The 2x2. A1 (bge, no sets) and A3 (bge, sets) are already above.
export ALIASES_EMBED_MODEL="$CHATLAS"
run "A7  free-SQL  chATLAS, no search-sets"          --system free-sql
run "A8  free-SQL  chATLAS + search-sets  <-- KEY"   --system free-sql --search-sets
unset ALIASES_EMBED_MODEL

# ---- typed ----------------------------------------------------------------
run "B1  typed     CONTROL (frozen baseline)"        --system planner --critic
KIND_SEMANTICS=1 \
run "B2  typed     + kind-semantics"                 --system planner --critic
run "B3  typed     + subgoal-status"                 --system planner --critic --subgoal-status
run "B4  typed     + reviewer"                       --system planner --critic --reviewer
run "B5  typed     + index-values"                   --system planner --critic --index-values

# ---- rule 2: did the ground move under us? --------------------------------
run "A1' free-SQL  CONTROL REPEAT (drift check)"     --system free-sql
run "B1' typed     CONTROL REPEAT (drift check)"     --system planner --critic

echo "############ BATCH END $(date '+%Y-%m-%d %H:%M') ############"
echo "READ A1 vs A1' AND B1 vs B1' FIRST. If either pair differs by more than"
echo "~0.06, the batch drifted and the arms between them are not comparable."
