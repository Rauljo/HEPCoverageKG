# Plan-ahead batching: chain dependent calls in one round

**Status**: proposed (2026-09-17, build freeze -- not built). Raul's observation while
writing the typed-agent section: a model may not want the immediate result but the one
three steps on, so it should be allowed to write the chain in one turn.

## What is true today

- The executor runs a round's calls **sequentially** and the set table is **live**
  (`graph.execute`, `for call in calls`; `known_entity_ids` and `sets` update after each
  call). A later call in the same round can therefore consume a set an earlier call saved.
- The prompt says the opposite: "a tool that depends on another tool's output cannot be
  called in the same turn as it" (`UNKNOWN_ID_MESSAGE`).
- Measured on the DIAS typed runs (3,010 rounds with a search plus other calls): same-round
  references to a `set_N` **succeeded 927 times, failed 82** -- every failure a wrong
  name (`set_2` referenced when only `set_1` exists). The model does it anyway, by guessing.
- Set names are deterministic already: `set_N` = the N-th search of the run.

## Why it should be allowed

- Round-trips to the model are the whole cost (0.29 ms per template vs seconds per call).
  Rounds 1-3 are a near-deterministic recipe (search -> hop -> resolve), so one turn saves
  two calls per question -- most of the typed/free-SQL cost gap (7.0 vs 2.8 calls).
- Free-SQL is the existence proof: its cost advantage is writing the 2-3-hop path as one
  JOIN (36% of records one statement, 68% of statements carry a JOIN). This is the typed
  equivalent.
- It is ReWOO / plan-then-execute; see the planning literature note kept in the
  methodology stub.

## The cheap version (no mechanism)

One prompt line documenting the contract: "search saves as `set_N` in order of searches
this run; you may reference the set a search in this same turn will create." Turns the
82 name guesses into a rule and removes the prompt/executor contradiction. Zero build.

## What it costs, and the rule

Blind chaining cannot react: a pre-written `subjects_of(set_1)` runs even if the search
returned 0 rows or 400 heterogeneous entities (the reason `_kept` sets exist). So the
rule is "chain when you know what you will get; observe when you do not" -- which is what
the model already does by choice. Errors already stop the chain and are shown next round.

## What to measure before claiming it

How often the dependent call in round k+1 is what the model would have written blind in
round k. If "usually", this is a cost mechanism to report beside anchoring (D-210).
Score: LLM calls and seconds at equal judged F1, paired.
