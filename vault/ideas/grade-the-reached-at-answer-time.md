# Grade the reached-but-ungraded papers at answer time

*Status: discussed (2026-09-08, overnight session). Not built.*

## The loss it targets

D-135's decomposition on the full stack (27 records, 318 gold slots): 37% named,
27% graded by the ranker but not named, 14% reached by some tool but never
graded, 22% never reached. The 14% is gf-07 almost entirely: 16 gold papers
that entered the run through `subjects_of` / `describe` rows and never went
through a `papers_of` or `facets` call, so the ranker never saw them and the
ranked answer (D-128, D-135, D-136) could not put them in front of the model.

## The mechanism

At the ranked-answer hook (both exits), before building the candidate list:
take every paper reachable from `session.known_entity_ids` (what the record's
`papers` field already computes at serialisation), subtract the papers with a
grade, and grade the remainder once with the ranker's judge. Then the ask sees
all of them. Nothing is removed; an ungraded paper is still a candidate at
the back (D-105's asymmetry).

## Why it was not built on 2026-09-08

- The runtime carries `conn` and the critic chat but not the ranker closure;
  the hook would have to build a judge client itself (small).
- On gf-07 the known set is 100-150 entities after enumeration; their papers
  number in the hundreds, so this is 7-20 judge calls at answer time per
  record, sequential -- 60-200 s added on a path that already brushes the
  1200 s record budget (enum3 lost a gf-02 record to it).
- Expected gain: half the bucket named at most, ~7% of gold slots, ~+0.03
  judged -- comparable to other steps of the night but the last one to be
  taken because it is the only one that adds a judge pass at answer time.

## If built

Cap the ungraded pool (e.g. 60, best by BM25 score of the row that reached
them), grade in the ranker's chunks, count `graded_at_answer` in the record,
and measure on gf-07 alone with 6 repeats first (D-138's lesson: single
questions need their own spread).
