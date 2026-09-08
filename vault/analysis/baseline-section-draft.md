# Proposal for "Results and Discussion -- Baseline" (experiments.tex §Baseline)

*Draft material, 2026-09-08. Numbers from analysis/controls-and-gabriel.md and
D-133, D-141..D-150. Written as prose you can adapt; the register is a results
chapter that diagnoses rather than ranks. Every number has a run id in the
analysis document.*

## Opening paragraph (what the baseline is for)

The baseline runs answer one question: before any mechanism is added, where
does each of the two ways of querying the graph lose? Two systems answer the
same questions over the same graph with the same model: the typed planner
(tool calls over the typed schema: search, facets, subjects_of, papers_of,
contents_of) without its relevance critic, and the free-SQL agent (the same
search tool, then SQL over the same schema). They are read side by side not
to declare a winner but because each fails in a way the other does not, and
the failures locate the losses: in retrieval, in the handoff from retrieval
to the written answer, in the metric, or in the graph itself. Configuration:
QwQ-32B-AWQ on the cluster for the 164-question set (one repeat per run, two
independent runs per arm for the noise floor); qwen3-32b on OpenRouter for the
supervisor's 9 questions (3 repeats). Critic on / off is reported where it
changes anything, which is almost nowhere (below).

## §1 Full question run (164)

### The first finding is about the answers, not the retrieval

State it first because it conditions every table: the typed agent frequently
does not write paper identifiers. On the 84 retrieval-type questions it names
no paper on 31; the answers are counts ("48 analyses define the 'Missing
transverse momentum' object"), examples ("such as 2001.06899, 2004.01678 ...")
or summaries, written after a chain that had already retrieved the right
papers (reach 0.98). This is why every set score is reported three ways
(headline; named-only with precision and recall; essay-only with the
fallback records scored 0), and why retrieval reach is reported beside the
answer score: the two mechanisms have to be judged separately. The single
largest gain of the whole project came from asking the model to put the ids
in the text (D-117), and the baseline is where that need is visible.

### Retrieval questions (84, one predicate over one entity, exact truth, median 5 papers)

Table: typed / free-SQL, two runs each: set_f1 headline, named-only f1 (p, r,
n), fallback rate, essay-only, reach.
  typed 0.196 / 0.205; named-only 0.19-0.22 (p 0.17-0.19, r 0.31-0.33, n 53-57); fallback 31 / 27 of 84; essay 0.12 / 0.15; reach 0.98
  free-SQL 0.284 / 0.252; named-only 0.33 / 0.28 (p 0.25 / 0.23, r 0.59 / 0.48, n 70-75); fallback 14 / 9; essay 0.27 / 0.25

Diagnosis, two sentences each:
- Typed: retrieval is complete; the loss is the handoff (counts, examples,
  summaries). Where it names, it names 12.8 papers for a truth of 5.4 with
  precision 0.17: the ids it writes are drawn from the whole footprint.
- Free-SQL: writes the list (a SQL result IS a column of arXiv ids) but cannot
  narrow the id set the search returns: `entity_id IN (every hit)` gives 14
  papers for a 3-paper "WZ background" truth (recall 1, precision 0.21); when
  the hits are the wrong family (Powheg Box v2: 34 named, 2 of 12 correct) it
  joins them all. It has no notion of which hit the question meant; canonical
  expansion and kind filters are exactly what the typed layer adds.
- Why free-SQL wins these: the questions were generated from the schema, one
  predicate over one concept, so a single well-formed statement returns the
  truth. This is a property of the question set and must be said as such
  (points forward to the supervisor's questions, where it reverses).

### Concept-set questions (14)
One paragraph: same two failures, n too small to add anything; report the
table and move on.

### Count questions (20)
count_correct typed 0.30 / 0.30, free-SQL 0.25 / 0.45; closeness (1 - |claimed-true|/true) typed 0.35, free-SQL 0.52-0.56; critic-on typed 0.15.
- Free-SQL counts in the database (COUNT(DISTINCT paper_id)) so it is close
  and usually over (LIKE over-matches: 10 over, 5 under, one answer of 1608).
- Typed counts what is on its page (25 rows) or hedges; in this run the
  count tool returned an empty preview and the model wrote "**X papers**" --
  a tool defect, to be fixed and re-run before the count numbers are quoted
  (flag it honestly as such).
- Neither counts over the canonical concept after alias resolution, which is
  the exact count.

### Per-paper questions (36, "what does analysis X use", truth = its entities)
- Typed retrieval is complete: 0.97 of the truth entities returned, by id.
- The "mentioned" metric as first computed measured copying (strict
  containment of the stored label): typed 0.30, free-SQL 0.50, with a correct
  paraphrase ("Drell-Yan, nonprompt lepton, triboson, ZZ ...") scoring 0.
  Re-scored with a token-overlap match: typed 0.77-0.81, free-SQL 0.75-0.81.
  Report the fuzzy numbers and say why the strict ones were wrong (one
  sentence; it is a methods point, not an embarrassment).
- Two things survive: the typed agent paraphrases, free-SQL copies rows; and
  the arms that ask for arXiv ids in the text say fewer labels (0.67) -- an
  answer that becomes an id list stops describing.

### The critic (one paragraph, here or in its own subsection)
On the 164: set_f1 0.234 vs 0.215 with a repeat floor of 0.030; counts
worse with it (0.15 vs 0.30). On the 9: the sign depends on the lane and both
gaps are about one standard error (cluster no-critic 0.44 vs critic 0.36,
n=9; OpenRouter 0.38 vs 0.45, n=27). Its only visible mechanism is
shortening the page, which helps the handoff on one model and hurts it on
another. Conclusion: not a relevance judgement that moves anything; report
and retire.

## §2 Expert ground-truth questions (Gabriel's 9)

### The table first
Per question: gold / judged sizes; for each control (typed critic, typed
no-critic, free-SQL; OpenRouter, 3 repeats): judged_f1, named, gold named,
judged-wrong, outside the judged set; replay columns reach / reached-not-named
/ missed; failure class. (This is section 2 of the analysis document; the
figure gabriel_question_issues.png can carry the classes.)

### The classes, with the question that shows each best
- A  surface form / enumeration: gf-02 (the model searches "ABCD" and never
  the "matrix method" and "sideband" clauses; the 5 missed papers carry exactly
  those labels), gf-05 (kind filter: Higgs held as process/channel, searched
  as detector object; 13 of 14 missed papers have it as an entity), gf-08
  (literal phrase "exactly two electrons" -> free-SQL's false "not in graph").
- B  truncation: gf-08 (24 gold, 25 rows shown).
- C  summarise / point / no set operation: gf-01-condition (reach 11/11, critic
  run names 1), gf-01-met ("23 analyses ... examples include"), gf-07 (two sets
  retrieved, no tool to intersect them, chain-of-thought scored as the
  answer), gf-08 (24-paper list not written).
- D  quote-only: gf-03 (1 of 5), gf-04 (3 of 18).
- E  tag is not selection: gf-01 -- and the measured point (D-149): asking the
  stricter question raises precision 0.39 -> 0.75 and halves recall, F1
  unchanged, because the requirement edge exists for half the gold papers.
- Graph ceiling: gf-04 (3 papers where nothing mentions unfolding).

### What the nine say about the graph (D-150) -- the paragraph that matters
Two layers of unequal quality. Mentions: complete (reach 0.97-0.98 generated,
0.7-0.9 supervisor). Relations: uneven -- results 27 edges each, regions 5,
methods/systematics/observables ~1; MET named in 16 region labels, a
requirement edge in 1. Every hard supervisor question asks about the relation
layer; every generated question asks about the mention layer plus one dense
predicate. That is why free-SQL wins the generated set and ties the typed
stack on the supervisor's questions, and why the agents' "both concepts occur
on the paper" habit passes there and fails here. The bottleneck on the
supervisor's questions is extraction of usage relations, not querying.

### Free-SQL on the 9 (say it plainly)
0.571 on the current gold, precision 0.84 on the judged set, recall 0.51;
ties the full typed stack (0.570) by a different route (analysis 1.8). Where
it loses: gf-05, gf-07, gf-08 (0.18 / 0.33 / 0.12) -- the questions whose
concept is spread across labels or whose relation has no edge. Where it
"wins" on gf-01-condition / gf-01-met it does so by listing 21-44 papers of
which 14-15 nobody judged; judged_f1 cannot see them (caveat: metric).

### Metric caveats to state in this section (short list)
1. judged_f1 ignores names outside the judged set; report that count.
2. set_f1's footprint fallback; report named-only and the fallback rate.
3. gf-01-met (3 yes of 13 judged) and gf-07 (10 of 53) are thin or wide.
4. Per-paper "mentioned" needed a fuzzy match.
5. Three repeats minimum; gf-08 alone has a 0.14 spread over six repeats.

## Corrections to the current notes in experiments.tex
- "01-btagged: precision in reality would be pretty bad": we cannot know --
  15 of free-SQL's names were never judged. Say "unpenalised", not "bad".
- "08: freesql abstains": that was the stale-gold run (28311). On the current
  gold (85223) free-SQL names 3 papers on gf-08 (1.7 gold), does not abstain;
  the literal-phrase search is still the mechanism.
- "01-met: didn't understand much": the point is the metric (3 yes of 13
  judged; everything named lands outside) plus summarising; retrieval is 3/3.
- Count "X papers" placeholder: the count tool returned an empty preview in
  that run -- verify the tool before quoting count numbers (open item).
- Configuration line: add that the 164 runs are single-repeat with two
  independent jobs per arm, and that the 9 are 3 repeats; add the noise
  floors (0.030 set_f1; ~0.30 per record judged_f1).
