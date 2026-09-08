# The controls, by question type, and Gabriel's nine questions one by one

*Analysis document, 2026-09-08. Written from the run files named below; every
number can be regenerated with `eval/analysis/score_wave3.py`,
`eval/analysis/gabriel_replay.py` and the snippets in `logs/2026-09-08.md`.
This is the material for the first results section: what the two baseline
systems do, on which kinds of question, and exactly where each of Gabriel's
questions is lost.*

## 0. What is being compared, and how it is scored

**Two systems, one graph.** The *typed planner* answers with tools over the
typed graph (`search` over entity labels and aliases, `facets`, `subjects_of`,
`papers_of`, `contents_of`, `describe`), optionally with a *relevance critic*
(a second model that marks each retrieved entity relevant or not). The
*free-SQL* baseline is given the same graph as a SQLite schema and writes SQL
against it; it also has the entity `search` tool, and in practice it uses it
first to find entity ids and then writes the SQL (see the traces in section
2). Both are driven by the same answering model on a given lane.

**Two lanes.** The cluster lane (DIAS) runs QwQ-32B-AWQ as the answerer and
Qwen3.5-9B as the critic, on the 164-question file
(`eval/questions/discriminating-2026-09-03.jsonl`), one repeat per job; two
independent jobs per arm give the repeat-to-repeat spread. The OpenRouter lane
runs qwen3-32b as the answerer and llama-3.1-8b as the critic, on Gabriel's 9
questions with 3 repeats (27 records per arm).

**Question types in the 164** (from the file's own fields):

| type | n | shape | truth | truth size (median) | example |
|---|---|---|---|---|---|
| gen-retrieval | 84 | set | exact paper set from SQL | 5 | "Which analyses define a region requiring missing transverse momentum?" |
| gen-paper | 36 | set | the labels a paper's entities carry | -- | "Which collision systems does analysis 2405.18661 study?" |
| gen-concept (count) | 20 | count | an integer from SQL | 2 | "How many analyses used the generator OpenLoops?" |
| gen-conceptset | 14 | set | exact paper set | 2 | "Which analyses estimate the W+jets background?" |
| gabriel | 9 | set | Gabriel's yes-set, with the set he judged | 11 | see section 2 |
| type2 | 1 | set | exact | 8 | (one question; ignored below) |

**Scoring conventions that matter for reading the tables** (D-141, D-142,
D-145):

- `judged_f1` exists only where Gabriel judged: 9-10 records of the 164, 27
  per arm on the OpenRouter lane. It restricts to the papers he judged, so a
  paper named *outside* that set costs nothing (D-072). Over-listing is not
  penalised by this metric; it is by `set_f1`.
- `set_f1` is computed on the arXiv ids written in the answer text. When the
  text names none, it falls back to the retrieval footprint (~40 papers), and
  that fallback fires on a third to a half of the typed controls' set
  answers. Every set_f1 below is therefore given three ways: the headline, the
  *named-only* value (with its precision and recall and the number of records),
  and *essay-only* (fallback records scored 0). Named-only is the fair
  comparison between systems; the fallback rate is itself a finding.
- The per-paper questions are scored by label recall: `retrieved_label_recall`
  (did the tools return the truth labels) and `mentioned_label_recall` (did the
  answer say them); their gap is `truncation_loss`.
- Counts are scored by `count_correct` (exact) and `count_closeness`.

## 1. The controls on the 164, by question type

Cluster lane, QwQ. Two independent jobs per arm (a / b). Typed without critic
= 54257 / 54258; free-SQL = 54259 / 54260; typed with critic (wave 1) = 54251 /
54252.

### 1.1 Retrieval questions (84; "which analyses ...", exact truth, median 5 papers)

| arm | set_f1 | named-only f1 / p / r (n) | fallback | essay-only |
|---|---|---|---|---|
| typed, no critic a | 0.196 | 0.187 / 0.17 / 0.31 (53) | 31/84 | 0.118 |
| typed, no critic b | 0.205 | 0.215 / 0.19 / 0.33 (57) | 27/84 | 0.146 |
| free-SQL a | 0.284 | 0.326 / 0.25 / 0.59 (70) | 14/84 | 0.272 |
| free-SQL b | 0.252 | 0.277 / 0.23 / 0.48 (75) | 9/84 | 0.247 |

Free-SQL wins on every reading. Two separate facts make the gap. First, the
typed planner answers a third of these questions without writing a single
paper id (31 and 27 of 84), where free-SQL does so on 14 and 9. Second, when
both do name papers, free-SQL has both higher precision (0.25 vs 0.17) and
much higher recall (0.59 vs 0.31). These questions were generated from the
schema -- each is one predicate applied to one entity -- so a single
well-formed SQL query answers them exactly, and the typed planner's chain of
tool calls has more places to stop short.

### 1.2 Concept-set questions (14; exact truth, median 2 papers)

| arm | set_f1 | named-only f1 / p / r (n) | fallback | essay-only |
|---|---|---|---|---|
| typed, no critic a | 0.274 | 0.323 / 0.26 / 0.58 (11) | 3/14 | 0.253 |
| typed, no critic b | 0.335 | 0.579 / 0.52 / 0.77 (7) | 7/14 | 0.289 |
| free-SQL a | 0.399 | 0.399 / 0.38 / 0.54 (13) | 1/14 | 0.371 |
| free-SQL b | 0.344 | 0.370 / 0.35 / 0.46 (13) | 1/14 | 0.344 |

Same direction, small n; the typed b run's 0.579 named-only is on 7 records.

### 1.3 Count questions (20; "how many analyses ...")

| arm | count_correct | count_closeness |
|---|---|---|
| typed, no critic a / b | 0.30 / 0.30 | 0.35 / 0.37 |
| typed, critic a / b (w1) | 0.15 / 0.15 | 0.28 / 0.32 |
| free-SQL a / b | 0.25 / 0.45 | 0.52 / 0.56 |

Exact counts are rare for everyone (n = 20, so one question is 0.05). Free-SQL
is markedly *closer* when wrong (0.52-0.56 vs 0.35), which is what a COUNT
query buys; the typed planner counts rows it was shown, and it is shown at
most 25. The critic makes the typed system worse at counting (0.15 vs 0.30):
it removes rows before they are counted.

### 1.4 Per-paper questions (36; "what does analysis X use", truth = labels)

| arm | retrieved_label_recall | mentioned_label_recall | truncation_loss |
|---|---|---|---|
| typed, no critic a / b | 0.972 / 0.972 | 0.296 / 0.297 | 0.68 / 0.68 |
| typed, critic a / b (w1) | 0.972 / 0.972 | 0.368 / 0.350 | 0.60 / 0.62 |
| free-SQL a / b | 0.297 / 0.235 | 0.500 / 0.541 | -0.20 / -0.31 |

This is the cleanest picture of the two systems' characters. The typed
planner's `contents_of` returns 97% of the truth labels, and the answer then
mentions 30% of them: it finds and does not say. Free-SQL's queries retrieve a
third of the labels and the answer mentions half the truth -- it says more
than its own retrieval shows because the model fills in from the rows it saw
and from its own knowledge (negative "truncation loss" means the answer
mentioned labels the tools did not return). For the write-up: on questions
whose answer is a list of labels, the typed system's loss is entirely at the
handoff from retrieval to prose.

### 1.5 Gabriel's questions inside the 164 (9; one repeat per job)

| arm | judged_f1 (a / b) |
|---|---|
| typed, no critic | 0.473 / 0.403 |
| typed, critic (w1) | 0.354 / 0.361 |
| free-SQL | 0.232 / 0.375 |

Here the order reverses: typed beats free-SQL, and the no-critic typed run
beats the critic run. Single repeats on nine questions; the OpenRouter lane
(section 1.7) is where these arms are measured properly.

### 1.6 The critic

Wave 1 (critic on) against wave 2 (critic off), paired on the 164: set_f1
0.234 vs 0.215 with a repeat floor of 0.030; count 0.15 vs 0.30; Gabriel-10
judged 0.354/0.361 vs 0.426/0.363. Nothing the critic does on the 164 is
outside noise except making counts worse (D-133, corrected by D-141). Its
per-question effect on Gabriel's set is in section 2 (gf-01-condition).

### 1.7 Gabriel's 9 on the OpenRouter lane (qwen3-32b, 3 repeats, 27 records)

| arm | judged_f1 | reach | gold named / record | notes |
|---|---|---|---|---|
| typed, critic (runs 1914 + 63766, 6 repeats) | 0.452 (1914) | 0.69 | 3.5 | the control every mechanism was measured against |
| typed, no critic (run 84932) | *pending* | | | |
| free-SQL (run 85223) | *pending* | | | the earlier free-SQL run (28311, 2026-09-02) predates the current gold file -- gf-04 has no judged score in it -- and is not used for numbers |

## 2. Gabriel's nine questions, one by one

For each question: his yes-set ("gold") and the set he judged ("judged"); what
each control did (arXiv ids in the answer text; gold-named; FP = named,
judged, judged no; "outside" = named but not in his judged set, which
`judged_f1` ignores); the replayed tool trace against the DIAS database
(reach = gold papers any tool returned; r-not-n = reached but not named; miss
= in the graph, never reached; why-missed = whether an entity with the concept
exists on the missed paper, or only a quote, or nothing); and the failure
class in the D-118 vocabulary: **A** surface form / query formulation,
**B** truncation at max rows, **C** summarise or point instead of list,
**D** quote-only (extraction gap), **E** tag is not selection (false positive).

Runs: typed+critic OpenRouter = 1914 + 63766 (6 repeats); free-SQL OpenRouter
= 28311 (3 repeats, older gold -- traces are valid, judged numbers are not);
typed no-critic QwQ = 54257/54258; free-SQL QwQ = 54259/54260; typed critic QwQ
= 54251/54252.

### gf-01 -- searches whose selection uses both b-tagged jets and missing transverse momentum (gold 8, judged 33)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.53 | 17.3 | 5.7 | 7.7 | 4.0 |
| free-SQL OR (3) | 0.51 | 15.0 | 5.3 | 7.7 | 2.0 |
| typed no-critic QwQ (2) | 0.27 | 9.0 | 3.0 | 4.0 | 2.0 |
| typed critic QwQ (2) | 0.55 | 18.0 | 6.0 | 8.0 | 4.0 |
| free-SQL QwQ (2) | 0.00 | 1.5 | 0.0 | 0.0 | 1.5 |

Replay: reach 7/8 in every typed run; 1 gold paper never reached although an
entity carrying the concept exists on it (A). The loss is precision, not
recall: 8 false positives against 6 gold in the same answer. The typed planner
answers with `facets(objects=[BJet, MET], mode=all, category=search)`, which
returns every search paper that *mentions* both objects; free-SQL writes the
equivalent join on `entity_occurrence`. Both return co-occurrence, and the
question asks for joint *selection* -- **class E**, and it is the class no
mechanism this week fixed (the answer-critic filter struck gold, D-124). The
"searches rather than measurements" clause is honoured by both systems through
the paper category. One QwQ free-SQL repeat named nothing.

### gf-01-condition -- analyses that use b-tagged jets in their event selection (gold 11, judged 18)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.45 | 7.2 | 2.8 | 0.7 | 3.7 |
| free-SQL OR (3) | 0.44 | 21.3 | 4.7 | 1.3 | 15.3 |
| typed no-critic QwQ (2) | 0.76 | 18.0 | 8.0 | 2.0 | 8.0 |
| typed critic QwQ (2) | 0.17 | 3.0 | 1.0 | 0.0 | 2.0 |
| free-SQL QwQ (2) | 0.59 | 21.0 | 5.0 | 1.0 | 15.0 |

Replay: reach 11/11 in every typed run -- retrieval is complete. The typed
critic QwQ run then names 1 of the 11 (10 reached-not-named); the no-critic
run names 8. The critic's filtered set leads the model to summarise rather
than list -- the one question where the critic's effect is large, and it is
negative (**C**). Free-SQL's query is `label LIKE '%b%tag%'` over detector
objects: 44 papers, all 11 gold among them, 15 outside the judged set. Under
`judged_f1` that costs nothing, which is why a 44-paper list scores 0.59; under
an exact-set metric it would not. This question is where the metric's blind
spot to over-listing is most visible.

### gf-01-met -- analyses that require missing transverse momentum in their selection (gold 3, judged 13)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.29 | 12.3 | 0.5 | 0.5 | 11.3 |
| free-SQL OR (3) | 0.46 | 31.7 | 2.0 | 3.7 | 26.0 |
| typed no-critic QwQ (2) | 0.00 | 1.0 | 0.0 | 0.0 | 1.0 |
| typed critic QwQ (2) | 0.40 | 23.0 | 1.0 | 1.0 | 21.0 |
| free-SQL QwQ (2) | 0.20 | 6.0 | 0.5 | 0.5 | 5.0 |

Replay: reach 3/3 always. Gabriel judged 13 papers of which 3 yes, so the
metric is on 13 papers and almost everything either system names is outside
that set. The QwQ no-critic answer is the archetype of **class C**: "The graph
identifies 23 analyses that require MET ... Examples include 2004.14060,
2006.05880, and others listed in the tool response" -- a count and a pointer
where a list was asked for. This question is under-judged more than it is
mis-answered; in the write-up it belongs with gf-01-condition as evidence
about the metric.

### gf-02 -- backgrounds estimated with an ABCD method, or an ABCD-style sideband or matrix method (gold 11, judged 19)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.65 | 5.8 | 5.7 | 0.0 | 0.2 |
| free-SQL OR (3) | 0.92 | 12.7 | 9.7 | 0.3 | 2.7 |
| typed no-critic QwQ (2) | 0.71 | 6.0 | 6.0 | 0.0 | 0.0 |
| typed critic QwQ (2) | 0.71 | 6.0 | 6.0 | 0.0 | 0.0 |
| free-SQL QwQ (2) | 0.56 | 9.0 | 5.5 | 0.5 | 3.0 |

Replay: the typed planner reaches exactly 6 of 11 in every run and names all
6 with no false positive; the 5 missed papers all carry the concept as an
entity: "Matrix method for fake/non-prompt lepton estimation", "Z+jets
fake-factor measurement region", "Dilepton SS control region (fake-lepton
enriched)". The model searches "ABCD method" and never searches the
question's other two clauses -- **class A, the enumeration miss**, and the
cleanest case of it. Free-SQL on OpenRouter searched more broadly
(`modified-abcd-estimate` and a wider entity list) and reached 9.7 of 11.
The mechanism built for this, enumeration expansion (D-131), lifted the typed
system's gf-02 from 0.65 to 0.77 once it fired on facets-first runs.

### gf-03 -- analyses using the HistFitter framework (gold 5, judged 9)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.89 | 8.0 | 4.0 | 0.0 | 4.0 |
| free-SQL OR (3) | 0.89 | 5.0 | 4.0 | 0.0 | 1.0 |
| typed no-critic QwQ (2) | 0.89 | 4.0 | 4.0 | 0.0 | 0.0 |
| typed critic QwQ (2) | 0.89 | 8.0 | 4.0 | 0.0 | 4.0 |
| free-SQL QwQ (2) | 0.33 | 10.0 | 1.0 | 0.0 | 9.0 |

Everyone names the same 4 of 5. The fifth is **class D**: HistFitter appears
on that paper only inside a verbatim quote, no entity carries it. That is an
extraction-time gap, not a query-time one; indexing quotes was tested and did
not change reach (D-121). The QwQ free-SQL run wrote a wrong query (10 named,
1 gold) -- the free-SQL baseline's variance is in the SQL it happens to write.

### gf-04 -- analyses that unfold their measured distributions (gold 18, judged 26)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.67 | 9.8 | 9.3 | 0.2 | 0.3 |
| free-SQL OR (3) | n/a (older gold) | 11.7 | 9.3 | 0.7 | 1.7 |
| typed no-critic QwQ (2) | 0.71 | 10.0 | 10.0 | 0.0 | 0.0 |
| typed critic QwQ (2) | 0.71 | 10.0 | 10.0 | 0.0 | 0.0 |
| free-SQL QwQ (2) | 0.71 | 11.0 | 10.0 | 0.0 | 1.0 |

Replay: reach 10-11 of 18, named with perfect precision. The 7-8 never reached
split three ways: 3 papers where nothing on the paper mentions unfolding
(concept absent from the graph), 3 where it is quote-only (**D**), 2 where an
entity exists but under another surface form ("fiducial phase space (particle
level)", "TUnfold bin-by-bin migration correction") (**A**). This question's
ceiling is set by the graph, not the query system: at most 12-13 of 18 are
reachable by any query. Both systems reach it.

### gf-05 -- analyses that reconstruct a Higgs-boson candidate as a physical object they select on (gold 16, judged 38)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.07 | 2.0 | 0.7 | 1.3 | 0.0 |
| free-SQL OR (3) | 0.11 | 2.7 | 1.0 | 1.0 | 0.7 |
| typed no-critic QwQ (2) | 0.16 | 2.0 | 1.5 | 0.5 | 0.0 |
| typed critic QwQ (2) | 0.11 | 1.5 | 1.0 | 0.5 | 0.0 |
| free-SQL QwQ (2) | 0.33 | 5.0 | 3.5 | 1.5 | 0.0 |

The worst question for every control, and the clearest **class A**. Replay:
reach 2 of 16; 14 gold papers never reached, and on 13 of them an entity
carrying the concept exists. The typed planner searches "Higgs" with
`kind=detector_object` and gets 4 entities -- three signal regions of one
paper -- and answers "Only 2006.05880 explicitly reconstructs a Higgs-boson
candidate". The missed papers carry the concept as *processes* and *channels*
("H → bb̄ decay", "VH production", "0-lepton channel: ZH → νν bb̄", "Higgs
boson decay to a pair of muons"), never as a detector object. Free-SQL does
the same thing in SQL: `entity_id IN (higgs-boson-particle-level,
higgs_boson)` → 2 papers. The kind filter starves the search (D-119), and the
question's "rather than merely producing it" adds a semantic condition no
label encodes. The kind fallback (D-119) took live reach from 0.21 to 0.81 and
the full stack took judged_f1 from 0.07 to 0.41 (D-139); what remains is the
semantic condition.

### gf-07 -- analyses with a ttZ background and a control region used to normalise it (gold 10, judged 53)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.21 | 1.7 | 0.5 | 1.2 | 0.0 |
| free-SQL OR (3) | 0.43 | 3.7 | 2.7 | 1.0 | 0.0 |
| typed no-critic QwQ (2) | 0.00 | 1.5 | 0.0 | 0.5 | 1.0 |
| typed critic QwQ (2) | 0.00 | 1.5 | 0.0 | 0.0 | 1.5 |
| free-SQL QwQ (2) | 0.00 | 1.0 | 0.0 | 1.0 | 0.0 |

Replay: the typed planner reaches 8-9 of 10 and names 0-2. The trace shows
why: it retrieves the ttZ backgrounds (27 entities) and the control regions
(60), takes the subjects of each (52 and 50 results), and then has to
*intersect* the two paper sets -- and there is no tool for that. It tries
`papers_of(set_3)` on a set that does not exist (`unknown_entity_id`, twice),
runs out of rounds, and the QwQ record ends as chain-of-thought prose ("Okay,
let me try to figure out how to approach this ..."), which is scored as the
answer. **Class C by way of a missing operation**: a compound condition (a
background AND a region normalising it) that the tool set cannot express in
one step. Free-SQL can write the join, and on OpenRouter it reaches 2.7 of 10
with it -- the predicate for "control region used to normalise *that*
background" is not a single edge in the graph either. Gabriel judged 53 papers
here, the largest set, so false positives are cheap to make. In D-135's
decomposition this question holds most of the "reached but never graded"
bucket: the gold enters through `subjects_of` and never meets `papers_of`.

### gf-08 -- analyses requiring exactly two electrons OR exactly two muons as parallel selections (gold 24, judged 44)

| control | judged_f1 | named | gold | FP | outside |
|---|---|---|---|---|---|
| typed+critic OR (6) | 0.08 | 1.8 | 1.0 | 0.7 | 0.2 |
| free-SQL OR (3) | 0.10 | 0.7 | 0.7 | 0.0 | 0.0 |
| typed no-critic QwQ (2) | 0.44 | 23.5 | 9.0 | 8.0 | 6.5 |
| typed critic QwQ (2) | 0.04 | 1.0 | 0.5 | 0.0 | 0.5 |
| free-SQL QwQ (2) | 0.00 | 2.5 | 0.0 | 0.0 | 2.5 |

Replay: the typed planner reaches all 24 gold papers in every QwQ run (via
`facets(objects=[Electron, Muon])` and `subjects_of` over 200+ regions), then
names 1 with the critic and 7-11 without it, with 7-9 false positives. Two
failures stacked. The list is long (24) and the model writes a summary or a
handful -- **class C**, and with the critic on it collapses to one paper. And
both systems first search the literal phrase: "exactly two electrons" as a
detector-object label. Free-SQL takes that to its conclusion, `label LIKE
'%exactly two electrons%' INTERSECT ... '%exactly two muons%'` → 0 rows →
"the knowledge graph does not contain any analyses that require ..." -- a
false abstention from a surface-form miss (**A**). This is the question with
the largest handoff loss in the whole set (42 of 86 graded-but-unnamed gold
slots in D-135), the one the ranked answer was built for (D-135, D-136: 0.11
→ 0.34-0.36), and the one whose six-repeat spread is 0.14 (D-138).

## 3. What the nine say together

| class | questions | what it is | fixed by |
|---|---|---|---|
| A surface form / enumeration | gf-02, gf-05, gf-08 (literal phrase), gf-04 (2 papers) | the concept is in the graph under labels the query never asks for; the question's other clauses are never searched | enum-expand (gf-02), kind fallback (gf-05); the literal-phrase habit is unfixed |
| B truncation | gf-08 (24 gold vs 25 rows) | the list is longer than the rows shown | max-rows 100 (D-120/122) |
| C summarise / point / no intersection | gf-01-condition, gf-01-met, gf-07, gf-08 | the papers are on the page and the answer is a count, a pointer, a summary, or chain-of-thought | name-ids (D-117), answer gate, ranked answer (D-128/135/136); gf-07 needs a set operation the tools lack |
| D quote-only | gf-03 (1), gf-04 (3) | the concept exists only in a verbatim quote, no entity carries it | not at query time (D-121) |
| E tag is not selection | gf-01 | co-occurrence of two objects on a paper is returned where joint selection was asked | nothing tried worked (D-124) |
| graph ceiling | gf-04 (3 absent) | nothing on the paper mentions the concept | extraction |

Two facts about the controls specifically. The relevance critic's effect on
these nine is concentrated where it is negative: on gf-01-condition and gf-08
the critic-on control names 1 paper where the critic-off control names 8 and
7-11. And the free-SQL baseline is not weaker than the typed planner on
retrieval -- it is weaker on Gabriel's questions because it searches literal
phrases and abstains when they return nothing (gf-08), and stronger on the
generated questions because those are one query each.

## 4. Metric caveats to carry into the chapter

1. `judged_f1` ignores papers named outside Gabriel's judged set: a 44-paper
   answer to gf-01-condition scores 0.59. Report the outside-universe count
   beside it (this document does).
2. `set_f1` falls back to the retrieval footprint when nothing is named; the
   typed controls take that fallback on 30-50% of set answers. Report
   named-only and the fallback rate (section 1).
3. gf-01-met has 3 yes in 13 judged; gf-07 has 10 in 53. Per-question deltas
   on the 9 move by ±0.05 on gf-08 alone (D-138); three repeats are the
   minimum and single-repeat rows above are marked as such.
4. The free-SQL OpenRouter run used for the traces (28311) predates the
   current gold; its judged numbers are not comparable and the rerun (85223)
   replaces them in section 1.7 when it lands.
