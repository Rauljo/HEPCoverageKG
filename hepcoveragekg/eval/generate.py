"""
Evaluation harness: questions generated backwards from the graph (S-33).

Run the query FIRST, get the answer, then write the question it belongs to:

    SQL   count(sample_uses_generator, cluster(Pythia))   ->  58 papers
    ->    "How many analyses used Pythia?"

The label is the query result, never something a model invented. This is how the
text-to-SQL sets (Spider, BIRD) and the graph-QA ones (GrailQA) are built, for
the same reason: it is the only way to get exact labels at scale.

Two things worth being explicit about, because both are easy to get wrong and
neither is visible in the output.

**Where truth comes from, and what that means it measures.** Truth is computed
with `query/templates.py` -- the same SQL the planner runs. So these questions
measure **L1: did the agent pick the right operation and the right entities**
(system.md S-32). They cannot catch a bug *inside* a template, because a wrong
template would produce a wrong truth and a matching wrong answer.
That is the correct trade: template correctness is pinned by
`tests/test_templates.py` (which encodes the two known traps), while what is
actually uncertain -- and what the whole planner exists to do -- is choosing the
operation and resolving the entities.
The alternative, hand-written SQL per probe, was tried in the 2026-08-02 seed set
and immediately produced an unresolvable disagreement: a `LIKE '%herwig%'` truth
said 22 papers, the system said 21, and neither was obviously wrong because they
were answering subtly different questions. An independent truth is only useful if
it is independent *and* means the same thing.

**Generation must not need a model.** Phrasings are templates, so the whole
generator is deterministic, testable, and runnable with the cluster down. An LLM
pass to make the wording natural is an *upgrade*, applied afterwards -- the same
stub-then-real split as `systems.py`. It also means paraphrase groups come free:
several phrasings of one probe are, by construction, the same question asked
differently (S-34).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .questions import Question, Truth

# Predicates we can phrase as natural English, with the verb that fits. A
# predicate absent here is skipped rather than rendered awkwardly -- a question
# nobody would ask measures nothing, and "how many analyses object_has_selection
# X" is not a question.
#
# Only predicates whose object is an ENTITY appear: `object_has_selection` and
# `region_has_selection` carry their object as a free-text value, so there is no
# cluster to count over.
PREDICATE_PHRASES: dict[str, dict[str, str]] = {
    "sample_uses_generator": {
        "verb": "used the generator", "short": "used", "noun": "generator",
    },
    "result_has_systematic": {
        "verb": "report the systematic uncertainty", "short": "report",
        "noun": "systematic uncertainty",
    },
    "result_measures_observable": {
        "verb": "measured", "short": "measured", "noun": "observable",
    },
    "result_uses_sample": {
        "verb": "used the sample", "short": "used", "noun": "sample",
    },
    "region_requires_object": {
        "verb": "define a region requiring", "short": "require",
        "noun": "detector object",
    },
    "result_defines_region": {
        "verb": "define the region", "short": "define", "noun": "event region",
    },
    "result_defines_object": {
        "verb": "define the object", "short": "define", "noun": "detector object",
    },
    "background_uses_method": {
        "verb": "estimate a background using", "short": "use",
        "noun": "background method",
    },
    "result_estimates_background": {
        "verb": "estimate the background", "short": "estimate", "noun": "background",
    },
}

# Paper-anchored phrasings (Tier A, S-59). Written out per predicate rather than
# assembled from `verb`/`noun`, because English tense and plurals do not survive
# that kind of template assembly -- "How many detector objects does analysis X
# require?" reads properly; a generated version does not.
#
# More predicates appear here than in PREDICATE_PHRASES: a paper-anchored
# question needs no concept resolution, so predicates whose objects are free-text
# VALUES can be asked about too. Those get counting questions only -- there is no
# entity id to match a named answer against.
PAPER_PHRASINGS: dict[str, dict[str, str]] = {
    "sample_uses_generator": {
        "set": "Which generators does analysis {paper} use?",
        "count": "How many distinct generators does analysis {paper} use?"},
    "result_has_systematic": {
        "set": "Which systematic uncertainties does analysis {paper} report?",
        "count": "How many systematic uncertainties does analysis {paper} report?"},
    "result_uses_sample": {
        "set": "Which simulated samples does analysis {paper} use?",
        "count": "How many simulated samples does analysis {paper} use?"},
    "region_requires_object": {
        "set": "Which detector objects do the regions of analysis {paper} require?",
        "count": "How many distinct detector objects do the regions of analysis {paper} require?"},
    "result_defines_object": {
        "set": "Which detector objects does analysis {paper} define?",
        "count": "How many detector objects does analysis {paper} define?"},
    "channel_has_region": {
        "set": "Which event regions do the channels of analysis {paper} contain?",
        "count": "How many event regions do the channels of analysis {paper} contain?"},
    "result_estimates_background": {
        "set": "Which backgrounds does analysis {paper} estimate?",
        "count": "How many backgrounds does analysis {paper} estimate?"},
    "background_uses_method": {
        "set": "Which background-estimation methods does analysis {paper} use?",
        "count": "How many background-estimation methods does analysis {paper} use?"},
    "result_measures_observable": {
        "set": "Which observables does analysis {paper} measure?",
        "count": "How many observables does analysis {paper} measure?"},
    "result_uses_statistical_method": {
        "set": "Which statistical methods does analysis {paper} use?",
        "count": "How many statistical methods does analysis {paper} use?"},
    "result_targets_process": {
        "set": "Which physics processes does analysis {paper} target?",
        "count": "How many physics processes does analysis {paper} target?"},
    "result_uses_dataset": {
        "set": "Which datasets does analysis {paper} use?",
        "count": "How many datasets does analysis {paper} use?"},
    "result_has_collision_system": {
        "set": "Which collision systems does analysis {paper} study?",
        "count": "How many collision systems does analysis {paper} study?"},
    # Value-carrying: counting only, no entity to name.
    "object_has_selection": {
        "count": "How many object selection requirements does analysis {paper} define?"},
    "region_has_selection": {
        "count": "How many region selection requirements does analysis {paper} define?"},
    "result_reports_quantity": {
        "count": "How many quantities does analysis {paper} report?"},
}

# Several wordings of one probe. They are paraphrases by construction, which is
# what makes paraphrase-invariance measurable without trusting a model to have
# preserved the meaning.
COUNT_PHRASINGS = (
    "How many analyses {verb} {label}?",
    "In how many papers is {label} recorded as {noun}?",
    "Count the analyses that {verb} {label}.",
)
SET_PHRASINGS = (
    "Which analyses {verb} {label}?",
    "List the papers that {verb} {label}.",
)

# Below this, a question is about a corner of the graph rather than about the
# literature -- and a count of 1 cannot distinguish a good system from a lucky
# one. Kept low deliberately: rare entities are where retrieval should fail, and
# rare entities are what a coverage map is FOR.
MIN_PAPERS = 2


@dataclass
class Probe:
    """A query with a known answer, before it has been phrased as a question."""
    family: str
    predicate: str
    entity_id: str
    label: str
    kind: str
    papers: list[str]
    cluster_size: int      # distinct CANONICAL clusters the concept spans
    surface_forms: int     # distinct entity ids in the concept

    @property
    def n(self) -> int:
        return len(self.papers)

    @property
    def dedup_sensitive(self) -> bool:
        """The answer changes if OUR deduplication is turned off (S-11).

        Cluster size only. Two mechanisms produce multiple spellings and they are
        not the same thing, which cost a wrong tag on 2026-08-02:

          cluster_size > 1   the aliases layer merged separate entity_ids into
              one. Turning dedup off un-merges them and the count changes. This
              is the dedup-sensitive case.

          surface_forms > 1  extraction gave one entity_id several labels across
              papers. Nothing we do created that and nothing we turn off undoes
              it.

        **This flipped when the unit changed from cluster to concept, and the
        reasoning is worth keeping because it is easy to get backwards.**

        At CLUSTER level the test was `cluster_size > 1` -- "our aliases layer
        merged separate ids into this one, so turning it off changes the answer."

        At CONCEPT level that is wrong, and in fact inverted: `cluster_size` now
        counts how many *separate* canonical clusters the concept still spans, so
        a big number means deduplication **failed** to unify it. What actually
        makes the answer depend on treating several ids as one thing is simply
        that the concept contains several ids at all -- `surface_forms > 1`.

        Measured: 348 of 369 generated questions. That is near-universal, and it
        is the finding rather than a broken tag -- **a concept-level question is
        almost always about several entity ids**, which is the argument for the
        aliases layer stated as a number. `cluster_size` stays in provenance as
        the graded stratifier: how much of that unification dedup already did.
        """
        return self.surface_forms > 1

    def difficulty(self) -> str:
        """Rare entities are hard: few papers, and often only one spelling to
        find them by."""
        if self.n >= 20:
            return "easy"
        return "medium" if self.n >= 5 else "hard"

    def needs(self) -> list[str]:
        out = ["sql"]
        if self.dedup_sensitive:
            out.append("merged_entities")
        return out


def _qid(family: str, predicate: str, entity_id: str, variant: int = 0) -> str:
    """Stable across regeneration -- ids that move break every cross-run
    comparison, and the breakage is silent."""
    key = f"{family}|{predicate}|{entity_id}|{variant}".encode()
    return f"gen-{family}-{hashlib.sha1(key).hexdigest()[:8]}"


def _normalise(label: str) -> str:
    """Lowercase, split on punctuation, AND split letters from digits.

    That last part matters: `PYTHIA8` has no separator, so without it its head
    word is "pythia8" while `Pythia 8` gives "pythia" -- one concept split in
    two, which is the exact failure this module exists to avoid. Caught by the
    fixture, not by the real graph, where most labels happen to have a space.
    """
    text = re.sub(r"[^a-z0-9]+", " ", (label or "").lower())
    return re.sub(r"(?<=[a-z])(?=\d)", " ", text).strip()


def _head(label: str) -> str:
    """The word a physicist would actually name the thing by.

    `Pythia 8.230`, `PYTHIA 8.212` and `Pythia 8` are all *Pythia*. The version
    is a detail of the sample, not of the question anyone asks.
    """
    parts = _normalise(label).split()
    return parts[0] if parts else ""


def _papers_for(conn, predicate: str, object_ids: list[str]) -> list[str]:
    """The papers a predicate links these objects to -- the SAME join `count`
    uses, returning ids instead of a number.

    Not `list_papers`, and the reason is a defect found on 2026-08-02:
    **`list_papers` and `count` disagree.** `count` ties the subject's occurrence
    to the bundle the assertion came from (`eo.bundle_id = a.bundle_id`).
    `list_papers` composes `subjects_of` + `papers_of`, and `papers_of` is a
    generic entity -> papers lookup that cannot know which assertion it came
    from, so it returns every paper the subject appears in. A sample entity
    shared across ten papers contributes all ten even when only nine assert the
    link.

    Measured on the Pythia concept: `count` 58 papers, `list_papers` 59.
    `count` is right. Until `list_papers` is fixed in the query layer, truth is
    computed here the way `count` computes it, so a set question and a counting
    question about the same concept cannot disagree with each other.
    """
    from ..query import templates as T

    ids = T.expand_canonical(conn, object_ids)
    if not ids:
        return []
    rows = conn.execute(
        "SELECT DISTINCT eo.paper_id AS paper_id"
        "  FROM assertion a"
        "  JOIN entity_occurrence eo"
        "    ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id"
        " WHERE a.predicate = ?"
        f"   AND a.object_id IN ({','.join('?' * len(ids))})",
        (predicate, *ids),
    ).fetchall()
    return sorted({r["paper_id"] for r in rows if r["paper_id"]})


def probes(conn, predicate: str, limit: int = 40, min_papers: int = MIN_PAPERS,
           max_entities: int = 120, max_paper_share: float = 1.0) -> Iterator[Probe]:
    """Every CONCEPT a predicate is asked about, with its answer.

    Grouped by concept, not by canonical cluster -- and getting this wrong was
    the substantive bug of 2026-08-02, worth recording because the failure was
    invisible in every summary.

    The first version generated one question per canonical cluster: *"How many
    analyses used the generator Pythia 8?"*, truth 46. The system answered 58.
    Scored across 36 questions it got **1 right**, which looks like a broken
    system and was a broken question set.

    The cause: `search` returns a SET of entities and `count` counts over all of
    them, which is correct -- a physicist asking about Pythia means every Pythia.
    But `Pythia 8` and `Pythia 8.230` are separate canonical clusters, so a
    cluster is NOT a concept, and a question phrased at concept level cannot have
    a cluster-level answer. Measured on `region_requires_object`: one `Electron`
    cluster is 20 papers, all electron-ish entities are 50, and the system said
    52.

    So concepts are built by head word, and the truth is the count over the whole
    concept. That reproduces **Pythia = 58**, the one figure independently
    verified end to end on 2026-07-31.

    `max_entities` drops over-general head words (`Single top quark background`
    -> "single", which would match half the graph).

    `max_paper_share` defaults to 1.0, i.e. OFF. It was briefly 0.9, on the
    theory that a concept covering nearly every paper is true but uninformative
    -- and it silently deleted the single best question in the set, because
    Pythia is in 58 of 60 papers (97%). For a COVERAGE map "almost every analysis
    uses this" is a finding, not noise. Kept as a parameter, off by default.
    """
    from ..query import templates as T

    # Only needed by the share filter, which is off by default -- so it is not a
    # hard dependency on the `paper` table.
    total_papers = 1
    if max_paper_share < 1.0:
        total_papers = conn.execute("SELECT COUNT(*) FROM paper").fetchone()[0] or 1

    rows = conn.execute(
        """
        SELECT DISTINCT e.entity_id AS entity_id, e.label AS label, e.kind AS kind
          FROM assertion a JOIN entity e ON e.entity_id = a.object_id
         WHERE a.predicate = ? AND e.label IS NOT NULL AND TRIM(e.label) <> ''
        """,
        (predicate,),
    ).fetchall()

    groups: dict[tuple[str, str], list] = {}
    for row in rows:
        head = _head(row["label"])
        if not head:
            continue
        groups.setdefault((row["kind"] or "", head), []).append(row)

    found: list[Probe] = []
    for (kind, head), members in groups.items():
        if len(members) > max_entities:
            continue  # an over-general head word, e.g. "single"

        ids = [m["entity_id"] for m in members]
        papers = _papers_for(conn, predicate, ids)
        if len(papers) < min_papers:
            continue
        if max_paper_share < 1.0 and len(papers) / total_papers > max_paper_share:
            continue

        # The name to ask about: the shortest label in the concept, which is
        # almost always the bare form ("Pythia", not "Pythia 8.230 (A14 tune)").
        label = min((m["label"].strip() for m in members), key=len)
        clusters = {
            (conn.execute("SELECT canonical_id FROM entity_canonical WHERE entity_id = ?",
                          (i,)).fetchone() or [i])[0]
            for i in ids
        }
        found.append(Probe(
            family="count", predicate=predicate, entity_id=f"{kind}:{head}",
            label=label, kind=kind, papers=papers,
            cluster_size=len(clusters), surface_forms=len(ids),
        ))

    found.sort(key=lambda p: (-p.n, p.entity_id))
    yield from found[:limit]


def _phrase(template: str, probe: Probe) -> str:
    phrases = PREDICATE_PHRASES[probe.predicate]
    return template.format(label=probe.label, verb=phrases["verb"],
                           short=phrases["short"], noun=phrases["noun"])


def questions_from(probe: Probe, *, paraphrases: int = 1,
                   set_question: bool = True) -> list[Question]:
    """One probe -> a counting question, its paraphrases, and a set question.

    The paraphrases share a `group`, which is what `metamorphic.py` needs to
    check that a rewording does not change the answer (S-34).
    """
    out: list[Question] = []
    group = f"mm-{probe.predicate}-{probe.entity_id}" if paraphrases > 1 else None
    anchor = _qid("count", probe.predicate, probe.entity_id, 0)
    provenance = {
        "generated_at": "backwards from the graph (S-33)",
        "template": "templates.list_papers",
        "predicate": probe.predicate,
        "entity_id": probe.entity_id,
        "cluster_size": probe.cluster_size,
        "surface_forms": probe.surface_forms,
    }

    for variant in range(min(paraphrases, len(COUNT_PHRASINGS))):
        out.append(Question(
            qid=_qid("count", probe.predicate, probe.entity_id, variant),
            text=_phrase(COUNT_PHRASINGS[variant], probe),
            source="generated", split="dev", shape="count",
            needs=probe.needs(), difficulty=probe.difficulty(),
            truth=Truth(kind="count", value=probe.n, papers=probe.papers),
            truth_source="sql", provenance=provenance,
            group=group,
            relation=None if variant == 0 else f"paraphrase_of:{anchor}",
        ))

    if set_question:
        out.append(Question(
            qid=_qid("set", probe.predicate, probe.entity_id, 0),
            text=_phrase(SET_PHRASINGS[0], probe),
            source="generated", split="dev", shape="set",
            needs=probe.needs(), difficulty=probe.difficulty(),
            truth=Truth(kind="set", value=probe.n, papers=probe.papers),
            truth_source="sql", provenance=provenance,
            group=group, relation=f"set_form_of:{anchor}" if group else None,
        ))
    return out


def known_positive_questions(conn, predicate: str, limit: int = 40,
                             phrasings: int = 2) -> list[Question]:
    """Questions asking only whether a paper we KNOW qualifies comes back.

    The strongest shape available, and the only one that works on the concepts
    S-58 has to discard.

    Every other tier needs the *exact* answer set, which is why ambiguity kills
    them: is `Pythia 8.230` inside the Pythia concept or outside? But ambiguity
    only moves the **boundary** of the set -- it never moves a **known core
    member** out of it. Paper 2307.01094 says Pythia in its own text, so it
    belongs in any correct answer under every reading of the concept.

    So the truth here is a *subset*, not a set: these papers must appear;
    anything else is not counted against the system. Ground truth is free and
    certain, taken from the paper's own assertion -- no deduplication, no
    invariance test, no judgement.

    **Recall alone is gameable** and the scorer must say so: returning all 60
    papers scores a perfect recall and means nothing. `paper_recall` is therefore
    reported beside the size of the returned set, and a run whose sets approach
    the corpus size is a failed run whatever its recall.
    """
    if predicate not in PREDICATE_PHRASES:
        return []

    rows = conn.execute(
        """
        SELECT e.entity_id AS entity_id, e.label AS label, eo.paper_id AS paper_id,
               COUNT(*) AS n
          FROM assertion a
          JOIN entity e ON e.entity_id = a.object_id
          JOIN entity_occurrence eo
            ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id
         WHERE a.predicate = ? AND e.label IS NOT NULL AND TRIM(e.label) <> ''
         GROUP BY e.entity_id, eo.paper_id
         ORDER BY n DESC, e.entity_id, eo.paper_id
         LIMIT ?
        """,
        (predicate, limit * 4),
    ).fetchall()

    # Full set size per entity, used only to decide whether a prose answer could
    # reasonably list it (see scoring.PROSE_LISTABLE).
    concept_papers: dict[str, int] = {}
    for r in conn.execute(
        """SELECT a.object_id AS eid, COUNT(DISTINCT eo.paper_id) AS n
             FROM assertion a
             JOIN entity_occurrence eo
               ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id
            WHERE a.predicate = ? GROUP BY a.object_id""", (predicate,)):
        concept_papers[r["eid"]] = r["n"]

    out: list[Question] = []
    seen: set[str] = set()
    for row in rows:
        if len(out) >= limit * phrasings:
            break
        key = f"{row['entity_id']}|{row['paper_id']}"
        if key in seen:
            continue
        seen.add(key)

        probe = Probe(family="known", predicate=predicate, entity_id=row["entity_id"],
                      label=row["label"].strip(), kind="", papers=[row["paper_id"]],
                      cluster_size=1, surface_forms=1)
        group = f"mm-known-{row['entity_id']}-{row['paper_id']}"
        anchor = _qid("known", predicate, key, 0)
        for variant in range(min(phrasings, len(SET_PHRASINGS))):
            out.append(Question(
                qid=_qid("known", predicate, key, variant),
                text=_phrase(SET_PHRASINGS[variant], probe),
                source="generated", split="dev", shape="set",
                # No `merged_entities`: the point is that this survives without it.
                needs=["sql"], difficulty="medium",
                truth=Truth(kind="subset", value=1, papers=[row["paper_id"]]),
                truth_source="sql",
                provenance={"generated_at": "known-positive retrieval",
                            "predicate": predicate, "entity_id": row["entity_id"],
                            "known_paper": row["paper_id"],
                            "assertions_in_that_paper": row["n"],
                            # How many papers the whole concept covers. Above
                            # PROSE_LISTABLE the "did it mention it?" scorer
                            # abstains -- no prose answer lists 58 papers.
                            "concept_papers": concept_papers.get(row["entity_id"], 0),
                            "note": "recall of a certain positive; extra papers are not penalised"},
                group=group,
                relation=None if variant == 0 else f"paraphrase_of:{anchor}",
            ))
    return out


def union_question(a: Probe, b: Probe) -> Optional[Question]:
    """"A or B" -- deliberately UNLABELLED, and that is the point.

    Its value is the metamorphic relation, not a truth value: `count(A or B)`
    must be at least `max(count(A), count(B))`. A violation is a guaranteed bug
    found with no label at all (S-34), and this is exactly what caught the
    2026-08-02 failure -- the system answered 31 for a union whose smaller half
    is 58.
    """
    if a.predicate != b.predicate or a.entity_id == b.entity_id:
        return None
    phrases = PREDICATE_PHRASES[a.predicate]
    anchor_a = _qid("count", a.predicate, a.entity_id, 0)
    anchor_b = _qid("count", b.predicate, b.entity_id, 0)
    return Question(
        qid=_qid("union", a.predicate, f"{a.entity_id}+{b.entity_id}", 0),
        text=f"How many analyses {phrases['verb']} {a.label} or {b.label}?",
        source="generated", split="dev", shape="count",
        needs=["sql", "hops"], difficulty="medium",
        truth=Truth(), truth_source="none",
        provenance={"generated_at": "backwards from the graph (S-33)",
                    "relation": "union", "parts": [anchor_a, anchor_b],
                    "part_counts": [a.n, b.n]},
        group=f"mm-union-{a.entity_id}-{b.entity_id}",
        relation=f"union_of:{anchor_a},{anchor_b}",
    )


def generate(conn, *, predicates: Optional[Iterable[str]] = None,
             per_predicate: int = 12, paraphrases: int = 3,
             unions: int = 1) -> list[Question]:
    """The whole set: counting questions, paraphrases, set forms, unions."""
    predicates = list(predicates or PREDICATE_PHRASES)
    out: list[Question] = []
    for predicate in predicates:
        if predicate not in PREDICATE_PHRASES:
            continue
        found = list(probes(conn, predicate, limit=per_predicate))
        for probe in found:
            out.extend(questions_from(probe, paraphrases=paraphrases))
        # Unions pair the most common with a mid-frequency one, so the relation
        # is checkable without the two sets being nearly identical.
        for i in range(min(unions, max(len(found) - 1, 0))):
            question = union_question(found[i], found[i + len(found) // 2])
            if question:
                out.append(question)

    seen: set[str] = set()
    unique = []
    for q in out:
        if q.qid not in seen:
            seen.add(q.qid)
            unique.append(q)
    return unique


def paper_questions(conn, predicates: Optional[Iterable[str]] = None,
                    papers: Optional[Iterable[str]] = None,
                    min_items: int = 2, max_items: int = 25) -> list[Question]:
    """Tier A (S-59): questions anchored on ONE paper. The backbone of the set.

    Every other tier has to decide *which entity ids count as the thing being
    asked about* before it can state an answer -- and that decision **is** the
    deduplication problem, which is why the first two generators re-derived a
    worse version of it and failed (1/36, then 0/36).

    A paper id is a **hard identifier**. `2307.01094` is exactly one paper: no
    spellings, no variants, no boundary to draw. So the truth is exact, and it
    does not move when deduplication improves or breaks.

    It also reaches ground nothing else can. Three of the highest-volume
    predicates in the graph are 97-98% free-text VALUES (`object_has_selection`,
    `region_has_selection`, `result_reports_quantity`), so they have no entity to
    deduplicate and no concept to count over -- 31% of all assertions are
    invisible to concept-level questions and readable here.

    Two shapes per (paper, predicate):
      count   exact number. Works for value-carrying predicates too.
      labels  which entities. Truth is the entity IDS, not the label text --
              matching prose against "Simultaneous binned maximum-likelihood fit
              to SR m_bb distributions and CR event yields..." would measure
              wording. Readable labels go in provenance for the prose scorer.

    `max_items` drops questions whose answer is a long list. The planner writes
    out ~11 items before it stops, so a 40-item answer cannot be judged on what
    it names -- the same limit that made concept-level prose scoring unfair.
    """
    predicates = list(predicates or PAPER_PHRASINGS)
    wanted_papers = set(papers) if papers else None
    out: list[Question] = []

    for predicate in predicates:
        phrases = PAPER_PHRASINGS.get(predicate)
        if not phrases:
            continue

        rows = conn.execute(
            """
            SELECT eo.paper_id AS paper_id, a.object_id AS object_id,
                   a.object_value AS object_value, e.label AS label
              FROM assertion a
              JOIN entity_occurrence eo
                ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id
              LEFT JOIN entity e ON e.entity_id = a.object_id
             WHERE a.predicate = ?
            """,
            (predicate,),
        ).fetchall()

        by_paper: dict[str, dict[str, tuple[str, str]]] = {}
        for row in rows:
            paper = row["paper_id"]
            if not paper or (wanted_papers and paper not in wanted_papers):
                continue
            text = row["label"] if row["object_id"] else row["object_value"]
            if not (text or "").strip():
                continue
            # Deduplicate WITHIN the paper by normalised label (D-046). 22 cases
            # across 7 papers store one thing under two entity ids; without this
            # the truth would list a region twice and a system naming it once
            # would score as wrong.
            key = _normalise(text)
            by_paper.setdefault(paper, {}).setdefault(
                key, (row["object_id"] or "", text.strip()))

        for paper, items in sorted(by_paper.items()):
            n = len(items)
            if not (min_items <= n <= max_items):
                continue
            ids = sorted(i for i, _ in items.values() if i)
            labels = sorted(lbl for _, lbl in items.values())
            provenance = {
                "generated_at": "per-paper (Tier A)",
                "predicate": predicate,
                "paper": paper,
                "labels": labels[:40],
                "value_carrying": not ids,
            }

            out.append(Question(
                qid=_qid("paper", predicate, f"{paper}|count", 0),
                text=phrases["count"].format(paper=paper),
                source="generated", split="dev", shape="count",
                # No `merged_entities`: the paper anchors it, so the answer does
                # not change with deduplication. That is the whole point.
                needs=["sql"], difficulty="easy" if n <= 5 else "medium",
                truth=Truth(kind="count", value=n), truth_source="sql",
                provenance=provenance, group=f"mm-paper-{predicate}-{paper}",
            ))

            if "set" in phrases and ids:
                out.append(Question(
                    qid=_qid("paper", predicate, f"{paper}|set", 0),
                    text=phrases["set"].format(paper=paper),
                    source="generated", split="dev", shape="set",
                    needs=["sql"], difficulty="easy" if n <= 5 else "medium",
                    truth=Truth(kind="labels", value=n, items=ids),
                    truth_source="sql", provenance=provenance,
                    group=f"mm-paper-{predicate}-{paper}",
                    relation=f"set_form_of:{_qid('paper', predicate, f'{paper}|count', 0)}",
                ))
    return out


def load_merge_map(proposals_path: Optional[str] = None) -> dict[str, set[str]]:
    """Approved merge pairs as an undirected map, **one hop, never chained**.

    Chaining is what produced the 75-member statistical-methods cluster on
    2026-08-02 -- profile likelihood welded to the CLs procedure because each
    neighbouring pair looked similar. Neighbours-of-neighbours are not merges
    anyone judged.
    """
    import json
    from pathlib import Path

    merge: dict[str, set[str]] = {}
    if not proposals_path or not Path(proposals_path).exists():
        return merge
    for line in Path(proposals_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_match") and r.get("status", "ok") == "ok":
            merge.setdefault(r["id_a"], set()).add(r["id_b"])
            merge.setdefault(r["id_b"], set()).add(r["id_a"])
    return merge


def concept_questions(conn, predicates: Optional[Iterable[str]] = None,
                      proposals_path: Optional[str] = None,
                      min_papers: int = MIN_PAPERS,
                      paraphrases: int = 3) -> list[Question]:
    """Tier B (S-58/S-59): concept counting, but only where the answer is
    **invariant to where the concept boundary is drawn**.

    Tier A asks whether the system can read one paper. This asks whether it can
    count across the corpus, which is the actual product claim -- and it is the
    tier the first two generators kept getting wrong, because stating an answer
    requires deciding which entity ids constitute "the thing", and that decision
    IS the deduplication problem.

    So the answer is computed twice and the question is kept only if they agree:

        narrow  just this entity
        broad   this entity
                | everything whose normalised label contains it, or is contained
                  by it, of the same kind                            (spelling)
                | everything the adjudicator approved merging it with (semantic)

    If widening the boundary does not move the number, the boundary does not
    matter and the answer is defensible whatever the system retrieves. If it
    does move, NO number is defensible and the question should never be asked.

    Measured on the real graph: 45% survive on spelling alone, **19% once the
    merge proposals are folded in** -- 109 concepts. The drop is the point: those
    154 concepts were only ever *apparently* safe, and generating questions for
    them would repeat the earlier failure with answers nobody could reach.

    *Why using imprecise proposals here is safe*: a false positive widens `broad`
    and therefore only ever DROPS a question. It can never produce a wrong
    answer, so the filter does not depend on the adjudicator being accurate
    (S-60). The cost is a smaller set, and that set grows as deduplication
    improves -- which makes its size a measure of deduplication quality.
    """
    merge = load_merge_map(proposals_path)
    predicates = list(predicates or PREDICATE_PHRASES)
    out: list[Question] = []

    for predicate in predicates:
        if predicate not in PREDICATE_PHRASES:
            continue
        rows = conn.execute(
            """
            SELECT DISTINCT e.entity_id AS entity_id, e.label AS label, e.kind AS kind
              FROM assertion a JOIN entity e ON e.entity_id = a.object_id
             WHERE a.predicate = ? AND e.label IS NOT NULL AND TRIM(e.label) <> ''
            """,
            (predicate,),
        ).fetchall()

        by_kind: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            by_kind.setdefault(r["kind"] or "", []).append(
                (r["entity_id"], _normalise(r["label"])))

        for r in rows:
            key = _normalise(r["label"])
            if not key:
                continue
            spelling = {i for i, n in by_kind.get(r["kind"] or "", [])
                        if n and (key in n or n in key)}
            semantic = merge.get(r["entity_id"], set())
            broad = {r["entity_id"]} | spelling | semantic

            narrow_papers = _papers_for(conn, predicate, [r["entity_id"]])
            if len(narrow_papers) < min_papers:
                continue
            broad_papers = _papers_for(conn, predicate, sorted(broad))
            if len(narrow_papers) != len(broad_papers):
                continue  # ambiguous: no defensible answer exists

            label = r["label"].strip()
            probe = Probe(family="concept", predicate=predicate,
                          entity_id=r["entity_id"], label=label, kind=r["kind"] or "",
                          papers=narrow_papers, cluster_size=1,
                          surface_forms=len(broad))
            group = f"mm-concept-{predicate}-{r['entity_id']}"
            anchor = _qid("concept", predicate, r["entity_id"], 0)
            provenance = {
                "generated_at": "concept, invariance-tested (Tier B)",
                "predicate": predicate, "entity_id": r["entity_id"],
                # Readable name, not the id. Without it the rewriter is told to
                # preserve "hepkg:systematic:background-normalization" and duly
                # pastes the identifier into the question.
                "label": label, "kind": r["kind"] or "",
                "broad_entities": len(broad),
                "spelling_widened": len(spelling), "semantic_widened": len(semantic),
                "concept_papers": len(narrow_papers),
            }

            for variant in range(min(paraphrases, len(COUNT_PHRASINGS))):
                out.append(Question(
                    qid=_qid("concept", predicate, r["entity_id"], variant),
                    text=_phrase(COUNT_PHRASINGS[variant], probe),
                    source="generated", split="dev", shape="count",
                    needs=probe.needs(), difficulty=probe.difficulty(),
                    truth=Truth(kind="count", value=len(narrow_papers),
                                papers=narrow_papers),
                    truth_source="sql", provenance=provenance, group=group,
                    relation=None if variant == 0 else f"paraphrase_of:{anchor}",
                ))
            out.append(Question(
                qid=_qid("conceptset", predicate, r["entity_id"], 0),
                text=_phrase(SET_PHRASINGS[0], probe),
                source="generated", split="dev", shape="set",
                needs=probe.needs(), difficulty=probe.difficulty(),
                truth=Truth(kind="set", value=len(narrow_papers), papers=narrow_papers),
                truth_source="sql", provenance=provenance, group=group,
                relation=f"set_form_of:{anchor}",
            ))
    return out


def retrieval_questions(conn, predicates: Optional[Iterable[str]] = None,
                        min_papers: int = 1, limit_per_predicate: int = 80) -> list[Question]:
    """Ask about a concept, and check only that the TARGET ENTITY came back.

    The purest retrieval test available, and the one shape ambiguity cannot
    spoil. Tier B has to discard 81% of concepts because their *count* depends on
    where the boundary is drawn -- but "did `hepkg:object:b_jet` appear in what
    the planner retrieved?" has a yes/no answer whatever the boundary is. Those
    discarded concepts are not wasted; they answer a different question.

    Truth is the entity id, never the label: exact, and immune to how the answer
    happens to be worded.

    Deliberately NOT tagged `merged_entities`. Retrieval finding one entity does
    not depend on deduplication having merged anything -- and the stratifiers
    that matter here (how rare the entity is, how many spellings it has) go in
    provenance, where the report can break results down by them.
    """
    predicates = list(predicates or PREDICATE_PHRASES)
    out: list[Question] = []

    for predicate in predicates:
        if predicate not in PREDICATE_PHRASES:
            continue
        rows = conn.execute(
            """
            SELECT e.entity_id AS entity_id, e.label AS label, e.kind AS kind,
                   COUNT(DISTINCT eo.paper_id) AS papers
              FROM assertion a
              JOIN entity e ON e.entity_id = a.object_id
              JOIN entity_occurrence eo
                ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id
             WHERE a.predicate = ? AND e.label IS NOT NULL AND TRIM(e.label) <> ''
             GROUP BY e.entity_id
            HAVING papers >= ?
             ORDER BY papers DESC, e.entity_id
             LIMIT ?
            """,
            (predicate, min_papers, limit_per_predicate),
        ).fetchall()

        for r in rows:
            label = r["label"].strip()
            probe = Probe(family="retrieval", predicate=predicate,
                          entity_id=r["entity_id"], label=label, kind=r["kind"] or "",
                          papers=[""] * r["papers"], cluster_size=1, surface_forms=1)
            spellings = conn.execute(
                "SELECT COUNT(DISTINCT label) FROM entity_occurrence WHERE entity_id = ?",
                (r["entity_id"],),
            ).fetchone()[0]
            out.append(Question(
                qid=_qid("retrieval", predicate, r["entity_id"], 0),
                text=_phrase(SET_PHRASINGS[0], probe),
                source="generated", split="dev", shape="set",
                needs=["sql"],
                # Rarity is the difficulty here, not paper count: one paper and
                # one spelling is where retrieval should fail, and rare entities
                # are what a coverage map is for.
                difficulty="hard" if r["papers"] <= 2 else
                           ("medium" if r["papers"] < 10 else "easy"),
                truth=Truth(kind="entity", value=1, items=[r["entity_id"]]),
                truth_source="sql",
                provenance={"generated_at": "entity retrieval",
                            "predicate": predicate, "entity_id": r["entity_id"],
                            "label": label, "concept_papers": r["papers"],
                            "spellings": spellings},
            ))
    return out
