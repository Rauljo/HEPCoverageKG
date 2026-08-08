"""Tests for backwards question generation (S-33).

The recurring risk here is a generator that produces a large, plausible set of
questions that measure nothing -- unstable ids, a tag that fires on everything,
questions about entities so rare that a lucky system scores the same as a good
one. All of those look fine in a summary table.
"""
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.eval import generate as G
from hepcoveragekg.eval import questions as Q
from hepcoveragekg.query import templates as T


@pytest.fixture()
def conn(tmp_path):
    """Two generators: one merged from two spellings, one standalone. Plus a
    third used by a single paper, which should be filtered out."""
    db = tmp_path / "g.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);
        CREATE TABLE entity_occurrence (bundle_id TEXT, entity_id TEXT, paper_id TEXT,
            kind TEXT, label TEXT);
        CREATE TABLE assertion (assertion_id TEXT PRIMARY KEY, bundle_id TEXT,
            predicate TEXT, subject_id TEXT, object_id TEXT, object_value TEXT);
        CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, quote TEXT, section_title TEXT);
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);

        INSERT INTO entity VALUES
            ('gA','generator','Pythia 8'), ('gA2','generator','PYTHIA8'),
            ('gB','generator','Herwig 7'), ('gC','generator','Rare Gen');
        -- our aliases layer merged gA2 into gA; gB and gC stand alone
        INSERT INTO entity_canonical VALUES ('gA','gA'), ('gA2','gA');

        INSERT INTO entity_occurrence VALUES
            ('b1','s1','p1','sample','ttbar'), ('b2','s2','p2','sample','Zjets'),
            ('b3','s3','p3','sample','Wjets'), ('b4','s4','p4','sample','QCD'),
            ('b1','gA','p1','generator','Pythia 8'),
            ('b2','gA2','p2','generator','PYTHIA8'),
            ('b3','gB','p3','generator','Herwig 7'),
            ('b4','gB','p4','generator','Herwig 7'),
            ('b1','gC','p1','generator','Rare Gen');

        INSERT INTO assertion VALUES
            ('a1','b1','sample_uses_generator','s1','gA',NULL),
            ('a2','b2','sample_uses_generator','s2','gA2',NULL),
            ('a3','b3','sample_uses_generator','s3','gB',NULL),
            ('a4','b4','sample_uses_generator','s4','gB',NULL),
            ('a5','b1','sample_uses_generator','s1','gC',NULL);
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


def test_probes_group_by_concept_not_by_cluster(conn):
    """`Pythia 8` and `PYTHIA8` are one concept. Asking about each separately
    produced questions with cluster-level answers that the system -- which counts
    over a retrieved SET -- could never match: 1 correct out of 36 on
    2026-08-02."""
    found = {p.entity_id: p for p in G.probes(conn, "sample_uses_generator")}
    assert "generator:pythia" in found
    assert found["generator:pythia"].n == 2, "both papers, via the whole concept"


def test_a_concept_spans_separate_canonical_clusters(conn):
    """The bug in one line: a cluster is not a concept. `Pythia 8` and
    `Pythia 8.230` are different clusters and the same concept."""
    found = {p.entity_id: p for p in G.probes(conn, "sample_uses_generator")}
    assert found["generator:pythia"].surface_forms == 2, "gA and gA2"


def test_rare_entities_are_filtered_out(conn):
    """A count of 1 cannot tell a good system from a lucky one."""
    found = {p.entity_id for p in G.probes(conn, "sample_uses_generator")}
    assert "gC" not in found


def test_dedup_sensitivity_is_about_spanning_several_ids(conn):
    """At concept level the answer depends on treating several entity ids as one
    thing -- whether dedup unified them or retrieval breadth did."""
    found = {p.entity_id: p for p in G.probes(conn, "sample_uses_generator")}
    assert found["generator:pythia"].dedup_sensitive, "spans gA and gA2"
    assert not found["generator:herwig"].dedup_sensitive, "one id, several papers"


def test_list_papers_is_not_used_for_truth(conn):
    """`list_papers` and `count` disagree -- `papers_of` returns every paper an
    entity occurs in, untied to the assertion's bundle, so it over-counts
    (Pythia: 58 vs 59 on the real graph). Truth uses `count`'s join."""
    import inspect
    src = inspect.getsource(G._papers_for)
    assert "eo.bundle_id = a.bundle_id" in src
    assert "list_papers" not in inspect.getsource(G.probes)


def test_the_truth_is_the_query_result(conn):
    """The label must come from SQL, never from a model."""
    probe = next(p for p in G.probes(conn, "sample_uses_generator")
                 if p.entity_id == "generator:herwig")
    q = G.questions_from(probe)[0]
    assert q.truth.value == 2
    assert q.truth.papers == ["p3", "p4"]
    assert q.truth_source == "sql"


def test_ids_are_stable_across_regeneration(conn):
    """Ids that move break every cross-run comparison, silently."""
    first = [q.qid for q in G.generate(conn)]
    second = [q.qid for q in G.generate(conn)]
    assert first == second


def test_paraphrases_share_a_group_and_an_answer(conn):
    """S-34: the invariance is only checkable if the group ties them together."""
    probe = next(G.probes(conn, "sample_uses_generator"))
    qs = G.questions_from(probe, paraphrases=3)
    counts = [q for q in qs if q.shape == "count"]
    assert len(counts) == 3
    assert len({q.group for q in counts}) == 1
    assert len({q.truth.value for q in counts}) == 1
    assert sum(1 for q in counts if q.relation and "paraphrase_of" in q.relation) == 2


def test_paraphrases_are_actually_different_wordings(conn):
    probe = next(G.probes(conn, "sample_uses_generator"))
    texts = {q.text for q in G.questions_from(probe, paraphrases=3) if q.shape == "count"}
    assert len(texts) == 3


def test_union_questions_are_unlabelled_on_purpose(conn):
    """Their value is the relation, not a truth: count(A or B) >= max(count A,
    count B) is a bug detector that needs no label (S-34)."""
    ps = list(G.probes(conn, "sample_uses_generator"))
    u = G.union_question(ps[0], ps[1])
    assert u.truth.kind == "none" and u.truth_source == "none"
    assert u.relation.startswith("union_of:")
    assert u.provenance["part_counts"] == [ps[0].n, ps[1].n]


def test_a_union_of_one_thing_with_itself_is_not_a_question(conn):
    ps = list(G.probes(conn, "sample_uses_generator"))
    assert G.union_question(ps[0], ps[0]) is None


def test_generation_needs_no_model(conn):
    """Deterministic and offline, so the whole set can be rebuilt with the
    cluster down -- and so this test can exist at all."""
    qs = G.generate(conn)
    assert qs and all(q.text and "{" not in q.text for q in qs)


def test_every_generated_question_passes_the_loader(conn, tmp_path):
    """The generator and the validator must not drift apart."""
    path = Q.save(G.generate(conn), tmp_path / "gen.jsonl")
    assert len(Q.load(path)) > 0


def test_unphrasable_predicates_are_skipped_not_mangled(conn):
    """`object_has_selection` carries a free-text value, so there is no cluster
    to count; a question about it would be noise."""
    assert not list(G.generate(conn, predicates=["object_has_selection"]))


def test_known_positive_truth_is_a_subset_not_a_set(conn):
    """The point of the shape: ambiguity moves the BOUNDARY of an answer set, it
    never moves a known core member out of it. So the truth is 'these must
    appear', and extras are not counted against the system."""
    qs = G.known_positive_questions(conn, "sample_uses_generator", limit=3)
    assert qs
    for q in qs:
        assert q.truth.kind == "subset"
        assert len(q.truth.papers) == 1
        assert q.truth_source == "sql"


def test_known_positive_questions_do_not_claim_to_need_dedup(conn):
    """They work on the concepts S-58 has to discard -- that is the whole
    advantage, so tagging them `merged_entities` would misattribute it."""
    for q in G.known_positive_questions(conn, "sample_uses_generator", limit=3):
        assert "merged_entities" not in q.needs


def test_known_positive_paraphrases_share_a_group(conn):
    qs = G.known_positive_questions(conn, "sample_uses_generator", limit=1, phrasings=2)
    assert len({q.group for q in qs}) == 1
    assert len({q.text for q in qs}) == 2


def test_paper_questions_need_no_deduplication(conn):
    """A paper id is a hard identifier, so the answer does not move when
    deduplication improves or breaks. Tagging them `merged_entities` would
    misattribute exactly the property that makes Tier A the backbone."""
    for q in G.paper_questions(conn, min_items=1):
        assert "merged_entities" not in q.needs
        assert q.provenance["paper"]


def test_paper_truth_is_deduplicated_within_the_paper(tmp_path):
    """D-046: 22 cases store one thing under two entity ids in the SAME paper.
    Without collapsing them the truth lists a region twice, and a system naming
    it once scores as wrong for being right."""
    db = tmp_path / "d.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (bundle_id TEXT, entity_id TEXT, paper_id TEXT,
            kind TEXT, label TEXT);
        CREATE TABLE assertion (assertion_id TEXT PRIMARY KEY, bundle_id TEXT,
            predicate TEXT, subject_id TEXT, object_id TEXT, object_value TEXT);
        -- one concept, two ids, one paper: "Pythia 8" and "PYTHIA 8"
        INSERT INTO entity VALUES ('g1','generator','Pythia 8'),
                                  ('g2','generator','PYTHIA 8'),
                                  ('g3','generator','Herwig 7');
        INSERT INTO entity_occurrence VALUES ('b1','s1','p1','sample','ttbar');
        INSERT INTO assertion VALUES
            ('a1','b1','sample_uses_generator','s1','g1',NULL),
            ('a2','b1','sample_uses_generator','s1','g2',NULL),
            ('a3','b1','sample_uses_generator','s1','g3',NULL);
        """
    )
    c.commit()
    c.close()

    qs = G.paper_questions(T.read_only(db), min_items=1)
    count = next(q for q in qs if q.shape == "count")
    assert count.truth.value == 2, "the duplicate spelling collapses; 3 rows, 2 things"


def test_paper_questions_carry_readable_labels_for_the_prose_scorer(conn):
    """Truth is entity IDS -- matching prose against a 90-character label would
    measure transcription -- but the scorer still needs the words."""
    qs = G.paper_questions(conn, min_items=1)
    sets = [q for q in qs if q.shape == "set"]
    assert sets and all(q.provenance.get("labels") for q in sets)
    assert all(q.truth.items for q in sets), "ids, not labels, are the truth"


def test_value_carrying_predicates_get_counts_but_no_set_question(conn):
    """`object_has_selection` stores free text, so there is no entity id to match
    a named answer against -- but the COUNT is still exact, and that is 31% of the
    graph nothing else can reach."""
    assert "set" not in G.PAPER_PHRASINGS["object_has_selection"]
    assert "count" in G.PAPER_PHRASINGS["object_has_selection"]
