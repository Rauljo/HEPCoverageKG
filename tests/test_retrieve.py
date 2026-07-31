"""Tests for query-layer retrieval.

Embeddings are skipped throughout (`embed=False`): they need a model download and
they are not what these tests are about. What is tested is the part that is ours
-- tokenisation, the BM25 implementation, rank fusion, and cluster-aware
deduplication -- plus the property the whole module exists for: that a question
about a *concept* returns the whole set, not one node.
"""
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.query import retrieve as R
from hepcoveragekg.query import templates as T


@pytest.fixture()
def conn(tmp_path):
    """Generators that reproduce the real corpus's awkward shapes.

    - genA1/genA2 are merged into one cluster but worded differently
    - genB1/genB2 carry the SAME label and are NOT merged (the real graph has
      7 `MadGraph5_aMC@NLO` entities across 4 clusters)
    - `diboson` carries an alias only visible in entity_occurrence
    """
    db = tmp_path / "r.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT,
            label TEXT, aliases TEXT DEFAULT '[]');
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);

        INSERT INTO entity VALUES
            ('genA1','generator','PYTHIA 8.212'), ('genA2','generator','Pythia v8.212'),
            ('genB1','generator','MadGraph5'),    ('genB2','generator','MadGraph5'),
            ('bkg1','background','diboson'),      ('obj1','detector_object','b-jet');
        INSERT INTO entity_occurrence VALUES
            ('b1','genA1','p1','generator','PYTHIA 8.212','[]'),
            ('b1','genA2','p1','generator','Pythia v8.212','[]'),
            ('b2','genB1','p2','generator','MadGraph5','[]'),
            ('b3','genB2','p3','generator','MadGraph5','[]'),
            ('b1','bkg1','p1','background','diboson','["VV background","di-boson"]'),
            ('b1','obj1','p1','detector_object','b-jet','[]');
        INSERT INTO entity_canonical VALUES ('genA1','genA1'), ('genA2','genA1');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


@pytest.fixture()
def index(conn):
    return R.build(conn, embed=False)


# -- tokenisation ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("b-jet", {"b-jet", "b", "jet"}),
    ("PYTHIA 8.212", {"pythia", "8.212", "8", "212"}),
    ("sqrt(s)=13 TeV", {"sqrt", "s", "13", "tev"}),
])
def test_tokenizer_keeps_hep_surface_forms(text, expected):
    """A generic splitter destroys these. The whole form AND its parts are kept,
    so "b jet" in a question meets "b-jet" in a label."""
    assert expected <= set(R.tokenize(text))


def test_latex_is_stripped_not_indexed():
    toks = R.tokenize(r"$t\bar{t}$ dilepton")
    assert "dilepton" in toks
    assert not any("\\" in t or "$" in t for t in toks)


def test_tokenizer_survives_empty_and_none():
    assert R.tokenize("") == []
    assert R.tokenize(None) == []


# -- indexing --------------------------------------------------------------

def test_indexes_every_wording_not_just_the_merged_label(index):
    """`entity` keeps one modal label per id; alternatives are only in
    entity_occurrence. A question phrased as any paper phrased it must land."""
    assert "PYTHIA 8.212" in index.texts
    assert "Pythia v8.212" in index.texts


def test_aliases_are_indexed(index):
    assert "VV background" in index.texts
    assert "di-boson" in index.texts


# -- retrieval -------------------------------------------------------------

def test_bm25_finds_exact_tokens(index, conn):
    """The half of hybrid retrieval that embeddings are bad at: exact
    identifiers, versions and short codes."""
    hits = R.search(index, "8.212", conn=conn)
    assert hits, "an exact version string must be findable"
    assert any("8.212" in h.label for h in hits)


def test_search_reports_which_retriever_found_a_hit(index, conn):
    """Needed to tell later whether the sparse half is earning its place."""
    hits = R.search(index, "diboson", conn=conn)
    assert hits
    assert hits[0].found_by in ("lexical", "semantic", "both")


def test_merged_entities_collapse_to_one_hit(index, conn):
    """genA1 and genA2 are one thing after deduplication, so they must not
    consume two result slots."""
    hits = R.search(index, "pythia", conn=conn)
    ids = [h.entity_id for h in hits]
    assert len([i for i in ids if i in ("genA1", "genA2")]) == 1


def test_unmerged_lookalikes_stay_separate(index, conn):
    """genB1/genB2 share a label but were never merged. Retrieval must NOT
    silently join them -- that is deduplication's decision, not retrieval's."""
    hits = R.search(index, "MadGraph5", conn=conn)
    ids = {h.entity_id for h in hits}
    assert {"genB1", "genB2"} <= ids


def test_kind_filter(index, conn):
    hits = R.search(index, "diboson", conn=conn, kind="generator")
    assert all(h.kind == "generator" for h in hits)


def test_no_connection_still_dedupes_by_entity(index):
    """Without a connection there is no cluster information, but one entity must
    still not occupy several slots through its different wordings."""
    hits = R.search(index, "pythia", conn=None)
    ids = [h.entity_id for h in hits]
    assert len(ids) == len(set(ids))


def test_empty_query_returns_nothing(index, conn):
    assert R.search(index, "", conn=conn) == []


# -- concept ---------------------------------------------------------------

def test_concept_expands_through_canonical_clusters(index, conn):
    """The point of the module: hand a template the whole set.

    Searching "pythia" collapses genA1/genA2 to one hit, but `concept` must
    hand BOTH ids to the query -- otherwise deduplication would silently narrow
    the answer instead of widening it.
    """
    ids = R.concept(index, "pythia", conn=conn)
    assert {"genA1", "genA2"} <= set(ids)


def test_concept_gathers_unmerged_variants_of_one_idea(index, conn):
    """The 56-Pythia case in miniature: entities deliberately NOT merged, which
    a question about the concept is nonetheless about."""
    ids = R.concept(index, "MadGraph5", conn=conn, kind="generator")
    assert {"genB1", "genB2"} <= set(ids)


def test_concept_returns_ids_a_template_accepts(index, conn):
    ids = R.concept(index, "pythia", conn=conn)
    assert T.expand_canonical(conn, ids)  # does not raise, returns a usable set
