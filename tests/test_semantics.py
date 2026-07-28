# =============================================================================
# Tier 2 candidate generation.
#
# Two layers of test:
#   - pure logic (Jaccard, thresholds, blocking) runs always, with a stub encoder
#     so the suite stays fast and offline;
#   - the real separation property (synonyms must outrank look-alikes) needs the
#     actual model, so it is opt-in via HEPKG_EMBED_TESTS=1. That test is the one
#     that stops a future model swap from silently regressing the margin.
# =============================================================================
from __future__ import annotations

import os

import numpy as np
import pytest

from hepcoveragekg.aliases import semantics


# --- pure lexical logic ------------------------------------------------------


def test_jaccard_penalises_short_lookalikes_more_than_long_variants():
    """Why Jaccard beats Levenshtein here: a one-character swap wrecks a short
    string's trigrams but barely dents a long one. Every dangerous HEP pair is a
    one-character swap, and so is every true spelling variant -- edit distance
    cannot tell them apart, this can."""
    lookalike = semantics.jaccard_similarity("b-jet", "c-jet")
    variant = semantics.jaccard_similarity(
        "parton shower UE modelling", "parton shower UE modeling"
    )
    assert lookalike < 0.6
    assert variant > 0.8
    assert variant > lookalike


def test_jaccard_is_symmetric_and_bounded():
    assert semantics.jaccard_similarity("abc", "abc") == 1.0
    assert semantics.jaccard_similarity("abcdef", "uvwxyz") == 0.0
    assert semantics.jaccard_similarity("b-jet", "c-jet") == semantics.jaccard_similarity(
        "c-jet", "b-jet"
    )


def test_jaccard_ignores_case_and_spacing():
    assert semantics.jaccard_similarity("Signal Region", "signalregion") == 1.0


def test_jaccard_known_abbreviation_gap():
    """Documented limitation: abbreviations are invisible to both signals and
    need a dictionary, not a threshold."""
    assert semantics.jaccard_similarity("MET", "missing transverse momentum") < 0.2


# --- candidate generation, with a stub encoder -------------------------------


class _StubModel:
    """Encodes to fixed unit vectors so similarity is exact and offline."""

    def __init__(self, table):
        self._table = table

    def encode(self, items, **kwargs):
        vecs = np.array([self._table[i] for i in items], dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def stub_encoder(monkeypatch):
    def install(table):
        monkeypatch.setattr(semantics, "_get_model", lambda: _StubModel(table))
    yield install
    semantics.reset_model()


def test_semantic_pass_applies_the_threshold(stub_encoder):
    # "a" and "b" are near-identical; "c" is orthogonal to both.
    stub_encoder({"a": [1.0, 0.0], "b": [0.99, 0.14], "c": [0.0, 1.0]})
    pairs = semantics.generate_candidates(["a", "b", "c"], sem_threshold=0.9, lex_threshold=1.1)
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs and ("b", "c") not in pairs


def test_lexical_pass_fires_independently_of_the_embedding(stub_encoder):
    """Union, not intersection: either signal alone earns an LLM call."""
    stub_encoder({"pileup modelling": [1.0, 0.0], "pileup modeling": [0.0, 1.0]})
    pairs = semantics.generate_candidates(
        ["pileup modelling", "pileup modeling"], sem_threshold=0.99, lex_threshold=0.6
    )
    assert len(pairs) == 1  # orthogonal embeddings, but lexically near-identical


def test_thresholds_come_from_the_environment(stub_encoder, monkeypatch):
    stub_encoder({"a": [1.0, 0.0], "b": [0.9, 0.44]})
    monkeypatch.setenv("ALIASES_SEM_THRESHOLD", "0.99")
    monkeypatch.setenv("ALIASES_LEX_THRESHOLD", "1.1")
    assert semantics.generate_candidates(["a", "b"]) == []
    monkeypatch.setenv("ALIASES_SEM_THRESHOLD", "0.80")
    assert semantics.generate_candidates(["a", "b"]) == [("a", "b")]


def test_results_are_sorted_deduplicated_and_order_independent(stub_encoder):
    stub_encoder({"a": [1.0, 0.0], "b": [1.0, 0.01], "c": [0.0, 1.0]})
    first = semantics.generate_candidates(["a", "b", "c", "a"], sem_threshold=0.9, lex_threshold=1.1)
    second = semantics.generate_candidates(["c", "b", "a"], sem_threshold=0.9, lex_threshold=1.1)
    assert first == second == sorted(set(first))


def test_fewer_than_two_items_yields_nothing(stub_encoder):
    stub_encoder({"a": [1.0, 0.0]})
    assert semantics.generate_candidates(["a"]) == []
    assert semantics.generate_candidates([]) == []


# --- the real model: separation must hold ------------------------------------

_REASON = "set HEPKG_EMBED_TESTS=1 to run (downloads/loads the real encoder)"


@pytest.mark.skipif(not os.environ.get("HEPKG_EMBED_TESTS"), reason=_REASON)
def test_real_model_ranks_synonyms_above_lookalikes():
    """The property the model was chosen for. Guards already veto number-bearing
    pairs (13/14 TeV, Pythia 8.212/8.230), so those are excluded here -- the
    encoder is only responsible for what survives the deterministic vetoes."""
    synonyms = [
        ("colour reconnection", "color reconnection"),
        ("pileup modelling", "pileup modeling"),
        ("signal region", "signal regions"),
        ("Higgs characterisation", "Higgs characterization"),
        ("b-tagged jet", "b-jet"),
    ]
    lookalikes = [
        ("signal region", "control region"),
        ("W boson polarization", "Z boson polarization"),
        ("s-channel", "t-channel"),
        ("jet energy scale uncertainty", "luminosity uncertainty"),
        ("b-jet", "c-jet"),
    ]

    def score(pairs):
        flat = [t for p in pairs for t in p]
        emb = semantics.embed(flat)
        return [float(emb[2 * i] @ emb[2 * i + 1]) for i in range(len(pairs))]

    worst_synonym = min(score(synonyms))
    best_lookalike = max(score(lookalikes))

    # bge-base measured +0.031. Assert only that the margin is positive, so a
    # deliberate model change can pass while an accidental regression cannot.
    assert worst_synonym > best_lookalike, (
        f"no usable threshold: worst synonym {worst_synonym:.3f} <= "
        f"best look-alike {best_lookalike:.3f}"
    )
    assert semantics.DEFAULT_SEM_THRESHOLD < worst_synonym, (
        "default threshold would drop a true synonym"
    )


@pytest.mark.skipif(not os.environ.get("HEPKG_EMBED_TESTS"), reason=_REASON)
def test_real_model_embeddings_are_normalised():
    emb = semantics.embed(["signal region", "b-jet"])
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
