"""
HEPCoverageKG aliases: Phase A - Candidate Generation

Proposes match candidates from two independent signals, lexical (Jaccard over
character trigrams) and semantic (sentence embeddings). This is a BLOCKING
stage: it should over-generate. Precision comes from guards.py (deterministic
vetoes) and adjudicate.py (LLM), and a true synonym missed here can never be
recovered downstream, so the thresholds lean toward recall.

Model choice (measured on HEP probe pairs, not inherited):

    model                          worst synonym  best look-alike   margin
    allenai/scibert (previous)         0.877          0.975         -0.098
    all-MiniLM-L6-v2                   0.748          0.873         -0.125
    BAAI/bge-base-en-v1.5              0.892          0.861         +0.031
    BAAI/bge-large-en-v1.5             0.850          0.882         -0.032
    intfloat/e5-base-v2                0.932          0.929         +0.003

Only bge-base separates the two classes by a usable amount. SciBERT is a
masked-LM checkpoint with no trained similarity head -- sentence-transformers
silently wraps it as Transformer+Pooling, which crushes everything into a
narrow band (unrelated text floors at 0.67, so `signal region` vs `control
region` scored 0.855 against a 0.85 threshold). bge-large is WORSE than
bge-base here despite being 3x the size: it rates `b-tagged jet == b-jet` lower
and `W polarization =/= Z polarization` higher, which is exactly backwards.

Lexical measure: Jaccard is kept deliberately over Levenshtein. Every dangerous
HEP look-alike is a single-character swap (b/c jet, s/t channel, W/Z, 13/14
TeV) -- and so is every true spelling variant (colour/color). Edit distance
scores both classes alike (W/Z polarization: Jaccard 0.882, Levenshtein 0.944),
so it measures the wrong thing. Jaccard over trigrams penalises a changed
character far more on short strings than long ones, which is the behaviour we
want, since the dangerous look-alikes are mostly short.

Known gap: neither signal catches abbreviations. `MET` vs `missing transverse
momentum` scores 0.496 semantically and 0.091 lexically. That needs a
dictionary, not a threshold.
"""
from __future__ import annotations

import itertools
import logging
import os
from typing import Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# Measured separation for bge-base is [0.861 .. 0.892], so a boundary sits near
# 0.876. We deliberately sit BELOW it: this stage blocks rather than decides,
# and an extra candidate only costs one LLM call whereas a missed synonym is
# unrecoverable.
DEFAULT_SEM_THRESHOLD = 0.85
DEFAULT_LEX_THRESHOLD = 0.65

_model: Optional[SentenceTransformer] = None
_model_name: str = ""


def _get_model() -> SentenceTransformer:
    """Load (once) the encoder named by ALIASES_EMBED_MODEL.

    Compute nodes have no outbound network, so the model must already be in the
    HuggingFace cache -- pre-fetch it on the login node.
    """
    global _model, _model_name
    name = os.environ.get("ALIASES_EMBED_MODEL", DEFAULT_MODEL)
    if _model is None or _model_name != name:
        logger.info(f"Loading embedding model: {name}")
        _model = SentenceTransformer(name)
        _model_name = name
    return _model


def reset_model() -> None:
    """Drop the cached encoder so ALIASES_EMBED_MODEL takes effect."""
    global _model, _model_name
    _model, _model_name = None, ""


def _trigrams(s: str) -> Set[str]:
    """Character trigrams, ignoring spaces and case."""
    s = s.lower().replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jaccard_similarity(s1: str, s2: str) -> float:
    """Intersection over union of character trigrams."""
    t1, t2 = _trigrams(s1), _trigrams(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def embed(items: list[str]) -> np.ndarray:
    """L2-normalised embeddings, so a dot product IS the cosine similarity.

    Normalising here is what lets us drop scikit-learn: it was imported for one
    call to cosine_similarity, and was missing from requirements.txt anyway.
    """
    model = _get_model()
    return model.encode(items, show_progress_bar=False, normalize_embeddings=True)


def generate_candidates(
    items: list[str],
    sem_threshold: float | None = None,
    lex_threshold: float | None = None,
) -> list[tuple[str, str]]:
    """Pairs (a, b), a < b, exceeding EITHER the semantic or the lexical threshold.

    Generic over the strings: entity labels, qualifiers, anything. Taking the
    union of the two signals is intentional -- they fail on different things, so
    either one firing is enough to earn an LLM call.

    Thresholds fall back to ALIASES_SEM_THRESHOLD / ALIASES_LEX_THRESHOLD.
    """
    if sem_threshold is None:
        sem_threshold = float(os.environ.get("ALIASES_SEM_THRESHOLD", DEFAULT_SEM_THRESHOLD))
    if lex_threshold is None:
        lex_threshold = float(os.environ.get("ALIASES_LEX_THRESHOLD", DEFAULT_LEX_THRESHOLD))

    if len(items) < 2:
        return []

    # Sort so the returned tuples are ordered and the run is reproducible
    items = sorted(set(items))
    candidates: set[tuple[str, str]] = set()

    # 1. Lexical pass (Jaccard trigrams)
    logger.debug(f"Computing lexical trigram overlaps for {len(items)} items...")
    for a, b in itertools.combinations(items, 2):
        if jaccard_similarity(a, b) >= lex_threshold:
            candidates.add((a, b))

    n_lexical = len(candidates)

    # 2. Semantic pass (embeddings are normalised, so the dot product is cosine)
    logger.debug(f"Computing embeddings for {len(items)} items...")
    embeddings = embed(items)
    sim_matrix = embeddings @ embeddings.T

    rows, cols = np.where(np.triu(sim_matrix, k=1) >= sem_threshold)
    for i, j in zip(rows, cols):
        candidates.add((items[i], items[j]))

    logger.debug(
        f"{len(items)} items -> {len(candidates)} candidates "
        f"(lexical {n_lexical}, semantic added {len(candidates) - n_lexical})"
    )
    return sorted(candidates)
