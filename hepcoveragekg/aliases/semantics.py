"""
HEPCoverageKG aliases: Phase A - Candidate Generation

Pure, generic utilities for proposing match candidates based on lexical 
(Jaccard character trigrams) and semantic (SciBERT) similarities.
"""
from __future__ import annotations

import itertools
import logging
from typing import Iterable, Set

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid overhead on simple script imports
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SciBERT embedding model (allenai/scibert_scivocab_uncased)...")
        _model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    return _model


def _trigrams(s: str) -> Set[str]:
    """Extract character trigrams from a string, ignoring spaces and case."""
    s = s.lower().replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}


def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute Intersection over Union of character trigrams."""
    t1 = _trigrams(s1)
    t2 = _trigrams(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))


def generate_candidates(
    items: list[str], 
    sem_threshold: float = 0.85, 
    lex_threshold: float = 0.65
) -> list[tuple[str, str]]:
    """
    Given a list of strings, return pairs (a, b) where a < b and 
    the pair exceeds either the semantic (SciBERT) or lexical (Jaccard trigram) threshold.
    
    This is highly generic: it doesn't care if `items` are entity labels, qualifiers,
    or random text. It just returns the matching string pairs.
    """
    if len(items) < 2:
        return []

    # Sort items so our returned tuples are consistently ordered
    items = sorted(list(set(items)))
    candidates: set[tuple[str, str]] = set()

    # 1. Lexical pass (Jaccard Trigrams)
    logger.debug(f"Computing lexical trigram overlaps for {len(items)} items...")
    for a, b in itertools.combinations(items, 2):
        if jaccard_similarity(a, b) >= lex_threshold:
            candidates.add((a, b))

    # 2. Semantic pass (SciBERT)
    logger.debug(f"Computing SciBERT embeddings for {len(items)} items...")
    model = _get_model()
    embeddings = model.encode(items)
    
    sim_matrix = cosine_similarity(embeddings)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if sim_matrix[i][j] >= sem_threshold:
                candidates.add((items[i], items[j]))

    # Convert to sorted list of tuples
    return sorted(list(candidates))
