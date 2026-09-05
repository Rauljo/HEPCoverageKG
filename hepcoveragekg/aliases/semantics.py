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
import time
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


def _model_name_from_env() -> str:
    """The configured encoder, treating EMPTY as UNSET.

    `os.environ.get(var, DEFAULT)` returns "" when the variable exists but is
    blank, so a default only applies to an ABSENT variable. Slurm makes that
    easy to hit: `--export=ALL,ALIASES_EMBED_MODEL=` sets it to empty for every
    arm that wanted the default, and `SentenceTransformer("")` loads an object
    whose first module is None -- the failure surfaces much later as
    `'NoneType' object has no attribute 'tokenize'`, which names neither the
    variable nor the model. Four dev-200 jobs died this way in 9 seconds each
    on 2026-09-02.
    """
    return (os.environ.get("ALIASES_EMBED_MODEL") or "").strip() or DEFAULT_MODEL


def _get_model() -> SentenceTransformer:
    """Load (once) the encoder named by ALIASES_EMBED_MODEL.

    Compute nodes have no outbound network, so the model must already be in the
    HuggingFace cache -- pre-fetch it on the login node.
    """
    global _model, _model_name
    name = _model_name_from_env()
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


def embed_cached(items: list[str], cache: "Path | str | None" = None) -> np.ndarray:
    """`embed`, reusing anything already encoded by a previous run.

    Lives here rather than in the query layer because BOTH layers encode the
    same strings -- the aliases layer to generate merge candidates, retrieval to
    build its index -- and at 3,000 papers that is roughly 420,000 encodings
    done twice for no reason.

    Keyed on **each string**, not on the collection. Keying on the whole corpus
    means importing one paper invalidates everything and re-encodes all of it,
    which is the opposite of what a growing corpus needs. A given string always
    encodes to the same vector for a fixed model, so old vectors are reused
    verbatim.

    The model name is stored with the vectors and checked on load: switching
    ALIASES_EMBED_MODEL must not silently reuse another model's embeddings,
    which would be wrong in a way nothing downstream could detect.
    """
    from pathlib import Path as _Path

    _, model_name = _get_model(), _model_name_from_env()
    known: dict[str, np.ndarray] = {}

    if cache:
        cache = _Path(cache)
        if cache.exists():
            blob = np.load(cache, allow_pickle=False)
            cached_model = str(blob["model"]) if "model" in blob else ""
            if cached_model != model_name:
                logger.info(
                    f"embedding cache was built with '{cached_model or 'unknown'}', "
                    f"now using '{model_name}' -- discarding"
                )
            elif "texts" in blob and "vectors" in blob:
                known = dict(zip((str(t) for t in blob["texts"]), blob["vectors"]))
                logger.info(f"embedding cache: {len(known)} known strings")

    missing = [t for t in dict.fromkeys(items) if t not in known]
    if missing:
        logger.info(f"encoding {len(missing)} new strings "
                    f"({len(items) - len(missing)} reused)")
        for text, vector in zip(missing, embed(missing)):
            known[text] = vector

    if cache and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pairs = list(known.items())
        # ATOMIC: write a unique temp file in the same directory, then rename.
        # `np.savez` straight to `cache` is a multi-second non-atomic write, and
        # concurrent Slurm arms share this path -- a reader arriving mid-write
        # gets a truncated npz and dies hours in. os.replace is atomic within a
        # filesystem, so a reader sees either the old file or the new one.
        # PID+time in the temp name so two writers cannot collide on it either.
        tmp = cache.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp.npz")
        try:
            np.savez(
                tmp,
                texts=np.array([t for t, _ in pairs], dtype=object).astype("U"),
                vectors=np.stack([v for _, v in pairs]),
                model=np.array(model_name),
            )
            os.replace(tmp, cache)
        except Exception:  # noqa: BLE001 -- a cache is an optimisation, never
            # a reason to lose the run. The vectors are already in `known`.
            logger.warning("could not write embedding cache %s", cache, exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    if not items:
        return np.zeros((0, 768), dtype=np.float32)
    return np.stack([known[t] for t in items])


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
