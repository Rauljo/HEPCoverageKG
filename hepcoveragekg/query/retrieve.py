"""
HEPCoverageKG query layer: retrieval -- words to entity ids.

The only entry point that takes text rather than ids, so nothing else can start
a chain without it. Its job is not "find the best match" but **find the set the
question is about**: asking about one Pythia entity returns 1 paper, while the
56 that exist together return 58 (see templates.expand_canonical).

Three decisions, each made because the obvious alternative was measurably worse
on this corpus.

  index every wording, not the merged label.  `entity` keeps one modal label per
      id, which silently drops alternatives -- `hepkg:background:diboson` has
      six. Retrieval reads `entity_occurrence` and indexes all 5,721 distinct
      (entity_id, label) pairs plus 3,109 alias strings, so a question phrased
      the way *any* paper phrased it still lands.

  hybrid, fused by rank rather than by score.  Dense embeddings miss exact
      tokens ("SRA", "CR3l-VV", arXiv ids); BM25 misses paraphrase. Their scores
      are not comparable -- cosine sits in a narrow band while BM25 is unbounded
      -- so combining them by weighted sum needs a normalisation that shifts
      with the corpus. Reciprocal-rank fusion combines the *orderings*, which
      needs no calibration and cannot be broken by one retriever's scale.

  results are grouped by canonical cluster.  Plain top-k returns five spellings
      of one thing and calls it five results, which both wastes the budget and
      misleads a planner into thinking it found five things.

BM25 is implemented here rather than taken from `rank_bm25` because the
tokeniser is the part that matters and it is domain-specific: a generic splitter
destroys `b-jet`, `13 TeV`, `PYTHIA 8.212` and `sqrt(s)=13`.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Standard BM25 constants. k1 controls how fast term frequency saturates, b how
# strongly length is normalised. Untuned -- there is no labelled retrieval set to
# tune against yet, and inventing one would be worse than the defaults.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal-rank-fusion constant. 60 is the value from the original TREC work
# and is deliberately large: it flattens the head so that a confident-but-wrong
# top hit from one retriever cannot dominate the other's ranking.
RRF_K = 60

DEFAULT_LIMIT = 50

# LaTeX wrappers and maths punctuation carry no retrieval signal but do split
# tokens badly, so they go before tokenisation rather than being indexed.
_LATEX = re.compile(r"\\[a-zA-Z]+|[${}\\]")
_SPLIT = re.compile(r"[^0-9a-zÀ-ɏͰ-Ͽ.\-+]+")


def tokenize(text: str) -> list[str]:
    """Lowercased tokens, with HEP surface forms kept intact.

    Hyphenated and dotted terms are emitted BOTH whole and split: `b-jet` yields
    `b-jet`, `b`, `jet`, and `pythia 8.212` yields `8.212` and `8`. A question
    saying "b jet" and a label saying "b-jet" then meet on the parts, while a
    question naming the exact version still matches on the whole.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _LATEX.sub(" ", text).lower()
    out: list[str] = []
    for raw in _SPLIT.split(text):
        tok = raw.strip(".-+")
        if not tok:
            continue
        out.append(tok)
        if "-" in tok or "." in tok:
            out.extend(p for p in re.split(r"[.\-]", tok) if p)
    return out


@dataclass
class Hit:
    entity_id: str
    label: str
    kind: str
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    matched_on: str = ""  # the surface form that actually matched
    # Closed-vocabulary tags this entity carries, if the facet layer is derived.
    # Attached here rather than shipped as a vocabulary list in the prompt: a
    # hit already IS the answer to "which facet key covers this concept", it is
    # grounded in an entity that demonstrably exists, and it costs nothing as
    # the vocabulary grows. Measured on the phrasings the supervisor's Tier 1
    # questions use, the right key is on the rank-1 hit 8 times out of 8 with
    # BM25 alone (D-054).
    facets: list[str] = field(default_factory=list)

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        return "semantic" if self.dense_rank is not None else "lexical"


@dataclass
class Index:
    """Surface forms and their entity ids. One entity may hold many rows."""

    entity_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    _tokens: list[list[str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self.texts)

    # -- BM25 ------------------------------------------------------------
    def _prepare_sparse(self) -> None:
        self._df = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        self._avg_len = (sum(len(t) for t in self._tokens) / len(self._tokens)) if self._tokens else 0.0

    def bm25(self, query: str) -> np.ndarray:
        """Score every surface form against the query."""
        q = tokenize(query)
        n = len(self._tokens)
        scores = np.zeros(n, dtype=np.float32)
        if not n or not q:
            return scores
        for term in set(q):
            df = self._df.get(term, 0)
            if not df:
                continue
            # +0.5/+0.5 smoothing keeps the idf of a term appearing in most
            # documents at a small positive value rather than letting it go
            # negative and actively penalise a match.
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self._tokens):
                tf = toks.count(term)
                if not tf:
                    continue
                norm = 1 - BM25_B + BM25_B * (len(toks) / self._avg_len or 1.0)
                scores[i] += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm)
        return scores


def _surface_forms(conn) -> Iterable[tuple[str, str, str]]:
    """(entity_id, kind, surface form) for everything worth indexing.

    Reads `entity_occurrence` so that every wording any paper used is reachable,
    not only the modal label kept on `entity`.
    """
    seen: set[tuple[str, str]] = set()
    for r in conn.execute(
        "SELECT entity_id, kind, label, aliases FROM entity_occurrence"
    ):
        forms = [r["label"]]
        try:
            forms.extend(json.loads(r["aliases"] or "[]"))
        except (TypeError, ValueError):
            pass
        for form in forms:
            if not form or not str(form).strip():
                continue
            key = (r["entity_id"], str(form))
            if key in seen:
                continue
            seen.add(key)
            yield r["entity_id"], r["kind"] or "", str(form)


def build(conn, cache: Path | str | None = None, embed: bool = True) -> Index:
    """Index every surface form in the graph.

    Embeddings are cached to `cache` because encoding ~8,800 short strings costs
    real CPU and the graph only changes on reimport. The cache is keyed on the
    surface forms themselves, so a stale cache is detected rather than trusted.
    """
    index = Index()
    for entity_id, kind, form in _surface_forms(conn):
        index.entity_ids.append(entity_id)
        index.kinds.append(kind)
        index.texts.append(form)
    index._tokens = [tokenize(t) for t in index.texts]
    index._prepare_sparse()
    logger.info(f"indexed {len(index)} surface forms for "
                f"{len(set(index.entity_ids))} entities")

    if embed:
        index.embeddings = _embeddings(index.texts, cache)
    return index


def _embeddings(texts: list[str], cache: Path | str | None) -> np.ndarray:
    """Delegates to the aliases layer's cached encoder.

    Shared deliberately: the aliases layer encodes the same entity labels to
    generate merge candidates, so a separate cache here would do ~420,000
    encodings twice at full corpus size. Same model (D-035), same cache file.
    """
    from hepcoveragekg.aliases import semantics

    return semantics.embed_cached(texts, cache)


def _ranks(scores: np.ndarray, top: int) -> dict[int, int]:
    """{row index: 1-based rank} for the highest-scoring rows above zero."""
    order = np.argsort(-scores)[:top]
    return {int(i): rank for rank, i in enumerate(order, start=1) if scores[i] > 0}


def search(
    index: Index,
    query: str,
    conn=None,
    kind: str | None = None,
    limit: int = DEFAULT_LIMIT,
    pool: int = 200,
) -> list[Hit]:
    """The entity ids a question is about, best first.

    `pool` is how deep each retriever is considered before fusion -- it must be
    comfortably larger than `limit`, since the point of fusing is to promote
    things one retriever ranked poorly.

    Pass `conn` to group results by canonical cluster, so five spellings of one
    entity return as one hit rather than consuming five slots.
    """
    sparse = index.bm25(query)
    sparse_ranks = _ranks(sparse, pool)

    dense_ranks: dict[int, int] = {}
    if index.embeddings is not None and len(index.embeddings):
        from hepcoveragekg.aliases import semantics

        qv = semantics.embed([query])[0]
        sims = index.embeddings @ qv  # both normalised, so this is cosine
        dense_ranks = _ranks(sims, pool)

    # Reciprocal-rank fusion: rank-based, so the two retrievers' incomparable
    # score scales never need calibrating.
    fused: dict[int, float] = defaultdict(float)
    for i, rank in dense_ranks.items():
        fused[i] += 1.0 / (RRF_K + rank)
    for i, rank in sparse_ranks.items():
        fused[i] += 1.0 / (RRF_K + rank)

    hits: list[Hit] = []
    for i, score in sorted(fused.items(), key=lambda kv: -kv[1]):
        if kind and index.kinds[i] != kind:
            continue
        hits.append(Hit(
            entity_id=index.entity_ids[i],
            label=index.texts[i],
            kind=index.kinds[i],
            score=score,
            dense_rank=dense_ranks.get(i),
            sparse_rank=sparse_ranks.get(i),
            matched_on=index.texts[i],
        ))

    hits = _dedupe(hits, conn)
    hits = hits[:limit]
    _attach_facets(hits, conn)
    return hits


def _attach_facets(hits: list[Hit], conn) -> None:
    """Hang each hit's closed-vocabulary tags on it, in one query.

    After the slice, not before: tagging 200 pooled candidates to show 60 would
    be five sixths wasted work.

    Silent when the facet layer has not been derived. The query layer has to
    keep working against a database built before the facets tables existed --
    the alternative is that adding a derived layer breaks every older graph.
    """
    if conn is None or not hits:
        return
    ids = sorted({h.entity_id for h in hits})
    try:
        rows = conn.execute(
            "SELECT DISTINCT entity_id, value FROM entity_facet"
            f" WHERE entity_id IN ({','.join('?' * len(ids))})"
            " ORDER BY entity_id, value",
            ids,
        ).fetchall()
    except Exception:
        return
    by_entity: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_entity[row["entity_id"]].append(row["value"])
    for hit in hits:
        hit.facets = by_entity.get(hit.entity_id, [])


def _dedupe(hits: list[Hit], conn) -> list[Hit]:
    """Keep the best hit per entity, and per canonical cluster when `conn` is given.

    Without this, `search("Pythia")` spends its budget on spellings rather than
    on distinct entities -- and a planner reading the result would believe it had
    found more things than it had.
    """
    best_per_entity: dict[str, Hit] = {}
    for h in hits:
        if h.entity_id not in best_per_entity:
            best_per_entity[h.entity_id] = h
    ordered = sorted(best_per_entity.values(), key=lambda h: -h.score)
    if conn is None:
        return ordered

    from hepcoveragekg.query.templates import canonical_of

    seen: set[str] = set()
    out: list[Hit] = []
    for h in ordered:
        cluster = canonical_of(conn, h.entity_id)
        if cluster in seen:
            continue
        seen.add(cluster)
        out.append(h)
    return out


def concept(index: Index, query: str, conn=None, kind: str | None = None,
            limit: int = DEFAULT_LIMIT) -> list[str]:
    """Entity ids for a *concept*, ready to hand straight to a template.

    The counterpart to `search` for the common case: the planner does not want
    ranked hits, it wants the set. Expanded through canonical clusters, so what
    deduplication merged comes along automatically.
    """
    hits = search(index, query, conn=conn, kind=kind, limit=limit)
    ids = [h.entity_id for h in hits]
    if conn is None or not ids:
        return ids

    from hepcoveragekg.query.templates import expand_canonical

    return expand_canonical(conn, ids)
