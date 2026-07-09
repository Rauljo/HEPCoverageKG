# =============================================================================
# HEPCoverageKG: per-paper extraction state
#
# CatalogState + HybridRetriever ported from DeepCollector's core/state.py,
# rescoped from "one project, many datasets, many flat fields" to "one paper,
# one Result entity, many predicate assertions" (see PREDICATE_SCHEMA in
# config/schema.py). self.catalog keeps its name for continuity with the
# ported design, but now holds a flat list[CoverageAssertion] instead of
# DeepCollector's list[CatalogItem].
#
# EvidenceSpan/EntityRef/CoverageAssertion/PaperExtraction dataclasses ported
# from the supervisor's graph_schema.py (kept untouched at repo root).
# PaperExtraction's primary id is flipped to arxiv_id (required), with cds_id
# optional - our harvest chain is InspireHEP -> arXiv -> HEPData, not CDS.
# =============================================================================
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from hepcoveragekg.config.schema import (
    ALLOWED_EXPERIMENTS,
    ASSERTION_PREDICATES,
    ASSERTION_STATUSES,
    ENTITY_KINDS,
    EXTRACTED_PREDICATES,
    EXTRACTION_METHODS,
    EXTRACTION_SUPPORT,
    MISSING_DATA_PLACEHOLDERS,
    SCHEMA_VERSION,
)
from hepcoveragekg.harvesting.html_harvester import HarvestedPaper

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
except ImportError:
    Document = VectorStoreIndex = SentenceSplitter = None
    VectorIndexRetriever = BM25Retriever = None
    BaseRetriever = object


# --- Typed assertion contracts (ported from graph_schema.py) -----------------


@dataclass
class EvidenceSpan:
    """A paper location supporting an extracted assertion."""

    section: str
    snippet: str
    source_url: Optional[str] = None

    def validate(self) -> None:
        if not self.section.strip():
            raise ValueError("EvidenceSpan.section is required")
        if not self.snippet.strip():
            raise ValueError("EvidenceSpan.snippet is required")


@dataclass
class EntityRef:
    """A normalized or candidate graph entity."""

    kind: str
    label: str
    id: Optional[str] = None
    aliases: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise ValueError(f"Unsupported entity kind: {self.kind}")
        if not self.label.strip():
            raise ValueError("EntityRef.label is required")


@dataclass
class CoverageAssertion:
    """A typed (subject, predicate, object) triple ready for graph export."""

    subject: EntityRef
    predicate: str
    object: EntityRef
    evidence: list[EvidenceSpan]
    confidence: float
    support: str = "explicit"
    status: str = "accepted"
    qualifiers: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "llm"
    notes: Optional[str] = None

    def validate(self) -> None:
        self.subject.validate()
        self.object.validate()
        if self.predicate not in ASSERTION_PREDICATES:
            raise ValueError(f"Unsupported assertion predicate: {self.predicate}")
        if self.status not in ASSERTION_STATUSES:
            raise ValueError(f"Unsupported assertion status: {self.status}")
        if self.support not in EXTRACTION_SUPPORT:
            raise ValueError(f"Unsupported extraction support: {self.support}")
        if self.extraction_method not in EXTRACTION_METHODS:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")
        # Callers (rag_engine.py) are responsible for normalizing a raw LLM
        # confidence value (e.g. "85" or "8.5") into [0,1] before constructing
        # an assertion - this is a strict backstop, not a normalizer.
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("CoverageAssertion.confidence must be between 0 and 1")
        if self.status == "accepted" and not self.evidence:
            raise ValueError("Accepted assertions require evidence")
        for item in self.evidence:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperExtraction:
    """Validated assertions accumulated from one harvested paper."""

    arxiv_id: str
    experiment: str
    title: str
    result_type: str
    assertions: list[CoverageAssertion]
    schema_version: str = SCHEMA_VERSION
    cds_id: Optional[str] = None
    doi: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.arxiv_id.strip():
            raise ValueError("PaperExtraction.arxiv_id is required")
        if self.experiment not in ALLOWED_EXPERIMENTS:
            raise ValueError(f"Unsupported experiment: {self.experiment}")
        if not self.title.strip():
            raise ValueError("PaperExtraction.title is required")
        for assertion in self.assertions:
            assertion.validate()

    @property
    def accepted_assertions(self) -> list[CoverageAssertion]:
        return [a for a in self.assertions if a.status == "accepted"]

    @property
    def quarantined_assertions(self) -> list[CoverageAssertion]:
        return [a for a in self.assertions if a.status == "quarantined"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Hybrid (vector + keyword) retrieval, ported from core/state.py ---------

if VectorIndexRetriever is not None and BM25Retriever is not None:

    class HybridRetriever(BaseRetriever):
        """Combines vector (embedding) and BM25 (keyword) retrieval, deduped by node id."""

        def __init__(self, vector_retriever: Any, bm25_retriever: Any):
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            super().__init__()

        def _retrieve(self, query: Any, **kwargs: Any) -> list[Any]:
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
            seen: set[str] = set()
            merged = []
            for node in vector_nodes + bm25_nodes:
                if node.node.node_id not in seen:
                    merged.append(node)
                    seen.add(node.node.node_id)
            return merged[: self.vector_retriever.similarity_top_k]

else:
    HybridRetriever = None


# --- Per-paper session: retrieval index + accumulating assertions -----------


class CatalogState:
    """One paper's extraction session: a retrieval index over its text, plus
    the CoverageAssertions accumulated so far as PREDICATE_SCHEMA gets asked."""

    def __init__(self, config: Any, paper: HarvestedPaper):
        self.config = config
        self.paper = paper
        self.verbosity = getattr(config, "VERBOSITY_LEVEL", 1)

        self.catalog: list[CoverageAssertion] = []
        self.history: list[str] = []
        self.iteration: int = 0

        self.index: Optional[Any] = None
        self.documents: list[Any] = []
        self.all_nodes: list[Any] = []
        self.bm25_retriever: Optional[Any] = None
        self._indexed_section_ids: set[str] = set()

        self.MISSING_DATA_PLACEHOLDERS = MISSING_DATA_PLACEHOLDERS
        self.CONFIDENCE_LOCK_THRESHOLD = getattr(config, "CONFIDENCE_LOCK_THRESHOLD", 0.95)

        if VectorStoreIndex is not None:
            try:
                self.index = VectorStoreIndex.from_documents([])
            except Exception as error:
                if self.verbosity >= 2:
                    print(f"[CatalogState] index init failed: {error}")
                self.index = None

        self._index_paper_sections()

    def add_history(self, message: str) -> None:
        self.history.append(f"[Iter {self.iteration}] {message}")

    def _index_paper_sections(self) -> None:
        """Chunks the paper's title/abstract/sections into the retrieval index."""
        if self.index is None or Document is None:
            return

        sections = [("abstract", f"{self.paper.title}\n\n{self.paper.abstract}")]
        sections += [(s.heading, s.text) for s in self.paper.sections]

        new_documents = []
        for section_id, (heading, text) in enumerate(sections):
            key = str(section_id)
            if not text or key in self._indexed_section_ids:
                continue
            doc = Document(
                text=f"Section: {heading}\n{text}",
                metadata={"heading": heading, "source_url": self.paper.source_url},
                id_=key,
            )
            new_documents.append(doc)
            self._indexed_section_ids.add(key)

        if not new_documents:
            return
        self.documents.extend(new_documents)

        try:
            node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
            new_nodes = node_parser.get_nodes_from_documents(new_documents)
            self.all_nodes.extend(new_nodes)
            self.index.insert_nodes(new_nodes)
            if self.verbosity >= 1:
                print(f"[CatalogState] indexed {len(new_documents)} sections ({len(new_nodes)} chunks).")
        except Exception as error:
            if self.verbosity >= 2:
                print(f"[CatalogState] vector index error: {error}")

        if BM25Retriever is not None and self.all_nodes:
            try:
                self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.all_nodes, similarity_top_k=8)
            except Exception as error:
                if self.verbosity >= 2:
                    print(f"[CatalogState] bm25 index error: {error}")

    def get_retriever(self, similarity_top_k: int = 8, mode: str = "HYBRID") -> Any:
        if self.index is None or VectorIndexRetriever is None:
            return None

        actual_k = min(similarity_top_k, len(self.all_nodes)) or 1
        vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=actual_k)
        if mode == "VECTOR" or self.bm25_retriever is None:
            return vector_retriever

        self.bm25_retriever.similarity_top_k = actual_k
        if mode == "BM25":
            return self.bm25_retriever
        if HybridRetriever is not None:
            return HybridRetriever(vector_retriever, self.bm25_retriever)
        return vector_retriever

    # --- Assertion accumulation (confidence-beats rule, ported from
    # DeepCollector's rag_engine.py::_process_rag_results) -------------------

    def get_assertion(self, predicate: str, object_label: Optional[str] = None) -> Optional[CoverageAssertion]:
        for assertion in self.catalog:
            if assertion.predicate != predicate:
                continue
            if object_label is None or assertion.object.label.strip().lower() == object_label.strip().lower():
                return assertion
        return None

    def update_assertion(self, new_assertion: CoverageAssertion, multi_value: bool = False) -> bool:
        """Fill if missing; else confirm-if-higher-confidence when the new
        value agrees, or refine-if->=confidence when it disagrees.

        multi_value=True (e.g. a result can target several physics processes)
        looks up the existing assertion by (predicate, object label), so
        distinct objects coexist instead of competing. multi_value=False (e.g.
        collision energy) looks up by predicate alone, since two different
        answers ARE a conflict to arbitrate, not two facts to keep.
        """
        lookup_label = new_assertion.object.label if multi_value else None
        current = self.get_assertion(new_assertion.predicate, lookup_label)
        if current is None:
            self.catalog.append(new_assertion)
            return True

        if current.confidence >= self.CONFIDENCE_LOCK_THRESHOLD:
            return False

        old_val = current.object.label.strip().lower()
        new_val = new_assertion.object.label.strip().lower()
        is_fill = old_val in self.MISSING_DATA_PLACEHOLDERS and new_val not in self.MISSING_DATA_PLACEHOLDERS
        is_same = old_val == new_val

        should_update = (
            is_fill
            or (is_same and new_assertion.confidence > current.confidence)
            or (not is_same and not is_fill and new_assertion.confidence >= current.confidence)
        )
        if should_update:
            self.catalog.remove(current)
            self.catalog.append(new_assertion)
        return should_update

    def capture_confidence_metrics(self) -> dict[str, float]:
        """Completeness/avg-confidence over the predicates PREDICATE_SCHEMA defines."""
        if not EXTRACTED_PREDICATES:
            return {"avg_confidence": 0.0, "completeness": 0.0, "count": 0}

        total_confidence = 0.0
        filled = 0
        for predicate in EXTRACTED_PREDICATES:
            assertion = self.get_assertion(predicate)
            if assertion is not None and assertion.status != "rejected":
                total_confidence += assertion.confidence
                filled += 1
        total = len(EXTRACTED_PREDICATES)
        return {
            "avg_confidence": total_confidence / filled if filled else 0.0,
            "completeness": filled / total,
            "count": len(self.catalog),
        }
