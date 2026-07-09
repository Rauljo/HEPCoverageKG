"""Graph extraction contracts for public-result papers.

These models describe the first validated layer after PDF-to-Markdown
conversion. They intentionally stay dependency-free so the scraper can emit
validated JSON before any RDF or property-graph backend is chosen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EntityKind = Literal[
    "paper",
    "result",
    "dataset",
    "collision_system",
    "detector_object",
    "object_definition",
    "event_region",
    "selection_requirement",
    "observable",
    "physics_process",
    "background",
    "background_method",
    "result_quantity",
    "systematic_uncertainty",
]

AssertionPredicate = Literal[
    "paper_reports_result",
    "result_uses_dataset",
    "result_has_collision_system",
    "result_targets_process",
    "result_has_final_state",
    "result_defines_object",
    "result_defines_region",
    "region_requires_object",
    "region_vetoes_object",
    "object_has_selection",
    "result_estimates_background",
    "background_uses_method",
    "result_measures_observable",
    "result_reports_quantity",
    "result_has_systematic",
]

AssertionStatus = Literal["accepted", "quarantined", "rejected", "needs_review"]
ExtractionSupport = Literal["explicit", "implicit", "absent", "unclear"]

ENTITY_KINDS = set(EntityKind.__args__)
ASSERTION_PREDICATES = set(AssertionPredicate.__args__)
ASSERTION_STATUSES = set(AssertionStatus.__args__)
EXTRACTION_SUPPORT = set(ExtractionSupport.__args__)


@dataclass
class EvidenceSpan:
    """A paper location supporting an extracted assertion."""

    section: str
    snippet: str
    page: int | None = None
    source_path: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    def validate(self) -> None:
        if not self.section.strip():
            raise ValueError("EvidenceSpan.section is required")
        if not self.snippet.strip():
            raise ValueError("EvidenceSpan.snippet is required")
        if self.page is not None and self.page < 1:
            raise ValueError("EvidenceSpan.page must be positive")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("EvidenceSpan.char_start must be non-negative")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("EvidenceSpan.char_end must be non-negative")
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("EvidenceSpan.char_end must be greater than or equal to char_start")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceSpan:
        return cls(**data)


@dataclass
class EntityRef:
    """A normalized or candidate graph entity."""

    kind: EntityKind
    label: str
    id: str | None = None
    aliases: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise ValueError(f"Unsupported entity kind: {self.kind}")
        if not self.label.strip():
            raise ValueError("EntityRef.label is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityRef:
        return cls(**data)


@dataclass
class CoverageAssertion:
    """A typed assertion ready for validation, review, and graph export."""

    subject: EntityRef
    predicate: AssertionPredicate
    object: EntityRef
    evidence: list[EvidenceSpan]
    confidence: float
    support: ExtractionSupport = "explicit"
    status: AssertionStatus = "accepted"
    qualifiers: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "llm"
    notes: str | None = None

    def validate(self) -> None:
        self.subject.validate()
        self.object.validate()
        if self.predicate not in ASSERTION_PREDICATES:
            raise ValueError(f"Unsupported assertion predicate: {self.predicate}")
        if self.status not in ASSERTION_STATUSES:
            raise ValueError(f"Unsupported assertion status: {self.status}")
        if self.support not in EXTRACTION_SUPPORT:
            raise ValueError(f"Unsupported extraction support: {self.support}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("CoverageAssertion.confidence must be between 0 and 1")
        if not self.extraction_method.strip():
            raise ValueError("CoverageAssertion.extraction_method is required")
        if self.status == "accepted" and not self.evidence:
            raise ValueError("Accepted assertions require evidence")
        for item in self.evidence:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageAssertion:
        payload = dict(data)
        payload["subject"] = EntityRef.from_dict(payload["subject"])
        payload["object"] = EntityRef.from_dict(payload["object"])
        payload["evidence"] = [EvidenceSpan.from_dict(item) for item in payload.get("evidence", [])]
        return cls(**payload)


@dataclass
class PaperExtraction:
    """Validated assertions extracted from one public-result document."""

    cds_id: str
    experiment: Literal["ATLAS", "CMS"]
    title: str
    result_type: str
    assertions: list[CoverageAssertion]
    schema_version: str = "public-results-graph-v0.1"
    doi: str | None = None
    arxiv_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.cds_id.strip():
            raise ValueError("PaperExtraction.cds_id is required")
        if self.experiment not in {"ATLAS", "CMS"}:
            raise ValueError(f"Unsupported experiment: {self.experiment}")
        if not self.title.strip():
            raise ValueError("PaperExtraction.title is required")
        if not self.result_type.strip():
            raise ValueError("PaperExtraction.result_type is required")
        for assertion in self.assertions:
            assertion.validate()

    @property
    def accepted_assertions(self) -> list[CoverageAssertion]:
        return [assertion for assertion in self.assertions if assertion.status == "accepted"]

    @property
    def quarantined_assertions(self) -> list[CoverageAssertion]:
        return [assertion for assertion in self.assertions if assertion.status == "quarantined"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperExtraction:
        payload = dict(data)
        payload["assertions"] = [
            CoverageAssertion.from_dict(item) for item in payload.get("assertions", [])
        ]
        return cls(**payload)
