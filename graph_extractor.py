"""First-pass detector-object graph extraction from converted Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph_schema import CoverageAssertion, EntityRef, EvidenceSpan, PaperExtraction
from .io import read_json, write_json
from .markdown_normalizer import preferred_markdown_path
from .models import PaperRecord
from .storage import CatalogStore

GRAPH_SCHEMA_VERSION = "public-results-graph-v0.1"


@dataclass(frozen=True)
class TermPattern:
    id: str
    label: str
    pattern: re.Pattern[str]
    confidence: float = 0.72


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


DETECTOR_OBJECTS = [
    TermPattern("electron", "electron", _compile(r"\belectrons?\b|\be[+-]\b")),
    TermPattern("muon", "muon", _compile(r"\bmuons?\b|\\mu")),
    TermPattern("photon", "photon", _compile(r"\bphotons?\b|\\gamma")),
    TermPattern("small_r_jet", "small-R jet", _compile(r"\bjets?\b|anti-?k[_{ -]?t")),
    TermPattern("large_r_jet", "large-R jet", _compile(r"\blarge[- ]R jets?\b|R\s*=\s*1\.0")),
    TermPattern("b_tagged_jet", "b-tagged jet", _compile(r"\bb[- ]?tag(?:ged|ging)?\b|\bb[- ]?jets?\b")),
    TermPattern("hadronic_tau", "hadronic tau", _compile(r"\bhadronic tau\b|\btau leptons?\b|\\tau")),
    TermPattern(
        "missing_transverse_momentum",
        "missing transverse momentum",
        _compile(r"\bmissing transverse (?:momentum|energy)\b|\bE(?:T|_T)\^?miss\b|\\met"),
    ),
    TermPattern("track", "track", _compile(r"\btracks?\b")),
    TermPattern("vertex", "vertex", _compile(r"\bvertices\b|\bvertex\b")),
]

PHYSICS_PROCESSES = [
    TermPattern("higgs_boson", "Higgs boson", _compile(r"\bHiggs\b")),
    TermPattern("top_quark", "top quark", _compile(r"\btop quarks?\b|\bt\bar\{?t\}?\b|\bttbar\b")),
    TermPattern("w_boson", "W boson", _compile(r"\bW bosons?\b|\bW[+-]\b")),
    TermPattern("z_boson", "Z boson", _compile(r"\bZ bosons?\b|\bZ/gamma\b")),
    TermPattern("diboson", "diboson", _compile(r"\bdiboson\b|\bWW\b|\bWZ\b|\bZZ\b")),
    TermPattern("supersymmetry", "supersymmetry", _compile(r"\bsupersymmetry\b|\bSUSY\b")),
    TermPattern("dark_matter", "dark matter", _compile(r"\bdark matter\b")),
    TermPattern("heavy_ion", "heavy-ion", _compile(r"\bheavy[- ]ion\b|\bPb\+?Pb\b|\bpPb\b")),
    TermPattern("standard_model", "Standard Model", _compile(r"\bStandard Model\b|\bSM\b")),
]

OBSERVABLES = [
    TermPattern("cross_section", "cross section", _compile(r"\bcross[- ]sections?\b")),
    TermPattern("differential_cross_section", "differential cross section", _compile(r"\bdifferential cross[- ]sections?\b")),
    TermPattern("mass", "mass", _compile(r"\binvariant mass\b|\bmass distribution\b|\bmass spectrum\b")),
    TermPattern("transverse_momentum", "transverse momentum", _compile(r"\btransverse momentum\b|\bp[_{ ]?T\b")),
    TermPattern("limit", "limit", _compile(r"\bupper limits?\b|\bexclusion limits?\b|\bexcluded\b")),
    TermPattern("branching_fraction", "branching fraction", _compile(r"\bbranching (?:fraction|ratio)\b")),
]

BACKGROUNDS = [
    TermPattern("multijet_background", "multijet background", _compile(r"\bmultijet background\b|\bQCD background\b")),
    TermPattern("fake_background", "fake/nonprompt background", _compile(r"\bfake\b|\bnon[- ]prompt\b|\bmisidentified\b")),
    TermPattern("top_background", "top background", _compile(r"\btop(?:-quark)? background\b|\bt\bar\{?t\}? background\b")),
    TermPattern("zjets_background", "Z+jets background", _compile(r"\bZ\+jets\b|\bZ boson background\b")),
    TermPattern("wjets_background", "W+jets background", _compile(r"\bW\+jets\b|\bW boson background\b")),
    TermPattern("diboson_background", "diboson background", _compile(r"\bdiboson background\b")),
]

SECTION_HEADINGS = _compile(r"^(#{1,6})\s+(.+?)\s*$")
SELECTION_SECTION = _compile(
    r"selection|object|reconstruction|fiducial|signal region|control region|validation region|event"
)
BACKGROUND_SECTION = _compile(r"background|estimation|control region")
RESULT_SECTION = _compile(r"result|measurement|limit|cross[- ]section|interpretation")
ENERGY = _compile(r"(?:sqrt\{?s\}?|center-of-mass energy|centre-of-mass energy)[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*TeV")
LUMINOSITY = _compile(r"(?:integrated luminosity|luminosity)[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:fb|pb)\s*\^?[-\u2212]?1")
REGION = _compile(r"\b(signal|control|validation) regions?\b|\b(SR|CR|VR)[0-9A-Za-z_-]*\b")


@dataclass(frozen=True)
class Section:
    title: str
    text: str


def extraction_path(store: CatalogStore, paper: PaperRecord, schema_version: str = GRAPH_SCHEMA_VERSION) -> Path:
    return store.record_dir(paper) / "graph" / schema_version / "extraction.json"


def extract_for_store(
    store: CatalogStore,
    pipeline_version: str,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    method: str = "deterministic",
    provider: str = "auto",
    model: str | None = None,
    skip_missing_pages: bool = True,
    stop_on_error: bool = True,
    cds_ids: set[str] | None = None,
) -> dict[str, int]:
    counts = {
        "catalog": 0,
        "missing_markdown": 0,
        "skipped_missing_pages": 0,
        "skipped_existing": 0,
        "attempted": 0,
        "written": 0,
        "failed": 0,
        "assertions": 0,
    }
    for paper in store.load_catalog():
        counts["catalog"] += 1
        if cds_ids is not None and paper.cds_id not in cds_ids:
            continue
        if limit is not None and counts["attempted"] >= limit:
            break
        output_path = extraction_path(store, paper)
        if output_path.exists() and not overwrite:
            counts["skipped_existing"] += 1
            continue
        markdown_path = preferred_markdown_path(store.conversion_dir(paper, pipeline_version))
        if markdown_path is None:
            counts["missing_markdown"] += 1
            continue
        if skip_missing_pages and has_missing_pages(store, paper, pipeline_version, markdown_path):
            counts["skipped_missing_pages"] += 1
            continue
        counts["attempted"] += 1
        try:
            if method == "llm":
                from .llm_graph_extractor import available_provider, extract_paper_llm

                selected_provider = available_provider(provider)
                if selected_provider is None:
                    raise RuntimeError("No LLM API key is available for provider selection")
                extraction = extract_paper_llm(paper, markdown_path, provider=selected_provider, model=model)
            elif method == "auto":
                from .llm_graph_extractor import available_provider, extract_paper_llm

                selected_provider = available_provider(provider)
                extraction = (
                    extract_paper_llm(paper, markdown_path, provider=selected_provider, model=model)
                    if selected_provider
                    else extract_paper(paper, markdown_path)
                )
            elif method == "deterministic":
                extraction = extract_paper(paper, markdown_path)
            else:
                raise ValueError(f"Unsupported extraction method: {method}")
            extraction.validate()
            write_json(output_path, extraction.to_dict())
        except Exception as error:
            counts["failed"] += 1
            write_json(
                output_path,
                {
                    "cds_id": paper.cds_id,
                    "error": str(error),
                    "schema_version": GRAPH_SCHEMA_VERSION,
                    "status": "failed",
                },
            )
            if stop_on_error:
                raise
            continue
        counts["written"] += 1
        counts["assertions"] += len(extraction.assertions)
    return counts


def has_missing_pages(
    store: CatalogStore,
    paper: PaperRecord,
    pipeline_version: str,
    markdown_path: Path,
) -> bool:
    conversion = read_json(store.conversion_path(paper, pipeline_version), {})
    if conversion.get("missing_page_markers"):
        return True
    if conversion.get("status") == "failed":
        return True
    try:
        return "MISSING_PAGE" in markdown_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def extract_paper(paper: PaperRecord, markdown_path: Path) -> PaperExtraction:
    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    sections = split_sections(text)
    assertions = [
        CoverageAssertion(
            subject=EntityRef(kind="paper", id=f"{paper.experiment.lower()}:{paper.cds_id}", label=paper.title),
            predicate="paper_reports_result",
            object=EntityRef(kind="result", id=result_id(paper), label=paper.title),
            evidence=[
                EvidenceSpan(
                    section="metadata",
                    snippet=paper.title,
                    source_path=str(markdown_path),
                )
            ],
            confidence=1.0,
            extraction_method="metadata",
        )
    ]
    assertions.extend(extract_collision_and_dataset(paper, sections, markdown_path))
    assertions.extend(extract_terms(paper, sections, markdown_path, DETECTOR_OBJECTS, "detector_object", "result_defines_object"))
    assertions.extend(extract_terms(paper, sections, markdown_path, PHYSICS_PROCESSES, "physics_process", "result_targets_process"))
    assertions.extend(extract_terms(paper, sections, markdown_path, OBSERVABLES, "observable", "result_measures_observable"))
    assertions.extend(extract_terms(paper, sections, markdown_path, BACKGROUNDS, "background", "result_estimates_background"))
    assertions.extend(extract_regions(paper, sections, markdown_path))
    assertions.extend(extract_region_object_links(paper, sections, markdown_path))
    return PaperExtraction(
        cds_id=paper.cds_id,
        experiment=paper.experiment,  # type: ignore[arg-type]
        title=paper.title,
        result_type=paper.result_type,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        assertions=deduplicate_assertions(assertions),
    )


def result_id(paper: PaperRecord) -> str:
    return f"{paper.experiment.lower()}:{paper.cds_id}:result"


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title = "document"
    current_lines: list[str] = []
    for line in text.splitlines():
        match = SECTION_HEADINGS.match(line)
        if match:
            if current_lines:
                sections.append(Section(current_title, "\n".join(current_lines).strip()))
            current_title = clean_text(match.group(2))[:120] or "section"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append(Section(current_title, "\n".join(current_lines).strip()))
    return [section for section in sections if section.text]


def extract_collision_and_dataset(
    paper: PaperRecord,
    sections: list[Section],
    markdown_path: Path,
) -> list[CoverageAssertion]:
    assertions: list[CoverageAssertion] = []
    for section in sections[:8]:
        for match in ENERGY.finditer(section.text):
            snippet = snippet_around(section.text, match.start(), match.end())
            assertions.append(
                CoverageAssertion(
                    subject=result_entity(paper),
                    predicate="result_has_collision_system",
                    object=EntityRef(kind="collision_system", id=f"{match.group(1)}tev", label=f"{match.group(1)} TeV"),
                    evidence=[evidence(section, snippet, markdown_path)],
                    confidence=0.82,
                    qualifiers={"energy_tev": match.group(1)},
                    extraction_method="regex",
                )
            )
        for match in LUMINOSITY.finditer(section.text):
            snippet = snippet_around(section.text, match.start(), match.end())
            assertions.append(
                CoverageAssertion(
                    subject=result_entity(paper),
                    predicate="result_uses_dataset",
                    object=EntityRef(kind="dataset", label=f"{paper.experiment} dataset, {match.group(1)} inverse fb/pb"),
                    evidence=[evidence(section, snippet, markdown_path)],
                    confidence=0.78,
                    qualifiers={"integrated_luminosity": match.group(1)},
                    extraction_method="regex",
                )
            )
    return assertions


def extract_terms(
    paper: PaperRecord,
    sections: list[Section],
    markdown_path: Path,
    terms: list[TermPattern],
    entity_kind: str,
    predicate: str,
) -> list[CoverageAssertion]:
    assertions: list[CoverageAssertion] = []
    for term in terms:
        best = best_term_match(sections, term)
        if not best:
            continue
        section, match = best
        snippet = snippet_around(section.text, match.start(), match.end())
        assertions.append(
            CoverageAssertion(
                subject=result_entity(paper),
                predicate=predicate,  # type: ignore[arg-type]
                object=EntityRef(kind=entity_kind, id=term.id, label=term.label),  # type: ignore[arg-type]
                evidence=[evidence(section, snippet, markdown_path)],
                confidence=term.confidence,
                support="explicit",
                extraction_method="regex",
            )
        )
    return assertions


def extract_regions(paper: PaperRecord, sections: list[Section], markdown_path: Path) -> list[CoverageAssertion]:
    assertions: list[CoverageAssertion] = []
    seen: set[str] = set()
    for section in sections:
        if not SELECTION_SECTION.search(section.title) and not SELECTION_SECTION.search(section.text[:1000]):
            continue
        for match in REGION.finditer(section.text):
            label = clean_text(match.group(0)).lower()
            region_id = label.replace(" ", "_")
            if region_id in seen:
                continue
            seen.add(region_id)
            snippet = snippet_around(section.text, match.start(), match.end())
            assertions.append(
                CoverageAssertion(
                    subject=result_entity(paper),
                    predicate="result_defines_region",
                    object=EntityRef(kind="event_region", id=region_id, label=label),
                    evidence=[evidence(section, snippet, markdown_path)],
                    confidence=0.76,
                    extraction_method="regex",
                )
            )
    return assertions


def extract_region_object_links(
    paper: PaperRecord,
    sections: list[Section],
    markdown_path: Path,
) -> list[CoverageAssertion]:
    assertions: list[CoverageAssertion] = []
    for section in sections:
        if not SELECTION_SECTION.search(section.title) and not SELECTION_SECTION.search(section.text[:1000]):
            continue
        for paragraph in paragraphs(section.text):
            region_label: str | None = None
            for sentence in sentences(paragraph):
                region_match = REGION.search(sentence)
                if region_match:
                    region_label = clean_text(region_match.group(0)).lower()
                if region_label is None:
                    continue
                veto_sentence = _compile(r"\bveto(?:es|ed)?\b|\breject(?:s|ed)?\b").search(sentence)
                for term in DETECTOR_OBJECTS:
                    object_match = term.pattern.search(sentence)
                    if not object_match:
                        continue
                    predicate = "region_vetoes_object" if veto_sentence else "region_requires_object"
                    assertions.append(
                        CoverageAssertion(
                            subject=EntityRef(kind="event_region", id=region_label.replace(" ", "_"), label=region_label),
                            predicate=predicate,  # type: ignore[arg-type]
                            object=EntityRef(kind="detector_object", id=term.id, label=term.label),
                            evidence=[
                                evidence(
                                    section,
                                    snippet_around(sentence, object_match.start(), object_match.end()),
                                    markdown_path,
                                )
                            ],
                            confidence=0.68,
                            extraction_method="regex",
                        )
                    )
    return assertions


def best_term_match(sections: list[Section], term: TermPattern) -> tuple[Section, re.Match[str]] | None:
    fallback: tuple[Section, re.Match[str]] | None = None
    preferred_sections = SELECTION_SECTION if term in DETECTOR_OBJECTS else RESULT_SECTION
    if term in BACKGROUNDS:
        preferred_sections = BACKGROUND_SECTION
    for section in sections:
        match = term.pattern.search(section.text)
        if not match:
            continue
        if preferred_sections.search(section.title) or preferred_sections.search(section.text[:1000]):
            return section, match
        fallback = fallback or (section, match)
    return fallback


def deduplicate_assertions(assertions: list[CoverageAssertion]) -> list[CoverageAssertion]:
    deduplicated: list[CoverageAssertion] = []
    seen: set[tuple[str | None, str, str | None, str]] = set()
    for assertion in assertions:
        key = (
            assertion.subject.id or assertion.subject.label,
            assertion.predicate,
            assertion.object.id or assertion.object.label,
            assertion.evidence[0].section if assertion.evidence else "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(assertion)
    return deduplicated


def result_entity(paper: PaperRecord) -> EntityRef:
    return EntityRef(kind="result", id=result_id(paper), label=paper.title)


def evidence(section: Section, snippet: str, markdown_path: Path) -> EvidenceSpan:
    return EvidenceSpan(section=section.title, snippet=snippet, source_path=str(markdown_path))


def snippet_around(text: str, start: int, end: int, window: int = 260) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = clean_text(text[left:right])
    return snippet[:900]


def paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extraction_status(store: CatalogStore, pipeline_version: str) -> dict[str, int]:
    counts = {
        "catalog": 0,
        "missing_markdown": 0,
        "missing_extraction": 0,
        "extracted": 0,
        "failed": 0,
        "assertions": 0,
    }
    for paper in store.load_catalog():
        counts["catalog"] += 1
        markdown_path = preferred_markdown_path(store.conversion_dir(paper, pipeline_version))
        if markdown_path is None:
            counts["missing_markdown"] += 1
            continue
        path = extraction_path(store, paper)
        data: dict[str, Any] = read_json(path, {})
        if not data:
            counts["missing_extraction"] += 1
        elif data.get("status") == "failed":
            counts["failed"] += 1
        else:
            counts["extracted"] += 1
            counts["assertions"] += len(data.get("assertions", []))
    return counts
