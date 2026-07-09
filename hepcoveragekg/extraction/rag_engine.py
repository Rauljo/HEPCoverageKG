# =============================================================================
# HEPCoverageKG: cellular RAG engine
#
# Loops over config/schema.py's PREDICATE_SCHEMA, retrieves relevant context
# per predicate via CatalogState's hybrid retriever, asks Groq via
# generate(), validates the answer (controlled vocab / plausibility), and
# applies the confidence-beats rule via CatalogState.update_assertion.
#
# Ported from DeepCollector's rag_engine.py: execute_cellular_rag /
# _extract_cell_data_rag / _parse_response / _validate_plausibility. Dropped
# on purpose (not needed at our scale of ~6 predicates x 1 paper, vs
# DeepCollector's thousands of dataset x field cells): async batching +
# throttling, multi-query expansion, retry-with-shrinking-context, and the
# discovery/planning phases entirely (PREDICATE_SCHEMA already fixes what to
# ask - there's no "find unknown datasets" step here).
# =============================================================================
import json
import re
from typing import Any, Optional

from hepcoveragekg.config.schema import MISSING_DATA_PLACEHOLDERS, PLAUSIBILITY_THRESHOLDS, PREDICATE_SCHEMA
from hepcoveragekg.extraction.generate import generate
from hepcoveragekg.extraction.state import CatalogState, CoverageAssertion, EntityRef, EvidenceSpan

RAG_TOP_K = 6
CONTEXT_MAX_CHARS = 8000
EVIDENCE_SNIPPET_CHARS = 400


class RAGEngine:
    def __init__(self, config: Any):
        self.config = config
        self.verbosity = getattr(config, "VERBOSITY_LEVEL", 1)

    def run(self, state: CatalogState, result_label: str) -> list[CoverageAssertion]:
        """Asks every predicate in PREDICATE_SCHEMA about state.paper's result."""
        retriever = state.get_retriever(similarity_top_k=RAG_TOP_K, mode="HYBRID")
        if retriever is None:
            return []

        subject = EntityRef(kind="result", label=result_label)
        applied: list[CoverageAssertion] = []
        for predicate, spec in PREDICATE_SCHEMA.items():
            multi_value = spec.get("multi_value", False)
            if not multi_value:
                existing = state.get_assertion(predicate)
                if existing is not None and existing.confidence >= state.CONFIDENCE_LOCK_THRESHOLD:
                    continue

            for assertion in self._extract_predicate(state, subject, predicate, spec, retriever):
                if state.update_assertion(assertion, multi_value=multi_value):
                    applied.append(assertion)

        return applied

    def _extract_predicate(
        self, state: CatalogState, subject: EntityRef, predicate: str, spec: dict, retriever: Any
    ) -> list[CoverageAssertion]:
        vocab = spec.get("controlled_vocabulary")
        qualifier_field = spec.get("qualifier_field")
        query = spec["query"].format(result=subject.label, vocab=", ".join(vocab) if vocab else "")

        nodes = retriever.retrieve(query)
        if not nodes:
            return []

        context = self._format_context(nodes)[:CONTEXT_MAX_CHARS]
        prompt = self._build_prompt(query, context)

        try:
            raw_response = generate(prompt)
        except Exception as error:
            if self.verbosity >= 1:
                print(f"[RAGEngine] generate() failed for {predicate}: {error}")
            return []

        value, confidence, rationale = self._parse_response(raw_response)
        if value.lower() in MISSING_DATA_PLACEHOLDERS:
            return []

        evidence = [
            EvidenceSpan(
                section=node.node.metadata.get("heading", ""),
                snippet=node.node.get_content()[:EVIDENCE_SNIPPET_CHARS],
                source_url=node.node.metadata.get("source_url"),
            )
            for node in nodes[:2]
        ]

        if qualifier_field:
            is_plausible, reason = self._validate_plausibility(qualifier_field, value)
            if not is_plausible:
                if self.verbosity >= 2:
                    print(f"[RAGEngine] {predicate} value '{value}' rejected: {reason}")
                return []
            return [
                CoverageAssertion(
                    subject=subject,
                    predicate=predicate,
                    object=EntityRef(kind=spec["object_kind"], label=value),
                    evidence=evidence,
                    confidence=confidence,
                    qualifiers={qualifier_field: value},
                    extraction_method="llm",
                    notes=rationale or None,
                )
            ]

        if vocab is not None:
            matches = self._extract_vocab_terms(raw_response, vocab)
            return [
                CoverageAssertion(
                    subject=subject,
                    predicate=predicate,
                    object=EntityRef(kind=spec["object_kind"], label=term),
                    evidence=evidence,
                    confidence=term_confidence,
                    extraction_method="llm",
                    notes=term_rationale or None,
                )
                for term, term_confidence, term_rationale in matches
            ]

        return []

    @staticmethod
    def _format_context(nodes: list[Any]) -> str:
        parts = []
        for node in nodes:
            heading = node.node.metadata.get("heading", "")
            parts.append(f"--- [{heading}] ---\n{node.node.get_content()}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            f"Question: {query}\n\nContext:\n{context}\n\n"
            'Instructions: Answer strictly based on the context above. If the answer '
            'isn\'t in the context, set value to "[missing]".\n'
            "Respond with ONLY raw JSON, no markdown fences: "
            '{"value": "...", "confidence": 0.0-1.0, "rationale": "..."}'
        )

    @staticmethod
    def _extract_json_objects(text: str) -> list[dict]:
        """Parses one or more flat (non-nested) JSON objects from raw LLM output.

        llama-3.1-8b-instant sometimes ignores the "respond with ONE JSON
        object" instruction for controlled-vocabulary "choose all that
        apply" questions, and instead emits one object per candidate term -
        a per-term verdict - rather than a single comma-joined value. This
        parses either shape instead of fighting the model into one format.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        objects = []
        for match in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
            try:
                objects.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
        return objects

    @staticmethod
    def _normalize_confidence(raw_confidence: Any) -> float:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return 0.0
        # LLMs sometimes answer confidence on a 0-10 or 0-100 scale rather
        # than 0-1 - normalize here so CoverageAssertion.validate()'s strict
        # [0,1] check never has to reject a well-formed answer for this.
        if confidence > 10.0:
            confidence /= 100.0
        elif confidence > 1.0:
            confidence /= 10.0
        return min(max(confidence, 0.0), 1.0)

    @classmethod
    def _parse_response(cls, text: str) -> tuple[str, float, str]:
        objects = cls._extract_json_objects(text)
        if not objects:
            return "[missing]", 0.0, ""
        data = objects[0]
        value = str(data.get("value", "[missing]")).strip()
        rationale = str(data.get("rationale", ""))
        confidence = cls._normalize_confidence(data.get("confidence", 0.0))
        return value, confidence, rationale

    @classmethod
    def _extract_vocab_terms(cls, raw_response: str, vocab: list[str]) -> list[tuple[str, float, str]]:
        """Returns (term, confidence, rationale) per accepted controlled-
        vocabulary term, handling both the well-behaved single-object,
        comma-joined answer and the per-term-verdict multi-object shape."""
        objects = cls._extract_json_objects(raw_response)
        accepted: list[tuple[str, float, str]] = []
        seen: set[str] = set()

        for obj in objects:
            value = str(obj.get("value", "")).strip()
            if not value or value.lower() in MISSING_DATA_PLACEHOLDERS:
                continue
            rationale = str(obj.get("rationale", ""))
            confidence = cls._normalize_confidence(obj.get("confidence", 0.0))

            if len(objects) == 1:
                # Single object: value may name several terms at once.
                for term in vocab:
                    if term in seen:
                        continue
                    if term.lower() in value.lower() or term.replace("_", " ") in value.lower():
                        accepted.append((term, confidence, rationale))
                        seen.add(term)
                continue

            # Multi-object (per-term verdict): only accept a term this
            # specific object names exactly, above chance confidence.
            exact_term = next((t for t in vocab if t.lower() == value.lower()), None)
            if exact_term and exact_term not in seen and confidence > 0.5:
                accepted.append((exact_term, confidence, rationale))
                seen.add(exact_term)

        return accepted

    @staticmethod
    def _validate_plausibility(field: str, value: str) -> tuple[bool, str]:
        thresholds = PLAUSIBILITY_THRESHOLDS.get(field)
        if not thresholds:
            return True, ""
        numbers = re.findall(r"\d+\.?\d*", value.replace(",", ""))
        if not numbers:
            return True, ""
        max_value = max(float(n) for n in numbers)
        if "min" in thresholds and max_value < thresholds["min"]:
            return False, f"{max_value} below minimum {thresholds['min']}"
        if "max" in thresholds and max_value > thresholds["max"]:
            return False, f"{max_value} above maximum {thresholds['max']}"
        return True, ""
