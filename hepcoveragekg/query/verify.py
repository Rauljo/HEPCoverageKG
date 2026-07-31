"""
HEPCoverageKG query layer: the faithfulness check.

Every claim in an answer must trace to a row that was actually retrieved
(system.md S-14). This checks it **mechanically**, not by asking a model.

Why mechanical matters. The standard approach is LLM-as-judge -- ask a second
model whether the answer is grounded. That inherits every weakness of the first:
it is fluent, it is agreeable, and it cannot tell a correct-sounding invention
from a retrieved fact. This graph makes something stronger possible, because
97% of assertions carry the verbatim sentence they came from, so "did this
number come out of the database" has a yes/no answer that needs no judgement.

The specific failure this is aimed at is the one PURPOSE warns the model about:
a physics-competent model filling a gap with something TRUE ABOUT THE WORLD but
absent from this corpus. That is invisible to a domain expert reading the answer
-- it reads as authoritative and it is correct physics -- and it is precisely
what a coverage map must never do, because the whole claim of the project is
that it reports what the literature contains rather than what is true.

Deliberately asymmetric: a claim is challenged only when the retrieved rows
contain nothing resembling it. Over-flagging would train the reader to ignore
the check, which is worse than missing a few.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# What gets checked. Numbers and identifiers are the claims that can be wrong in
# a way nobody notices; prose is left alone because paraphrase is legitimate and
# flagging it would drown the real findings.
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")
_ENTITY_ID = re.compile(r"\bhepkg:[a-z_]+:[A-Za-z0-9_\-]+\b")

# Numbers that mean something other than a retrieved quantity, and would be
# flagged constantly for no benefit: small counts used rhetorically ("the two
# analyses"), years, and the collision energies that appear in every paper.
_UNREMARKABLE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                 "13", "13.6", "14", "7", "8"}


@dataclass
class Claim:
    text: str
    kind: str  # "number" | "arxiv_id" | "entity_id"
    supported: bool
    note: str = ""


@dataclass
class Verification:
    claims: list[Claim] = field(default_factory=list)
    checked: int = 0
    cited_evidence: int = 0

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    @property
    def score(self) -> float:
        """Share of checkable claims the retrieved rows support. 1.0 when there
        was nothing to check -- an answer making no factual claims cannot be
        unfaithful, only unhelpful."""
        return 1.0 if not self.checked else 1.0 - len(self.unsupported) / self.checked

    def report(self) -> str:
        if not self.checked:
            return "no checkable claims in the answer"
        lines = [f"{self.checked - len(self.unsupported)}/{self.checked} claims grounded "
                 f"({self.score:.0%}); {self.cited_evidence} evidence quotes available"]
        for c in self.unsupported:
            lines.append(f"  UNSUPPORTED [{c.kind}] {c.text}"
                         + (f" -- {c.note}" if c.note else ""))
        return "\n".join(lines)


def _normalise(token: str) -> set[str]:
    """The forms a number might legitimately have been written in.

    "1,286" retrieved and "1286" written are the same claim; so are "58" and
    "58.0". Without this the check would flag formatting as invention.
    """
    plain = token.replace(",", "")
    forms = {token.lower(), plain.lower()}
    if plain.replace(".", "").isdigit():
        try:
            value = float(plain)
            forms.add(str(int(value)) if value.is_integer() else str(value))
        except ValueError:
            pass
    return forms


def verify(answer: str, seen_values: Iterable[str],
           evidence_ids: Optional[Iterable[str]] = None) -> Verification:
    """Check an answer against everything the tools actually returned.

    `seen_values` is collected by the planner as results arrive -- *before*
    truncation, because a claim is grounded if the graph returned it at all,
    not only if it survived into the model's context window.
    """
    # Normalise BOTH sides. Normalising only the answer meant a row containing
    # "1,286" failed to ground an answer saying "1286" -- formatting reported as
    # invention, which is the one thing this check must never do.
    seen: set[str] = set()
    for value in seen_values:
        seen |= _normalise(str(value))
    result = Verification(cited_evidence=len(list(evidence_ids or [])))

    for pattern, kind in ((_ARXIV, "arxiv_id"), (_ENTITY_ID, "entity_id"),
                          (_NUMBER, "number")):
        for token in dict.fromkeys(pattern.findall(answer)):
            if kind == "number":
                if token.replace(",", "") in _UNREMARKABLE:
                    continue
                # An arXiv id is also a number; do not check it twice.
                if _ARXIV.fullmatch(token):
                    continue
            forms = _normalise(token)
            supported = bool(forms & seen)
            note = ""
            if not supported and kind == "number":
                note = "no retrieved row contained this value"
            elif not supported:
                note = "not among the ids returned"
            result.claims.append(Claim(token, kind, supported, note))

    result.checked = len(result.claims)
    return result


def verify_session(session) -> Verification:
    """Convenience: check a planner Session against its own trace."""
    return verify(session.answer, session.seen_values, session.evidence_ids)
