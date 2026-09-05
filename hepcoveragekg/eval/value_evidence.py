"""Evidence for a VALUE row, chosen because it contains the value.

THE BUG THIS EXISTS TO FIX. The value rows carried `all_quotes` from the run --
the sentences the agent RETRIEVED on its way to an answer. That is the retrieval
footprint, and it is not the same thing as support. Three of the seven rows
stated a number that appeared in none of the sentences shown beside it:

    gf-12  "875 GeV"              three quotes about figures and SR binning
    gf-14  "300 GeV / 160 GeV"    two quotes about previous searches
    gf-15  "observed 3, expected 5.7"   quotes about validation regions

The paper contains the supporting sentence in every one of those cases
("Masses of the t2 up to 875 GeV are excluded at 95% CL..."). We were showing
the wrong sentences, not missing ones. Asking the supervisor "is 875 GeV right?"
beside evidence that never mentions 875 forces him back into the PDF, which is
the whole cost the review app exists to remove.

HOW A SENTENCE IS CHOSEN. Take the numbers out of our answer, find sentences in
that paper containing them, and prefer the numbers that are RARE in the paper.
Rarity is what makes this work: "5.7" occurs in one block of 2006.05880 and
picks out the yields table immediately, while "3" occurs in most of them and
would pick out noise. An anchor common enough to be uninformative is dropped
rather than allowed to drag in a sentence that merely contains a 3.

WHEN NOTHING MATCHES we return nothing and the caller keeps the run's own
quotes. A row whose answer is prose ("what role does each region play") has no
number to anchor to, and inventing a match would be worse than the footprint.
"""
from __future__ import annotations

import re
import sqlite3

from . import latex_html
from typing import Iterable, Optional

#: A number worth anchoring to has to be a VALUE, not an identifier. The first
#: version took every digit, so the 1 in "chi^0_1" and the 2 in "t2" were
#: anchors, and they matched citation brackets and subscripts all over the
#: paper -- gf-06 came back with the paper's title and a list of references.
#: A value is a number carrying a unit, or a decimal (5.7, 4.8), or a count
#: with its noun ("3 events").
_UNIT_BARE = r"(?:GeV|TeV|MeV|fb|pb|%|events?|sigma|standard deviations?)"
_UNIT = rf"(?:\s*{_UNIT_BARE})"
_NUM = re.compile(rf"\d+\.\d+|\d+(?={_UNIT})", re.I)

#: An anchor in more than this share of the paper's blocks tells us nothing
#: about WHICH sentence supports the answer.
_TOO_COMMON = 0.20

#: Table blocks are one long run of cells. Showing the whole thing is unreadable
#: and showing one sentence is impossible, so we show a window around the hit.
_WINDOW = 260


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.:])\s+(?=[A-Z(\\$])", text)
    return [p for p in (p.strip() for p in parts) if p]


def anchors(answer: str) -> list[str]:
    """The values our answer commits to, longest first.

    Longest first because a match on "1220" is worth more than a match on the
    "12" inside it.
    """
    seen, out = set(), []
    for n in _NUM.findall(answer or ""):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return sorted(out, key=lambda n: (-len(n), n))


def _blocks(conn: sqlite3.Connection, paper_id: str) -> list[tuple[str, bool]]:
    rows = conn.execute(
        """SELECT b.text, b.table_rows FROM source_block b
             JOIN source_snapshot s ON s.source_hash = b.source_hash
            WHERE s.paper_id = ? AND b.text IS NOT NULL
            ORDER BY b.block_order""", (paper_id,)).fetchall()
    # COLLAPSED BEFORE ANYTHING CUTS IT. `\mathchoice{a}{b}{c}{d}` is found by
    # counting braces, so a window that lands inside one leaves a fragment with
    # no four arguments to match and the renderer prints the command name and
    # three redundant copies of the particle -- which is exactly how gf-12's
    # second quote read. Collapsing here means every later step sees text that
    # is safe to cut anywhere.
    return [(latex_html.collapse_mathchoice(t), bool(tr))
            for t, tr in rows if (t or "").strip()]


def _hit(number: str, text: str) -> bool:
    """Does `text` use `number` as a VALUE?

    A whole-number anchor must carry a unit here, exactly as it did in the
    answer. Without that rule the "15" of "|mll-mZ|<15 GeV" matched the
    citation bracket "[ 14 , 15 ]", the "60" of "60% b-tagging" matched
    "Phys. Rev. D 60 (1999)", and the "166" of an author's affiliation matched
    anything -- gf-11 was given two quotes from the bibliography and the CMS
    author list. Decimals (5.7, 4.8) are exempt: they are almost never
    citation numbers, and the yields table Gabriel needs for gf-15 writes them
    bare.
    """
    core = rf"(?<![\d.]){re.escape(number)}(?![\d])"
    if "." not in number:
        # PAPERS FOLD TWO VALUES INTO ONE PHRASE: "approximately 10 (60)%
        # tagging efficiencies for c (b) quark jets" is the sentence gf-11
        # needs, and neither 10 nor 60 sits directly against the %. Requiring
        # strict adjacency sent gf-11 to a sentence about jet momentum
        # resolution that happened to say "5 to 10%". A parenthesised
        # alternative, or a closing paren, may intervene -- but nothing else,
        # or citation brackets come back.
        core += rf"(?!\.\d)(?:\s*\(\d+(?:\.\d+)?\))?\)?\s*{_UNIT_BARE}"
    return re.search(core, text, re.I) is not None


def find(conn: sqlite3.Connection, paper_id: str, answer: str,
         limit: int = 3) -> list[str]:
    """Sentences from `paper_id` that contain the values `answer` commits to.

    Sentences are ranked by how much of the answer they account for, not one
    per anchor. Taking one sentence per anchor filled three rows with a good
    quote followed by two irrelevant ones, because a weak anchor always matches
    SOMETHING. Ranking lets a single sentence carrying both "875 GeV" and
    "350 GeV" win outright, and lets a weak anchor contribute nothing.
    """
    blocks = _blocks(conn, paper_id)
    if not blocks:
        return []

    weight: dict[str, float] = {}
    for a in anchors(answer):
        n = sum(1 for text, _ in blocks if _hit(a, text))
        if n and n <= max(1, int(_TOO_COMMON * len(blocks))):
            weight[a] = 1.0 / n          # rarer anchor, heavier
    if not weight:
        return []

    scored: list[tuple[float, str]] = []
    for text, is_table in blocks:
        present = [a for a in weight if _hit(a, text)]
        if not present:
            continue
        # SPLIT FIRST, WINDOW ONLY AS A FALLBACK. Windowing whole blocks up
        # front put gf-11 in the wrong place: the b-tagging block is prose but
        # carries `table_rows`, so it was cut around its FIRST number and the
        # window closed before "10 (60)% tagging efficiencies" -- the one
        # sentence that answers the question. Splitting into sentences works
        # for prose, and a real table simply does not split, so it falls
        # through to the window with nothing lost.
        for piece in _sentences(text):
            hits = [a for a in present if _hit(a, piece)]
            if not hits:
                continue
            if len(piece) > 3 * _WINDOW:
                piece = _window(piece, hits)
                hits = [a for a in hits if _hit(a, piece)] or hits
            scored.append((sum(weight[a] for a in hits) + 0.001 * len(hits),
                           " ".join(piece.split())))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    # A sentence far weaker than the best one is not support, it is a
    # coincidence -- the noise rows all sat an order of magnitude below.
    floor = scored[0][0] * 0.5
    return dedupe((q for score, q in scored if score >= floor), keep=limit)


def _window(text: str, present: list[str]) -> str:
    """A readable slice of a table around the densest run of matches.

    Snapped to spaces: the first version cut mid-word and produced
    "... gnitude, p T miss, is required to be less than 40 GeV".
    """
    m = min((re.search(rf"(?<![\d.]){re.escape(a)}(?![\d])", text) for a in present),
            key=lambda m: m.start() if m else 10 ** 9)
    start, end = max(0, m.start() - _WINDOW // 2), m.start() + _WINDOW
    if start:
        start = text.find(" ", start) + 1 or start
    if end < len(text):
        cut = text.rfind(" ", start, end)
        end = cut if cut > start else end
    piece = text[start:end]
    return ("… " if start else "") + piece + (" …" if end < len(text) else "")


def dedupe(quotes: Iterable[str], keep: int = 3) -> list[str]:
    """Drop quotes that say the same thing.

    Three of the value rows showed the SAME sentence three times, because the
    agent retrieved one passage from three LaTeXML renderings that differ only
    in spacing and how a t-bar is encoded. Comparing letters and digits alone
    collapses them; the reader loses nothing, since they were identical to read.
    """
    out, seen = [], set()
    for q in quotes:
        if not (q or "").strip():
            continue
        # Compared on a PREFIX. Two renderings of one sentence differ in
        # spacing and in whether a tie is written "ℓ≡e,μ" or "ℓequiv e,μ", so
        # a whole-string key let gf-06 show the same sentence twice. The first
        # 120 characters are enough to identify a sentence and short enough to
        # land before the encodings diverge.
        key = re.sub(r"[^a-z0-9]+", "", re.sub(r"<[^>]+>", "", q).lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= keep:
            break
    return out
