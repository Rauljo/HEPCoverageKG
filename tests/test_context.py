# =============================================================================
# Entity context for adjudication: every wording plus the defining quotes.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import context


def test_section_priority_is_learned_per_kind(monkeypatch):
    """A single global ordering was wrong: physics_process takes 30% of its
    quotes from the Introduction and 2% from reconstruction, while
    detector_object is the reverse. Priority is measured per kind."""
    monkeypatch.setattr(context, "_SECTION_PRIORS", {
        "detector_object": {"object reconstruction": 500, "introduction": 5},
        "physics_process": {"object reconstruction": 5, "introduction": 400},
    })
    # same two sections, opposite verdicts depending on the kind
    assert context.section_rank("Object reconstruction", "detector_object") < \
           context.section_rank("Introduction", "detector_object")
    assert context.section_rank("Introduction", "physics_process") < \
           context.section_rank("Object reconstruction", "physics_process")


def test_unseen_sections_rank_neutral_not_last(monkeypatch):
    """An unusual heading is not evidence of being uninformative."""
    monkeypatch.setattr(context, "_SECTION_PRIORS",
                        {"detector_object": {"results": 100, "introduction": 50}})
    unseen = context.section_rank("Three-body event selection", "detector_object")
    assert unseen == 0
    assert unseen > context.section_rank("Results", "detector_object")   # known-good still wins
    assert context.section_rank(None, "detector_object") == 0


def test_render_omits_empty_sections():
    c = context.EntityContext("e1", "detector_object", labels=["Electron"])
    text = c.render("ENTITY A")
    assert "ENTITY A:" in text and 'primary name: "Electron"' in text
    for absent in ("also written as", "aliases:", "attributes:", "quotes from the papers"):
        assert absent not in text


def test_render_shows_every_wording_and_labels_the_section():
    c = context.EntityContext(
        "e1", "background",
        labels=["Diboson background", "Diboson ($VV$) background"],
        aliases=["diboson"],
        attributes={"level": "reconstructed"},
        papers=8,
        quotes=[("Event selection", "Diboson production is estimated from simulation.")],
    )
    text = c.render("ENTITY B")
    assert 'also written as: "Diboson ($VV$) background"' in text
    assert "appears in 8 paper(s)" in text
    assert "quotes from the papers:" in text
    assert '- [Event selection] "Diboson production is estimated from simulation."' in text


def test_long_quotes_are_truncated_not_dropped():
    long_quote = "x" * (context.QUOTE_CHARS + 200)
    out = context._truncate(long_quote)
    assert len(out) <= context.QUOTE_CHARS
    assert out.endswith("…")


def test_whitespace_is_collapsed():
    assert context._truncate("a\n\n  b\tc") == "a b c"
