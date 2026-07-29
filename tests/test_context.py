# =============================================================================
# Entity context for adjudication: every wording plus the defining quotes.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import context


def test_definitional_sections_outrank_prose():
    assert context.section_rank("Object reconstruction") < context.section_rank("Results")
    assert context.section_rank("Event selection") < context.section_rank("Introduction")
    assert context.section_rank("Data and simulated samples") < context.section_rank("Abstract")
    # unknown sections must not be dumped at the bottom
    assert context.section_rank("Some bespoke heading") < context.section_rank("Introduction")
    assert context.section_rank(None) == context.section_rank("")


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
