"""LaTeX to readable HTML, for the sentences a physicist is asked to judge.

70 of batch 2's 104 rows carry a maths span. Left raw, the supervisor does the
typesetting himself forty times before he can judge anything.
"""
from __future__ import annotations

import pytest

from hepcoveragekg.eval.latex_html import to_html


@pytest.mark.parametrize("raw,expect", [
    (r"$p_{\mathrm{T}}^{\text{miss}}$", "p<sub>T</sub><sup>miss</sup>"),
    (r"$H_{\mathrm{T}}>500\,\text{Ge\hskip-0.80002ptV}$", "H<sub>T</sub>&gt;500 GeV"),
    (r"$t\bar{t}$", "tt̄"),
    (r"$\tilde{\chi}^{0}_{2}$", "χ̃<sup>0</sup><sub>2</sub>"),
    (r"$H\to Z\eta_{c}$", "H → Zη<sub>c</sub>"),
    (r"$|\eta|<2.4$", "|η|&lt;2.4"),
])
def test_the_patterns_this_corpus_actually_contains(raw, expect):
    assert to_html(raw) == expect


def test_the_papers_own_angle_brackets_cannot_become_tags():
    """`120 < m < 135` is physics, not markup. Escaping happens BEFORE our own
    sub/sup tags go in."""
    out = to_html(r"$120<m<135$ and <script>alert(1)</script>")
    assert "&lt;" in out and "<script>" not in out
    assert "<sub>" not in out


def test_symbols_are_replaced_before_accents():
    r"""The other order puts the combining mark on the backslash: `\tilde{\chi}`
    rendered as `\̃chi`, because the accent handler takes the first CHARACTER of
    the body and the body still began with a command."""
    assert to_html(r"$\tilde{\chi}$") == "χ̃"


def test_an_unknown_command_keeps_its_text():
    r"""Deleting the name ate the L from `\LHC`, and a reader cannot tell a
    silent deletion from a paper that never said it."""
    assert "LHC" in to_html(r"an \LHC collision")
    assert "Q" in to_html(r"the $\mathcal{Q}$ operator")


def test_escaped_braces_do_not_orphan_a_backslash():
    r"""`\left\{\mathcal{Q}\right\}` came out as `\Q\`: the final brace-strip
    removed the `{` and left the backslash behind."""
    assert "\\" not in to_html(r"operators $\left\{\mathcal{Q}\right\}$")


def test_typesetting_noise_is_dropped_but_the_word_survives():
    """LaTeXML splits GeV with a kern. The kern is noise; GeV is content."""
    assert to_html(r"$40\,\text{Ge\kern-1.00006ptV}$") == "40 GeV"


def test_empty_and_plain_text_pass_through():
    assert to_html("") == ""
    assert to_html("no maths here at all") == "no maths here at all"
