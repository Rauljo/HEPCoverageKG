"""The value rows must show the sentence that CONTAINS the value.

They shipped showing `all_quotes` -- the sentences the agent read on its way to
an answer. For three of the seven rows the number appeared in none of them, so
the sheet asked "is 875 GeV right?" beside three quotes that never said 875.
"""
import sqlite3

import pytest

from hepcoveragekg.eval import value_evidence as V


@pytest.fixture
def paper():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE source_snapshot(source_hash TEXT, paper_id TEXT);
        CREATE TABLE source_block(source_hash TEXT, block_order INT,
                                  text TEXT, table_rows TEXT);
        INSERT INTO source_snapshot VALUES('h','2006.05880');
    """)
    blocks = [
        "Masses of the t2 up to 875 GeV are excluded at 95% CL for a chi01 "
        "mass of about 350 GeV.",
        "Searches for top squark production have been performed previously "
        "by both ATLAS [ 20 , 21 ] and CMS [ 22 , 23 ] .",
        "TeV -scale masses are favoured [ 12 , 13 ] for the partners of the "
        "gluons, and the top squarks [ 14 , 15 ] .",
        "J. M. Campbell, Phys. Rev. D 60 (1999) 113006, arXiv:hep-ph/9905386 .",
        "N.L. Abraham 155 , H. Abramowicz 160 , H. Abreu 159 , Y. Abulaiti 6 ,",
    ]
    for i, t in enumerate(blocks):
        conn.execute("INSERT INTO source_block VALUES('h',?,?,NULL)", (i, t))
    return conn


def test_finds_the_sentence_holding_the_value(paper):
    got = V.find(paper, "2006.05880", "875 GeV")
    assert got and "875 GeV" in got[0]


def test_citation_brackets_are_not_evidence(paper):
    """`[ 14 , 15 ]` is a reference list, not "15 GeV"."""
    assert V.find(paper, "2006.05880", "|mll-mZ| < 15 GeV") == []


def test_journal_volume_is_not_a_percentage(paper):
    """"Phys. Rev. D 60 (1999)" once answered a question about 60% b-tagging."""
    for q in V.find(paper, "2006.05880", "60% for b quark jets"):
        assert "Campbell" not in q


def test_author_affiliation_numbers_are_not_evidence(paper):
    for q in V.find(paper, "2006.05880", "up to 160 GeV"):
        assert "Abramowicz" not in q


def test_prose_answer_falls_back_rather_than_inventing(paper):
    """No value to anchor to -> nothing, and the caller keeps the run quotes."""
    assert V.find(paper, "2006.05880", "the region targets large splittings") == []


def test_paired_values_still_match():
    """Papers fold two numbers into one phrase: "10 (60)% efficiencies"."""
    assert V._hit("10", "approximately 10 (60)% tagging efficiencies")
    assert V._hit("60", "approximately 10 (60)% tagging efficiencies")


def test_decimals_need_no_unit():
    """The yields table writes "5.7 +- 1.0" bare, and it is the evidence."""
    assert V._hit("5.7", "Total (post-fit) SM events 5.7 ± 1.0 12.1 ± 2.0")


def test_dedupe_collapses_one_sentence_rendered_twice():
    """LaTeXML gives the same sentence with different spacing and ties."""
    a = "distinguished from the SM top quark pair production ( tt̄ ) background"
    b = "distinguished from the SM top quark pair production (t t̄) background"
    assert len(V.dedupe([a, b])) == 1
