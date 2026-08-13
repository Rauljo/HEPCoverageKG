"""The review sheet Gabriel actually reads.

Bugs here are worse than bugs in the pipeline: they mislead a human whose
judgements become our ground truth, and he has no way to see the error.
"""
import pathlib
import tempfile

from hepcoveragekg.eval import review as RV
from hepcoveragekg.eval import supervisor as S



def test_a_single_paper_item_carries_the_whole_gathered_set():
    """One sentence for a three-part question is the D-058 failure, committed
    against the human reviewer instead of the judge."""
    rows = [{"qid": "gf-11", "paper_id": "2001.06899",
             "all_quotes": ["the algorithm is CSVv2",
                            "a medium working point is used"],
             "answers": ["CSVv2, medium"], "quote_supports": True,
             "support_why": "both parts are established"}]
    qs = [q for q in S.build_records() if q["qid"] == "gf-11"]
    item, = RV.single_paper_items(rows, qs)
    assert len(item["quotes"]) == 2
    assert item["_judge"] is True
    assert item["_why"] == "both parts are established"


def test_the_html_shows_every_gathered_sentence():
    items = [{"qid": "gf-11", "question": "which algorithm and working point?",
              "paper_id": "p1", "quote": "the algorithm is CSVv2",
              "quotes": ["the algorithm is CSVv2", "a medium working point is used"],
              "candidates": [], "_machine": True, "_judge": True,
              "_why": "both parts", "_by": "single-paper run", "row": 1}]
    with tempfile.TemporaryDirectory() as d:
        p = RV.write_html(items, pathlib.Path(d) / "s.html", "t")
        html = p.read_text(encoding="utf-8")
    assert "a medium working point is used" in html
    assert "gathered 2 sentences" in html


def test_the_sheet_never_reveals_a_verdict_before_it_is_asked_for():
    """The sheet produces our human ground truth. If it shows what we concluded,
    his marks become a copy of ours and the exercise measures nothing."""
    import re
    items = [{"qid": "gf-10", "question": "what efficiency?", "paper_id": "p1",
              "quote": "the cut retains 80%", "quotes": ["a", "the cut retains 80%"],
              "candidates": [], "_machine": True, "_judge": False,
              "_why": "that is an uncertainty, not an efficiency",
              "_by": "single-paper run", "row": 1}]
    with tempfile.TemporaryDirectory() as d:
        html = RV.write_html(items, pathlib.Path(d) / "s.html", "t").read_text()
    visible = re.sub(r"<details>.*?</details>", "", html, flags=re.S)
    for word in ("does NOT support", "supports the claim", "not checked by our judge"):
        assert word not in visible
    assert "that is an uncertainty" not in visible, "the judge's reason must stay hidden too"
    assert "does NOT support" in html, "but it must be there once opened"


def test_the_sheet_says_a_paper_missing_from_his_list_is_not_our_error():
    """His gold was computed by filtering the pilot export -- graph-agreement,
    not physics truth. A reviewer meeting an unfamiliar paper would naturally
    read it as our mistake, which would convert every genuine discovery into a
    false positive and make the sheet unable to measure what it is for."""
    items = [{"qid": "gf-05", "question": "q", "paper_id": "p1", "quote": "x" * 40,
              "candidates": [], "_machine": True, "_judge": True, "_why": "w",
              "_by": "stage-1", "row": 1}]
    with tempfile.TemporaryDirectory() as d:
        html = RV.write_html(items, pathlib.Path(d) / "s.html", "t").read_text()
    assert "your own list does not contain" in html
    assert "not automatically our error" in html
    assert "against your list" in html


def test_every_tsv_record_is_exactly_one_line():
    """Excel and Sheets do not handle quoted TSV reliably. If his rows shift by
    one, every verdict after that attaches to the wrong paper, nothing looks
    wrong, and the ground truth is silently corrupted."""
    items = [{"qid": "gf-01", "question": "does it\nuse b-jets?", "paper_id": "p1",
              "quote": "a sentence\twith a tab\nand a newline", "candidates": [],
              "_machine": True, "_judge": True, "_why": "", "_by": "s", "row": 1}]
    with tempfile.TemporaryDirectory() as d:
        p = RV.write_sheet(items, pathlib.Path(d) / "s.tsv")
        raw = p.read_text(encoding="utf-8")
    assert len(raw.strip().splitlines()) == 2, "one header line, one data line"
    assert '"' not in raw, "no quoting needed once the fields are single-line"


def test_the_tsv_shows_the_same_evidence_as_the_html():
    items = [{"qid": "gf-11", "question": "algorithm and working point?",
              "paper_id": "p1", "quote": "the algorithm is CSVv2",
              "quotes": ["the algorithm is CSVv2", "a medium working point is used"],
              "candidates": [], "_machine": True, "_judge": True, "_why": "",
              "_by": "s", "row": 1}]
    with tempfile.TemporaryDirectory() as d:
        tsv = RV.write_sheet(items, pathlib.Path(d) / "s.tsv").read_text()
    assert "medium working point" in tsv, "the TSV must not show one of three sentences"
