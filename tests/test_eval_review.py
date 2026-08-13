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
