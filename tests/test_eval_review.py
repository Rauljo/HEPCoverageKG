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
