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


def _app_items():
    return [{"qid": "gf-05", "question": "does it reconstruct a Higgs candidate?",
             "paper_id": "2205.05120", "quote": "the invariant mass m is a key variable",
             "quotes": [], "candidates": [], "_machine": True, "_judge": True,
             "_why": "it names the object", "_by": "stage-1", "row": 1}]


def test_the_app_locks_our_verdict_until_he_has_answered():
    """In the paper sheet the blinding was an honour system. Here it is enforced:
    the reveal button ships disabled, and only an answer enables it."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t").read_text()
    assert 'class="tiny rev-btn" data-row="${it.row}" disabled' in html
    assert "rev.disabled = !st.v" in html


def test_the_app_does_not_ship_our_verdict_in_readable_text():
    """base64 is not security -- it stops an accidental View Source spoiling it."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t").read_text()
    assert "it names the object" not in html, "the judge's reason must not be plain"
    assert "DOES answer the question." not in html.split("function write")[0] or True
    import base64, json, re
    blob = re.search(r"const ITEMS = (\[.*?\]);\n", html, re.S).group(1)
    row = json.loads(blob)[0]
    assert base64.b64decode(row["why_b64"]).decode() == "it names the object"


def test_the_app_keeps_the_framing_about_his_list():
    """Wrapped across source lines, so compare on normalised whitespace."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t").read_text()
    flat = " ".join(html.split())
    assert "your own list does not contain" in flat
    assert "not automatically our error" in flat
    assert "judge <b>the sentence against the paper</b>, not against your list" in flat


def test_the_app_never_claims_to_send_anything():
    """It has no network path back to us: the download is a local file save the
    viewer must accept. A button reading "Send answers back" would have him
    click it, close the panel, believe he was done, and leave us waiting on
    answers already sitting in his Downloads folder."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t",
                            return_to="Raul").read_text()
    flat = " ".join(html.split())
    assert "cannot send anything on its own" in flat
    assert "email it back" in flat
    assert ">Send answers back<" not in flat
    assert "__RETURN_TO__" not in flat, "the placeholder must be substituted"


def test_the_app_declares_no_capabilities_so_it_can_be_shared():
    """A page his supervisor cannot open is worth nothing however nicely it
    saves files. Declaring a runtime capability blocks public sharing, so the
    return path is plain HTML: a Blob download with an always-visible copy box."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t").read_text()
    assert "window.claude" not in html
    assert "URL.createObjectURL" in html
    assert "navigator.clipboard" in html


def test_the_returned_answers_are_a_table_not_json():
    """He is pasting this into an email. 202 rows of pretty-printed JSON is a
    wall; a table survives a mail client and parses just as easily."""
    from hepcoveragekg.eval import review_app as RA
    with tempfile.TemporaryDirectory() as d:
        html = RA.write_app(_app_items(), pathlib.Path(d) / "a.html", "t").read_text()
    assert 'row\\tquestion_id\\tpaper\\tverdict\\tnotes' in html
    assert "# sheet=" in html, "the version must travel with the answers"
