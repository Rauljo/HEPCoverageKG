"""
Tests for the reference reader — the independent second opinion from the papers.

The load-bearing test in this file is the quote verifier. Everything the reader
produces is trusted only because a claimed citation is checked mechanically, so
if the verifier is wrong the whole gold is wrong, in whichever direction it errs:

  too strict  -> true quotes rejected -> correct answers suppressed -> the gold
                 is falsely conservative, and nothing about it looks wrong
  too loose   -> fabrications accepted -> the gold is a model's imagination with
                 a citation stapled to it

Both are tested here explicitly, including the exact case that broke the first
implementation (a model dropping LaTeX from an otherwise verbatim quote).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hepcoveragekg.eval import reader as R
from hepcoveragekg.eval import supervisor as S


@pytest.fixture()
def conn(tmp_path):
    """A miniature paper: real markup, a real section split, and a bibliography
    and author list that must never be read."""
    db = tmp_path / "src.db"
    c = sqlite3.connect(db)
    c.executescript(
        r"""
        CREATE TABLE source_snapshot (source_hash TEXT PRIMARY KEY, paper_id TEXT);
        CREATE TABLE source_block (
            source_hash TEXT, block_id TEXT, kind TEXT, block_order INTEGER,
            section_title TEXT, text TEXT);
        INSERT INTO source_snapshot VALUES ('h1','p1'), ('h2','p2');

        INSERT INTO source_block VALUES
          ('h1','b1','paragraph',1,'Event selection',
           'Its magnitude, $p_{\mathrm{T}}^{\text{miss}}$ , is required to be less than 40  GeV , which results in a signal efficiency of ${\approx}80\%$ with the $\mathup{{{t}}}$ rejection factor of ${\approx}4.8$ .'),
          ('h1','b2','paragraph',2,'Background estimation',
           'The ABCD method is used to estimate the multijet contribution from four non-overlapping regions defined by two uncorrelated variables.'),
          ('h1','b3','bibliography',3,'References',
           'A. Author et al., A novel deep neural network for boosted top quark tagging, JHEP 2019.'),
          ('h1','b4','paragraph',4,'Appendix A The CMS Collaboration',
           'Yerevan Physics Institute, Yerevan, Armenia. A. Person, B. Person.'),
          ('h1','b5','paragraph',5,'Acknowledgments',
           'We thank the technical staff at CERN for their contributions.'),
          ('h2','b1','paragraph',1,'Event selection',
           'Events are required to contain exactly two electrons or exactly two muons.');
        """
    )
    c.commit()
    c.close()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


# --- the verifier -------------------------------------------------------------


def test_verbatim_quote_verifies(conn):
    assert R.verify_quote(
        conn, "p1",
        r"is required to be less than 40  GeV , which results in a signal efficiency")


def test_true_quote_with_the_latex_dropped_still_verifies(conn):
    """The case that broke the first implementation.

    Substring matching rejected this, which is the dangerous direction: a true
    quote rejected means a correct "yes" is silently discarded and the gold
    quietly under-reports.
    """
    assert R.verify_quote(
        conn, "p1",
        "is required to be less than 40 GeV, which results in a signal efficiency "
        "of approximately 80%")


def test_fabricated_quote_is_rejected(conn):
    """Plausible, on-topic, and present nowhere in the paper's own prose. The
    words appear only in a bibliography entry, which is never read."""
    assert not R.verify_quote(
        conn, "p1",
        "The analysis uses a novel deep neural network to identify boosted top quarks")


def test_a_quote_from_the_wrong_paper_is_rejected(conn):
    assert not R.verify_quote(
        conn, "p2",
        r"is required to be less than 40  GeV , which results in a signal efficiency")


def test_short_quotes_cannot_support_a_claim(conn):
    """"40 GeV" appears in the paper and proves nothing."""
    assert not R.verify_quote(conn, "p1", "40 GeV")
    assert not R.verify_quote(conn, "p1", "")
    assert not R.verify_quote(conn, "p1", "the ABCD method")


def test_a_run_shorter_than_the_threshold_does_not_count(conn):
    """Seven matching words then a fabricated tail. Accepting this would let a
    model anchor on a real phrase and invent the claim."""
    assert not R.verify_quote(
        conn, "p1", "The ABCD method is used to estimate lepton fake rates by inversion")


# --- what the reader is allowed to see ----------------------------------------


def test_passages_drop_bibliography_authors_and_acknowledgments(conn):
    text = " ".join(p.text for p in R.passages(conn, "p1"))
    assert "ABCD method" in text
    assert "JHEP" not in text                 # bibliography
    assert "Yerevan" not in text              # author list
    assert "technical staff" not in text      # acknowledgments


def test_section_routing_restricts_what_is_read(conn):
    got = R.passages(conn, "p1", ("background",))
    assert len(got) == 1 and "ABCD" in got[0].text


def test_routing_picks_sections_from_the_question(conn):
    assert "background" in R.route("Which analyses use an ABCD background estimate?")
    assert "systematic" in R.route("What systematic uncertainties are considered?")
    sel = R.route("Which searches select b-jets and missing transverse momentum?")
    assert "selection" in sel or "object" in sel


def test_a_question_with_no_route_falls_through_to_the_whole_paper(conn):
    assert R.route("What is the meaning of this analysis?") == ()


# --- chunking -----------------------------------------------------------------


def test_chunks_combine_whole_sections_but_never_split_one(conn):
    """The rule that a careless version got backwards: flushing on every section
    change turned a 4-window paper into 16."""
    out = R.chunks(R.passages(conn, "p1"), limit=10_000)
    assert len(out) == 1, "two small sections must share one window"
    assert "40  GeV" in out[0].text and "ABCD" in out[0].text


def test_an_oversized_section_is_split_with_overlap():
    big = R.Passage("p1", "Results", "word " * 20_000)
    out = R.chunks([big], limit=10_000)
    assert len(out) > 1
    assert all(c.chars <= 10_000 for c in out)


# --- reply parsing ------------------------------------------------------------


def test_parse_reply_reads_a_clean_answer():
    answer, quote, why = R.parse_reply(
        '{"answer": "yes", "quote": "the ABCD method is used", "why": "stated directly"}')
    assert answer is True and quote == "the ABCD method is used" and why


def test_parse_reply_tolerates_surrounding_prose():
    answer, _, _ = R.parse_reply('Sure!\n{"answer": "no", "quote": "", "why": "absent"}\nHope that helps')
    assert answer is False


def test_unparseable_replies_are_none_not_no():
    """A broken call and a genuine negative are different things. Collapsing them
    would turn API failures into evidence of absence."""
    assert R.parse_reply("")[0] is None
    assert R.parse_reply("I could not find it.")[0] is None
    assert R.parse_reply('{"answer": "maybe"}')[0] is None


# --- consensus ----------------------------------------------------------------


def _v(answer, verified):
    return R.Verdict(paper_id="p1", qid="q", answer=answer, quote="x" * 40,
                     quote_verified=verified)


def test_a_yes_without_a_verified_quote_does_not_count():
    assert _v(True, True).supported
    assert not _v(True, False).supported
    assert not _v(False, True).supported


def test_unanimous_verdicts_become_the_answer():
    c = R.Consensus("p1", "q", [_v(True, True)] * 3)
    assert c.unanimous and c.answer is True


def test_a_split_vote_is_flagged_rather_than_resolved():
    """Taking a majority would hide precisely the cases worth a human look."""
    c = R.Consensus("p1", "q", [_v(True, True), _v(False, False), _v(True, True)])
    assert not c.unanimous
    assert c.answer is None


# --- scope is mechanical ------------------------------------------------------


def test_scope_comes_from_the_question_text_not_the_gold():
    """Q8's gold names one paper; the QUESTION names none. Scoping to the gold
    would make the check circular -- it could only ever confirm the answer."""
    q8 = next(q for q in S.SUPERVISOR_QUESTIONS if q["qid"] == "gf-08")
    assert S.paper_scope(q8["text"]) is None
    assert q8["gabriel_gold"]["papers"] == ["2001.06899"]


def test_questions_naming_a_paper_are_scoped_to_it():
    assert S.paper_scope("Why did 2001.06899 choose 40 GeV for the MET cut?") == ["2001.06899"]
    assert S.paper_scope("Which regions of 2006.05880 require at least three leptons?") \
        == ["2006.05880"]


def test_the_sweep_and_single_paper_split_is_what_we_expect():
    sweep = {r["qid"] for r in S.sweep_questions()}
    single = {r["qid"] for r in S.single_paper_questions()}
    assert sweep == {"gf-01", "gf-02", "gf-03", "gf-04", "gf-05", "gf-07", "gf-08"}
    assert single == {"gf-06", "gf-09", "gf-10", "gf-11", "gf-12", "gf-13", "gf-14", "gf-15"}


def test_his_gold_is_recorded_but_never_loaded_as_truth():
    """Loading it as truth would score agreement with his pipeline and call it
    accuracy. It is provenance, and the reader fills the truth."""
    for r in S.build_records():
        assert r["truth"]["kind"] == "none"
        assert r["truth_source"] == "none"
        assert "gabriel_gold" in r["provenance"]


def test_the_disputed_tier4_premises_are_flagged():
    disputed = {r["qid"] for r in S.build_records() if r["provenance"]["premise_disputed"]}
    assert disputed == {"gf-12", "gf-13", "gf-14"}


# --- the whole pipeline, against a stub endpoint -------------------------------


class _StubClient:
    """An OpenAI-shaped client that replays canned replies.

    Lets the orchestration — streaming, the cascade, repeats, the circuit
    breaker — be tested without a GPU. Every DIAS failure so far has been in
    this layer rather than in the prompt, so it is the part worth pinning.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.chat = self                      # client.chat.completions.create
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _run(conn, monkeypatch, replies, records, papers_for, tmp_path, **kw):
    import asyncio

    from hepcoveragekg.eval import reader as RR

    client = _StubClient(replies)
    monkeypatch.setattr(RR, "_client", lambda: (client, "stub-model"))
    out = tmp_path / "r.jsonl"
    meta = asyncio.run(RR.run(conn, records, papers_for, out, **kw))
    lines = [__import__("json").loads(l) for l in out.read_text().splitlines()]
    return meta, lines, client


ABCD_Q = [{"qid": "q-abcd", "text": "Which analyses use an ABCD background estimate?"}]


def test_a_verified_yes_is_recorded_with_its_quote(conn, monkeypatch, tmp_path):
    reply = ('{"answer":"yes","quote":"The ABCD method is used to estimate the multijet '
             'contribution from four non-overlapping regions","why":"stated"}')
    meta, lines, _ = _run(conn, monkeypatch, [reply], ABCD_Q,
                          lambda q: ["p1"], tmp_path, repeats=3)
    assert len(lines) == 1
    assert lines[0]["answer"] is True
    assert lines[0]["unanimous"] and "ABCD method" in lines[0]["quote"]
    assert meta["yes"] == 1


def test_a_yes_with_a_fabricated_quote_becomes_a_no(conn, monkeypatch, tmp_path):
    """The whole design rests on this: an unverifiable citation is not evidence."""
    reply = ('{"answer":"yes","quote":"A novel deep neural network identifies boosted '
             'top quarks in this analysis","why":"stated"}')
    _, lines, _ = _run(conn, monkeypatch, [reply], ABCD_Q,
                       lambda q: ["p1"], tmp_path, repeats=3)
    assert lines[0]["answer"] is False, "a fabricated quote must not carry a yes"


def test_disagreement_is_reported_not_majority_voted(conn, monkeypatch, tmp_path):
    good = ('{"answer":"yes","quote":"The ABCD method is used to estimate the multijet '
            'contribution from four non-overlapping regions","why":"stated"}')
    bad = '{"answer":"no","quote":"","why":"not shown"}'
    _, lines, _ = _run(conn, monkeypatch, [good, bad, good], ABCD_Q,
                       lambda q: ["p1"], tmp_path, repeats=3, cascade=False)
    assert len(lines[0]["verdicts"]) == 3, "must actually have read three times"
    assert lines[0]["answer"] is None
    assert lines[0]["unanimous"] is False


def test_the_cascade_stops_once_a_verified_yes_is_found(conn, monkeypatch, tmp_path):
    """gf-02 routes to the background sections. A yes there must not trigger the
    whole-paper escalation — that saving is the point of the cascade."""
    reply = ('{"answer":"yes","quote":"The ABCD method is used to estimate the multijet '
             'contribution from four non-overlapping regions","why":"stated"}')
    _, lines, client = _run(conn, monkeypatch, [reply], ABCD_Q,
                            lambda q: ["p1"], tmp_path, repeats=2, cascade=True)
    assert lines[0]["answer"] is True
    assert lines[0]["escalated"] is False
    assert len(client.calls) == 2, "one window, two repeats; no escalation"


def test_a_no_in_the_routed_sections_escalates_to_the_whole_paper(conn, monkeypatch, tmp_path):
    _, lines, client = _run(conn, monkeypatch, ['{"answer":"no","quote":"","why":"absent"}'],
                            ABCD_Q, lambda q: ["p1"], tmp_path, repeats=1, cascade=True)
    assert lines[0]["answer"] is False
    assert lines[0]["escalated"] is True, "a no must be retried against the full text"
    assert len(client.calls) > 1


def test_api_failures_are_recorded_as_unknown_not_as_no(conn, monkeypatch, tmp_path):
    """An endpoint that dies must not silently become evidence of absence."""
    _, lines, _ = _run(conn, monkeypatch, [RuntimeError("connection reset")],
                       ABCD_Q, lambda q: ["p1"], tmp_path, repeats=2, cascade=False)
    assert lines[0]["verdicts"], "an empty verdict list would pass the next line vacuously"
    assert lines[0]["answer"] is None
    assert all(v["answer"] is None for v in lines[0]["verdicts"])


def test_output_is_streamed_so_a_dead_run_keeps_its_work(conn, monkeypatch, tmp_path):
    reply = ('{"answer":"yes","quote":"The ABCD method is used to estimate the multijet '
             'contribution from four non-overlapping regions","why":"stated"}')
    meta, lines, _ = _run(conn, monkeypatch, [reply], ABCD_Q,
                          lambda q: ["p1", "p2"], tmp_path, repeats=1, cascade=False)
    assert len(lines) == 2
    assert (tmp_path / "r.meta.json").exists(), "meta is written after the file closes"
    assert meta["reads"] == 2


# --- extraction mode: the questions that ask WHAT and WHY ----------------------


def test_extraction_mode_returns_the_answer_not_just_yes():
    """gf-12 asks "what is the observed 95% CL limit on the stop mass". A yes/no
    reply throws away the number that makes it an answer — and the number is
    exactly what gets compared against his gold and against the graph."""
    found, quotes, answer = R.parse_reply(
        '{"found": true, "answer": "875 GeV", "quote": "Masses of the stop2 up to 875 GeV '
        'are excluded at 95% CL"}', mode=R.EXTRACTION)
    assert found is True
    assert answer == "875 GeV"
    # a list now -- a question asking for several things is answered by several
    # sentences -- but a singular "quote" from an older reply still parses
    assert isinstance(quotes, list) and "875 GeV" in quotes[0]


def test_extraction_not_found_is_a_real_answer():
    found, quotes, answer = R.parse_reply(
        '{"found": false, "answer": "", "quote": ""}', mode=R.EXTRACTION)
    assert found is False and not answer and not quotes


def test_extraction_tolerates_a_stringly_typed_found_flag():
    assert R.parse_reply('{"found": "true", "answer": "875 GeV", "quote": "x"}',
                         mode=R.EXTRACTION)[0] is True
    assert R.parse_reply('{"found": "no", "answer": "", "quote": ""}',
                         mode=R.EXTRACTION)[0] is False


def test_extraction_unparseable_is_none_not_false():
    assert R.parse_reply("", mode=R.EXTRACTION)[0] is None
    assert R.parse_reply('{"answer": "875 GeV"}', mode=R.EXTRACTION)[0] is None


def test_the_two_modes_read_different_fields():
    """Existence keys on "answer": yes|no; extraction keys on "found". Feeding a
    reply to the wrong parser must fail loudly rather than half-work."""
    existence_reply = '{"answer": "yes", "quote": "x" ,"why": "z"}'
    extraction_reply = '{"found": true, "answer": "875 GeV", "quote": "x"}'
    assert R.parse_reply(existence_reply, mode=R.EXISTENCE)[0] is True
    assert R.parse_reply(existence_reply, mode=R.EXTRACTION)[0] is None
    assert R.parse_reply(extraction_reply, mode=R.EXTRACTION)[0] is True
    assert R.parse_reply(extraction_reply, mode=R.EXISTENCE)[0] is None


def test_extraction_prompt_asks_for_the_answer_and_forbids_outside_knowledge():
    prompt = R._read_prompt("What is the limit?", R.Passage("p1", "Results", "text"),
                            mode=R.EXTRACTION)
    assert '"found"' in prompt and '"answer"' in prompt
    assert "none of it is admissible" in prompt   # no answering from physics knowledge
    assert "EXACTLY" in prompt                 # the quote must be copied


def test_a_verified_extraction_is_recorded_with_its_value(conn, monkeypatch, tmp_path):
    reply = ('{"found": true, "answer": "approximately 80%", "quote": "is required to be '
             'less than 40 GeV, which results in a signal efficiency of approximately 80%"}')
    q = [{"qid": "gf-10", "text": "What signal efficiency does the pT^miss < 40 GeV cut retain?"}]
    _, lines, _ = _run(conn, monkeypatch, [reply], q, lambda _: ["p1"], tmp_path,
                       repeats=2, cascade=False, mode=R.EXTRACTION)
    assert lines[0]["answer"] is True
    assert lines[0]["answers"] == ["approximately 80%"]


def test_an_extraction_with_a_fabricated_quote_is_discarded(conn, monkeypatch, tmp_path):
    """Same rule as existence mode: an unverifiable citation is not evidence,
    however plausible the extracted value looks."""
    reply = ('{"found": true, "answer": "875 GeV", "quote": "A novel deep neural network '
             'identifies boosted top quarks in this analysis"}')
    q = [{"qid": "gf-12", "text": "What is the observed 95% CL limit?"}]
    _, lines, _ = _run(conn, monkeypatch, [reply], q, lambda _: ["p1"], tmp_path,
                       repeats=2, cascade=False, mode=R.EXTRACTION)
    assert lines[0]["answer"] is False
    assert lines[0]["answers"] == []


def test_no_cascade_still_reads_the_whole_paper(conn, monkeypatch, tmp_path):
    """The bug this pins: with cascade off, an earlier version fell through both
    branches and read NOTHING, returning an empty consensus. The single-paper
    job runs with --no-cascade, so all seven deep reads would have produced no
    data while the run reported success."""
    reply = ('{"answer":"yes","quote":"The ABCD method is used to estimate the multijet '
             'contribution from four non-overlapping regions","why":"stated"}')
    _, lines, client = _run(conn, monkeypatch, [reply], ABCD_Q,
                            lambda q: ["p1"], tmp_path, repeats=2, cascade=False)
    assert client.calls, "no LLM call was made at all"
    assert lines[0]["verdicts"], "no verdicts recorded"
    assert lines[0]["answer"] is True


def test_a_question_the_router_cannot_place_still_gets_read(conn, monkeypatch, tmp_path):
    """route() returns () for an unroutable question. That must mean "read
    everything", not "read nothing"."""
    q = [{"qid": "q-odd", "text": "What is the meaning of this analysis?"}]
    assert R.route(q[0]["text"]) == ()
    _, lines, client = _run(conn, monkeypatch, ['{"answer":"no","quote":"","why":"n/a"}'],
                            q, lambda _: ["p1"], tmp_path, repeats=1, cascade=True)
    assert client.calls, "an unroutable question must fall through to the full read"
    assert lines[0]["verdicts"]


def test_a_dead_endpoint_never_becomes_evidence_of_absence():
    """The failure this guards is silent and corpus-wide: `supported` is False
    for an errored verdict exactly as it is for a real negative, so deriving the
    consensus from it reported "this paper does not do X" for every paper the
    outage touched."""
    errored = [R.Verdict("p1", "q", None) for _ in range(3)]
    c = R.Consensus("p1", "q", errored)
    assert c.usable == []
    assert c.answer is None, "no successful call means unknown, not no"

    real_no = [R.Verdict("p1", "q", False) for _ in range(3)]
    assert R.Consensus("p1", "q", real_no).answer is False


def test_one_failed_sample_does_not_veto_the_others():
    """Two clean agreeing samples plus one dropped call is still an answer;
    discarding the whole read would throw away good work over a transient."""
    good = R.Verdict("p1", "q", True, quote="x" * 40, quote_verified=True)
    c = R.Consensus("p1", "q", [good, R.Verdict("p1", "q", None), good])
    assert len(c.usable) == 2
    assert c.answer is True


# --- the failure that reported itself as healthy -------------------------------


def test_windows_fit_the_served_context():
    """The first real run lost ~55% of its calls to
        400 - This model's maximum context length is 8192 tokens
    because CHUNK_CHARS was set from an ASSUMED 4 chars/token. Measured on this
    corpus with the served tokenizer: 3.46. This pins the arithmetic so the
    window can never again be sized by guesswork."""
    served_context = 8192
    budget = (R.CHUNK_CHARS / R.CHARS_PER_TOKEN) + R.MAX_COMPLETION_TOKENS + 500
    assert budget < served_context, (
        f"a full window needs ~{budget:.0f} tokens against a {served_context} ceiling")
    # and with real headroom, not just barely
    assert budget < 0.8 * served_context


def test_an_api_error_is_not_an_unparseable_reply_is_not_a_no():
    """Three different things the first run collapsed into one. `failed: 0` was
    reported while 55% of calls were 400ing, because `failed` only counted reads
    where EVERY sample died."""
    err = R.Verdict("p1", "q", None, raw="ERROR: BadRequestError: context length")
    junk = R.Verdict("p1", "q", None, raw="I could not find it, sorry.")
    no = R.Verdict("p1", "q", False, raw='{"answer":"no"}')

    assert err.errored and not junk.errored and not no.errored
    assert not err.supported and not junk.supported and not no.supported
    # and none of the three is a supported yes
    assert R.Consensus("p1", "q", [err, err]).answer is None
    assert R.Consensus("p1", "q", [no, no]).answer is False


def test_run_reports_a_call_error_rate(conn, monkeypatch, tmp_path):
    """A run that limps home on a minority of its calls must SAY so in the meta,
    not present a clean set of negatives."""
    meta, lines, _ = _run(conn, monkeypatch, [RuntimeError("boom")], ABCD_Q,
                          lambda q: ["p1"], tmp_path, repeats=2, cascade=False)
    assert meta["call_errors"] > 0
    assert meta["call_error_rate"] == 1.0
    assert meta["call_usable_rate"] == 0.0


# --- consensus is per WINDOW, never pooled across windows ----------------------


def _wv(window, supported, answer=True):
    return R.Verdict("p1", "q", answer, quote="x" * 40, quote_verified=supported,
                     window=window, section=f"s{window}")


def test_finding_the_answer_in_one_window_is_a_yes():
    """The bug that made the first sweep useless. A 23-window paper holds the
    answer in one window; the other 22 correctly say "not in this passage".
    Pooling them made unanimity impossible EXACTLY when the answer was found, so
    every successful read came back as "split" and was counted as not found --
    while papers where nothing was found agreed trivially and reported a
    confident False. Finding the answer was what made the result unusable."""
    verdicts = [_wv(0, True), _wv(0, True), _wv(0, True)]        # the hit
    verdicts += [_wv(w, False, answer=False) for w in range(1, 23) for _ in range(3)]
    c = R.Consensus("p1", "q", verdicts)
    assert c.answer is True
    assert c.best_quote


def test_all_windows_agreeing_it_is_absent_is_a_no():
    verdicts = [_wv(w, False, answer=False) for w in range(5) for _ in range(3)]
    assert R.Consensus("p1", "q", verdicts).answer is False


def test_disagreement_WITHIN_a_window_is_a_split():
    """Repeats of the same window are samples of the same question, so
    disagreement there is real uncertainty and must reach a human."""
    verdicts = [_wv(0, True), _wv(0, False, answer=False), _wv(0, True)]
    c = R.Consensus("p1", "q", verdicts)
    assert c.answer is None and not c.unanimous


def test_a_confident_find_outranks_an_unrelated_window_wobbling():
    """Window 3 found it and quoted it; window 7's samples disagreed about an
    unrelated passage. The evidence still exists."""
    verdicts = [_wv(3, True) for _ in range(3)]
    verdicts += [_wv(7, True), _wv(7, False, answer=False), _wv(7, False, answer=False)]
    assert R.Consensus("p1", "q", verdicts).answer is True


def test_rescore_recovers_answers_from_stored_verdicts(tmp_path):
    """Scoring changes must cost a file read, not 22,000 LLM calls."""
    rec = {
        "qid": "gf-02", "paper_id": "2004.01678", "answer": None, "unanimous": False,
        "quote": "", "verdicts": [
            {"answer": True, "quote": "A modified ABCD estimate of the total background",
             "quote_verified": True, "window": 0, "section": "Background"},
            {"answer": True, "quote": "A modified ABCD estimate of the total background",
             "quote_verified": True, "window": 0, "section": "Background"},
            {"answer": False, "quote": "", "quote_verified": False,
             "window": 1, "section": "Detector"},
        ],
    }
    src = tmp_path / "run.jsonl"
    src.write_text(json.dumps(rec) + "\n")

    info = R.rescore(src)
    assert info["changed"] == 1 and info["yes"] == 1
    out = json.loads(Path(info["out"]).read_text().splitlines()[0])
    assert out["answer"] is True
    assert out["previous_answer"] is None
    assert "ABCD" in out["quote"]


# --- the second pass: does the quote answer the question? ----------------------


def test_support_check_parses_a_verdict():
    """Separate from quote verification, and separate from the reader's own
    answer: a sentence can be genuinely in the paper and still answer something
    else. Measured on gf-08, half the cited sentences did."""
    import asyncio

    class _Stub:
        def __init__(s, reply): s.reply = reply; s.chat = s; s.completions = s; s.seen = []
        async def create(s, **kw):
            s.seen.append(kw)
            class M: content = s.reply
            class C: message = M()
            class R_: choices = [C()]
            return R_()

    ok, why = asyncio.run(R.check_support(
        _Stub('{"supports": false, "why": "the sentence is about muons only"}'),
        "m", "which analyses require 2 electrons OR 2 muons?",
        "exactly two oppositely charged muons, no identified electrons"))
    assert ok is False and "muons" in why

    ok, _ = asyncio.run(R.check_support(
        _Stub('{"supports": true, "why": "states the alternative directly"}'),
        "m", "q", "identified through reconstructed dielectrons or dimuons"))
    assert ok is True


def test_the_judge_never_sees_the_paper_or_the_readers_reasoning():
    """It gets the question and the quote. Showing it the surrounding text would
    let the argument that produced the mistake also excuse it."""
    prompt = R.SUPPORT_PROMPT.format(
        question="Q?", quote="a cited sentence", claim="")
    assert "a cited sentence" in prompt and "Q?" in prompt
    assert "supports" in prompt


def test_a_failed_support_check_downgrades_rather_than_deletes(tmp_path):
    """The read happened and the quote is real; the record of the model reading
    it wrongly IS the measurement, so it is kept and marked."""
    import asyncio

    row = {"qid": "gf-08", "paper_id": "2009.04363", "answer": True,
           "quote": "exactly two oppositely charged muons, no identified electrons",
           "answers": ["muons"], "verdicts": []}
    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps(row) + "\n")

    class _Stub:
        chat = None
        def __init__(s): s.chat = s; s.completions = s
        async def create(s, **kw):
            class M: content = '{"supports": false, "why": "muons only"}'
            class C: message = M()
            class R_: choices = [C()]
            return R_()

    R._client = lambda: (_Stub(), "stub")
    info = asyncio.run(R.verify_supports(src, [{"qid": "gf-08", "text": "e OR mu?"}]))
    out = json.loads(Path(info["out"]).read_text().splitlines()[0])
    assert out["answer"] is False
    assert out["quote_supports"] is False
    assert out["downgraded"]
    assert out["quote"], "the quote is kept -- it is evidence of the misreading"
    assert info["precision"] == 0.0


def test_the_judge_is_asked_about_ONE_paper_not_the_corpus():
    """The first support run downgraded 70 of 72 answers -- 3% precision -- by
    judging corpus-wide questions against single sentences:

        quote: "Distributions are unfolded to the particle level ..."
        judge: "does not specify WHICH measurements have distributions unfolded"

    That quote plainly shows the paper unfolds. The reader prompt reframes to
    "does THIS analysis do it"; the judge prompt did not, so it demanded a
    sentence answer a question about 60 papers at once.
    """
    prompt = R.SUPPORT_PROMPT.format(
        question="Which measurements unfold their distributions?",
        quote="Distributions are unfolded to the particle level.", claim="")
    assert "ONE PARTICULAR PAPER" in prompt
    assert "do not ask whether the sentence names" in prompt
    assert "THIS paper" in prompt


def test_every_sweep_question_has_a_per_paper_form():
    """A sweep IS 60 instances of "does THIS paper do X", and the phrasing
    decides the answer. Same window, same model:

        "Which SUSY searches did their statistics in HistFitter?"  -> no
        "Does this analysis use the HistFitter framework?"         -> yes

    The reader prompt already instructs "does THIS analysis do it" and the model
    anchored on the question's wording regardless. Instructing around a
    corpus-wide question does not work; decomposing it does.
    """
    for r in S.sweep_questions():
        assert r.get("per_paper"), f"{r['qid']} has no per-paper phrasing"
        assert not r["per_paper"].lower().startswith("which "), (
            f"{r['qid']}: still phrased across the corpus")
    # the original is kept, so both are auditable
    for r in S.build_records():
        assert r["provenance"]["question_as_asked"] == r["text"]


def test_single_paper_questions_need_no_rephrasing():
    """They already name their paper, so they are already per-paper."""
    for r in S.single_paper_questions():
        assert r["provenance"]["paper_scope"]


# --- reasoning models -----------------------------------------------------------


def test_thinking_blocks_are_stripped_before_the_json_is_found():
    """A reasoning model emits its chain of thought as ordinary output tokens.
    Served without --reasoning-parser the <think> block arrives inline, and it
    routinely contains braces and draft JSON."""
    raw = ('<think>Could be {"answer":"no"} but the sentence does mention it</think>\n'
           '{"reasoning":"the text states it directly","quote":"a real sentence here","answer":"yes"}')
    answer, quote, _ = R.parse_reply(raw)
    assert answer is True
    assert quote == "a real sentence here"


def test_the_LAST_json_object_wins_not_the_first():
    """A model that reconsiders mid-thought leaves an abandoned draft behind.
    The answer it settled on is the last one."""
    raw = '{"answer":"no","quote":"","why":"first thought"} then {"answer":"yes","quote":"q","why":"settled"}'
    assert R.parse_reply(raw)[0] is True


def test_completion_budget_is_configurable_for_reasoning_models(monkeypatch):
    """500 tokens truncates a reasoning model mid-thought, so the reply never
    reaches its JSON -- which presents as a 100% unparseable rate and looks like
    the model cannot follow the format."""
    import importlib
    monkeypatch.setenv("READER_MAX_TOKENS", "4000")
    reloaded = importlib.reload(R)
    assert reloaded.MAX_COMPLETION_TOKENS == 4000
    monkeypatch.delenv("READER_MAX_TOKENS")
    importlib.reload(reloaded)


# --- multi-condition questions --------------------------------------------------


def test_the_AND_is_computed_over_the_paper_not_per_window():
    """gf-01 scored 2/18 while every single-condition question scored well.
    A ~12,000-char window almost never establishes 'a search' AND 'b-tagged jets'
    AND 'missing transverse momentum' at once, so the honest per-window answer is
    no and the paper came back no -- even though each condition was separately
    present somewhere in the text."""
    q = next(r for r in S.build_records() if r["qid"] == "gf-01")
    assert q["conditions"] and len(q["conditions"]) == 3
    for c in q["conditions"]:
        assert " AND " not in c, f"a condition must ask ONE thing: {c}"


def test_a_missing_condition_is_named_not_just_counted(conn, monkeypatch, tmp_path):
    """A failure should say WHICH part was absent, so 'no' is debuggable."""
    import asyncio

    replies = {
        "search": '{"reasoning":"it is a search","quote":"The ABCD method is used to estimate the multijet contribution","answer":"yes"}',
        "b-tag": '{"reasoning":"nothing here","quote":"","answer":"no"}',
    }

    class _Stub:
        def __init__(s): s.chat = s; s.completions = s
        async def create(s, **kw):
            prompt = kw["messages"][0]["content"]
            body = replies["b-tag"] if "b-tagged" in prompt else replies["search"]
            class M: content = body
            class C: message = M()
            class R_: choices = [C()]
            return R_()

    from hepcoveragekg.eval import reader as RR
    client = _Stub()
    out = asyncio.run(RR.read_conditions(
        conn, client, "stub", "gf-01",
        ["Is this a search?", "Does it use b-tagged jets?"], "p1", repeats=1))
    assert out["answer"] is False
    assert out["missing"] == ["Does it use b-tagged jets?"]
    assert out["conditions"]["Is this a search?"]["answer"] is True


def test_a_multi_condition_claim_is_judged_on_ALL_its_quotes():
    """The paper-level claim is a conjunction, so its evidence is the SET of
    sentences. Handing the judge ONE sentence and the whole three-part question
    guarantees rejection -- one sentence rarely shows all three. That is how four
    gf-01 papers with every condition confirmed were downgraded to False, the
    judge narrating the evidence as it rejected it:

        "mentions a search (not a measurement) and explicitly includes missing
         transverse momentum..."   -> downgraded
    """
    import asyncio

    seen = {}

    class _Stub:
        def __init__(s): s.chat = s; s.completions = s
        async def create(s, **kw):
            seen["prompt"] = kw["messages"][0]["content"]
            class M: content = '{"why":"together they establish it","supports":true}'
            class C: message = M()
            class R_: choices = [C()]
            return R_()

    row = {"qid": "gf-01", "paper_id": "p1", "answer": True,
           "quote": "first sentence only",
           "quotes": {"is it a search": "it is a search for new physics",
                      "b-tagged jets": "b-tagged jets are required",
                      "missing energy": "large missing transverse momentum is required"}}
    import json as _json, tempfile
    from pathlib import Path as _P
    src = _P(tempfile.mkdtemp()) / "r.jsonl"
    src.write_text(_json.dumps(row) + "\n")

    R._client = lambda: (_Stub(), "stub")
    asyncio.run(R.verify_supports(src, [{"qid": "gf-01", "text": "search AND b-jets AND MET?"}]))

    prompt = seen["prompt"]
    assert "b-tagged jets are required" in prompt
    assert "large missing transverse momentum" in prompt
    assert "it is a search for new physics" in prompt, "all three must reach the judge"


def test_the_judge_sees_every_distinct_quote_the_reader_found():
    """Measured on the sweep: 34 of 140 reads produced more than one verified
    quote, and 16 DOWNGRADED reads had other verified quotes the judge never saw.
    best_quote returned the first and the rest were dropped, so a read could be
    rejected on its weakest evidence while its strongest sat unused."""
    row = {"quote": "the first one", "verdicts": [
        {"quote": "the first one", "quote_verified": True},
        {"quote": "a genuinely different sentence establishing the claim", "quote_verified": True},
        {"quote": "unverified, must not appear", "quote_verified": False}]}
    out = R._evidence_for_judge(row)
    assert "the first one" in out
    assert "a genuinely different sentence" in out
    assert "unverified" not in out


def test_repetition_is_not_corroboration():
    """The same sentence recovered by two samples is one piece of evidence, and
    listing it twice would let the judge weigh quantity instead of content."""
    q = "one sentence found by every sample"
    row = {"quote": q, "verdicts": [{"quote": q, "quote_verified": True}] * 3}
    assert R._evidence_for_judge(row).count(q) == 1


def test_a_quote_contained_in_another_is_dropped():
    row = {"quote": "", "verdicts": [
        {"quote": "b-tagged jets are required by the selection", "quote_verified": True},
        {"quote": "b-tagged jets are required", "quote_verified": True}]}
    assert R._evidence_for_judge(row).count("b-tagged jets are required") == 1


def test_a_single_quote_is_not_dressed_up_as_a_list():
    row = {"quote": "only one", "verdicts": [{"quote": "only one", "quote_verified": True}]}
    assert R._evidence_for_judge(row) == "only one"


def test_an_unjudged_item_is_not_rendered_as_a_rejection(tmp_path):
    """`if item["_judge"]` treated None as falsy, so the seven single-paper items
    -- which never go to the judge -- were shown to the reviewer as 'does NOT
    support the claim'. A verdict nobody reached, contradicting our own reader."""
    from hepcoveragekg.eval import review

    items = [
        {"row": 1, "qid": "gf-11", "question": "q", "paper_id": "p1",
         "quote": "a real sentence", "_machine": True, "_judge": None,
         "_why": "our reader answered: X", "_by": "single-paper run"},
        {"row": 2, "qid": "gf-11", "question": "q", "paper_id": "p2",
         "quote": "another sentence", "_machine": True, "_judge": False,
         "_why": "off topic", "_by": "stage-1"},
    ]
    out = tmp_path / "r.html"
    review.write_html(items, out, "t")
    html = out.read_text()
    assert "not checked by our judge" in html
    assert html.count("does NOT support the claim") == 1, "only the truly rejected one"


def test_extraction_keeps_every_verified_sentence():
    """gf-11 asks for an algorithm AND a working point AND its performance. It
    returned the algorithm alone and stopped -- and the other two were in the
    SAME window it had just read."""
    found, quotes, answer = R.parse_reply(
        '{"reasoning":"three parts","quotes":["the algorithm is CSVv2",'
        '"a medium operating point is used","efficiencies are 10 and 60 percent"],'
        '"answer":"CSVv2, medium, 10/60%","found":true}', mode=R.EXTRACTION)
    assert found is True and len(quotes) == 3


def test_a_fabricated_quote_does_not_discard_the_real_ones_beside_it(conn):
    """Each cited sentence is checked on its own."""
    import asyncio

    reply = ('{"reasoning":"two parts","quotes":["The ABCD method is used to estimate '
             'the multijet contribution from four non-overlapping regions",'
             '"a completely invented sentence about neural networks"],'
             '"answer":"ABCD","found":true}')

    class _Stub:
        def __init__(s): s.chat = s; s.completions = s
        async def create(s, **kw):
            class M: content = reply
            class C: message = M()
            class R_: choices = [C()]
            return R_()

    v = asyncio.run(R.read_passage(conn, _Stub(), "m", "q", "question?",
                                   R.Passage("p1", "s", "t"), 0.0, mode=R.EXTRACTION))
    assert v.quote_verified
    assert len(v.quotes) == 1
    assert "ABCD method" in v.quotes[0]


def test_extraction_does_not_stop_at_the_first_window(conn, monkeypatch, tmp_path):
    """EXISTENCE stops on a hit -- one confirmation settles it. EXTRACTION must
    not: the answer's parts can be spread across the paper."""
    reply = ('{"reasoning":"r","quotes":["The ABCD method is used to estimate the '
             'multijet contribution from four non-overlapping regions"],'
             '"answer":"ABCD","found":true}')
    q = [{"qid": "x", "text": "what method?"}]
    _, _, client = _run(conn, monkeypatch, [reply], q, lambda _: ["p1"], tmp_path,
                        repeats=1, cascade=False, mode=R.EXTRACTION)
    calls_extraction = len(client.calls)

    _, _, client2 = _run(conn, monkeypatch, [reply.replace('"quotes"', '"quote_x"')
                                             .replace('"found":true', '"answer":"yes"')],
                         q, lambda _: ["p1"], tmp_path, repeats=1, cascade=False)
    assert calls_extraction >= len(client2.calls), (
        "extraction must read at least as many windows as existence")


def test_merge_unions_what_both_readers_gathered(tmp_path):
    """Recall is a union, not a vote. The 24B needs a concept named in the
    paper's own words; QwQ will reason to a weaker connection. Requiring both to
    agree would discard exactly the evidence the second model was added to
    recover -- the judge downstream is what stops the union being credulous."""
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"qid": "gf-11", "paper_id": "p1",
                             "all_quotes": ["the algorithm is CSVv2"],
                             "answers": ["CSVv2"]}) + "\n")
    b.write_text(json.dumps({"qid": "gf-11", "paper_id": "p1",
                             "all_quotes": ["a medium operating point is used",
                                            "the algorithm is CSVv2"],
                             "answers": ["CSVv2 medium"]}) + "\n")
    out = tmp_path / "m.jsonl"
    info = R.merge_gathered([a, b], out)
    assert info["pairs"] == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert len(row["all_quotes"]) == 2, "the duplicate collapses, the new one survives"
    assert any("medium operating point" in q for q in row["all_quotes"])
    assert row["answer"] is True


def test_merge_marks_a_pair_with_no_evidence_as_false(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps({"qid": "gf-10", "paper_id": "p1",
                             "all_quotes": [], "answers": []}) + "\n")
    out = tmp_path / "m.jsonl"
    R.merge_gathered([a], out)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["answer"] is False and row["all_quotes"] == []


def test_judge_sees_every_gathered_quote_not_just_the_first():
    """The merge is pointless if the judge only reads one of the union.

    A merged row carries no verdicts of its own, so the old code fell through to
    the single-`quote` default -- a three-part claim decided on a third of its
    evidence, which is the exact failure the merge was built to fix."""
    row = {"qid": "gf-11", "paper_id": "p1", "verdicts": [],
           "quote": "the algorithm is CSVv2",
           "all_quotes": ["the algorithm is CSVv2",
                          "a medium operating point is used",
                          "the efficiency is 70%"]}
    evidence = R._evidence_for_judge(row)
    assert evidence.count("\n") == 2
    for q in row["all_quotes"]:
        assert q in evidence


def test_merged_evidence_is_capped_but_far_above_the_sweep_cap():
    row = {"verdicts": [], "quote": "q0",
           "all_quotes": [f"sentence number {i}" for i in range(40)]}
    lines = R._evidence_for_judge(row).splitlines()
    assert len(lines) == R.MAX_MERGED_JUDGE_QUOTES > R.MAX_JUDGE_QUOTES


# Both of these are VERBATIM from run 48367 -- replies the old parser discarded.
# Between them they cost 124 of 366 calls on the QwQ arm and 31 on the 24B arm.

REPLY_LATEX_UNESCAPED = (
    '```json\n{"reasoning": "The text discusses the MET cut.", '
    '"quotes": ["Its magnitude, $p_{\\mathrm{T}}^{\\text{miss}}$ , is required to be '
    'less than 40  GeV , which results in a signal efficiency of ${\\approx}80\\%$ '
    'with the $\\mathup{{{t}}}$ rejection factor of ${\\approx}4.8$ ."], '
    '"answer": "40 GeV, ~80% efficiency.", "found": true}\n```')

REPLY_WITH_DRAFT_FIRST = (
    '<think>Let me draft: {"found": false} -- no wait, the sentence is there.</think>\n'
    '{"reasoning": "found it", "quotes": ["The b jets are identified with the '
    'CSVv2 algorithm at the medium working point."], '
    '"answer": "CSVv2, medium", "found": true}')


def test_latex_in_a_quote_no_longer_destroys_the_reply():
    """`{\\approx}` is brace-free and comes last, so the old regex picked it as
    'the JSON object' and threw the answer away. The loss selected for replies
    quoting cut values and efficiencies -- the content the questions ask for."""
    answer, quotes, text = R.parse_reply(REPLY_LATEX_UNESCAPED, R.EXTRACTION)
    assert answer is True
    assert len(quotes) == 1 and "40  GeV" in quotes[0]
    assert "80%" in text


def test_a_reasoning_model_draft_does_not_win_over_its_conclusion():
    answer, quotes, _ = R.parse_reply(REPLY_WITH_DRAFT_FIRST, R.EXTRACTION)
    assert answer is True, "the <think> draft said false; the conclusion said true"
    assert "CSVv2" in quotes[0]


def test_repairing_escapes_is_idempotent_and_never_breaks_valid_json():
    """The naive fix walks into the second character of an already-correct
    `\\\\mathup` and doubles that, breaking the replies that were right."""
    valid = '{"q": "a $\\\\mathup{x}$ b"}'
    assert json.loads(R._repair_latex_escapes(valid))["q"] == "a $\\mathup{x}$ b"
    once = R._repair_latex_escapes('{"q": "a $\\mathup{x}$ b"}')
    assert R._repair_latex_escapes(once) == once


def test_a_genuinely_truncated_reply_is_still_rejected():
    """Recovery must not turn into invention: a reply cut off mid-sentence has
    no verdict in it, and guessing one would manufacture evidence."""
    assert R.parse_reply('{"quotes": ["the analysis uses a $\\mathup{{{t}}}$ and',
                         R.EXTRACTION)[0] is None


def test_reparse_recovers_without_calling_the_model(tmp_path):
    src = tmp_path / "run.jsonl"
    src.write_text(json.dumps({
        "qid": "gf-13", "paper_id": "p1", "answer": None, "quote": "",
        "all_quotes": [], "answers": [],
        "verdicts": [{"paper_id": "p1", "qid": "gf-13", "answer": None,
                      "quote": "", "quote_verified": False, "quotes": [],
                      "reasoning": "", "answer_text": "", "section": "s",
                      "window": 0, "escalated": False,
                      "raw": REPLY_LATEX_UNESCAPED}]}) + "\n")
    info = R.reparse(src, tmp_path / "out.jsonl")
    assert info["recovered"] == 1 and info["still_unparsed"] == 0
    row = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    assert row["answer"] is True
    assert row["all_quotes"] and "40  GeV" in row["all_quotes"][0]
