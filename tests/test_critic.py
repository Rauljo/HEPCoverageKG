"""The candidate critic (D-060).

Two kinds of test here, and the second kind is the point. The mechanical ones
check parsing, chunking and ordering. The CONTROLS check that the critic is
capable of the two answers that make its verdicts mean anything -- "keep them
all" and "drop them all". Both were earned:

  a critic that can never say all-keep is hedging, not judging. AgentRivet's
      Claude-Opus critic never once returned "approved", even when instructed to
      (vault/ideas/multi-agent-extension.md).

  a critic that can never say all-drop invents relevance to fill a quota. Ranked
      order puts the weakest candidates together in the last chunk, so a model
      that feels obliged to keep something per chunk inflates every count
      downstream by roughly one item per chunk.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hepcoveragekg.query import critic as C


def hit(entity_id: str, label: str, kind: str = "generator", facets=None):
    return SimpleNamespace(entity_id=entity_id, label=label, kind=kind,
                           facets=facets or [])


def replying(*payloads):
    """A model that returns one prepared reply per call."""
    state = {"i": 0, "prompts": []}

    def chat(messages):
        state["prompts"].append(messages[0]["content"])
        payload = payloads[min(state["i"], len(payloads) - 1)]
        state["i"] += 1
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    chat.state = state
    return chat


def verdicts_for(n, rung=C.EXACT, start=1):
    return {"verdicts": [{"i": i, "why": "x", "rung": rung}
                         for i in range(start, start + n)]}


# -- the controls ----------------------------------------------------------

def test_it_can_keep_everything():
    """If the critic cannot say "all sixty belong", it is shrinking sets rather
    than judging them, and nothing downstream is trustworthy."""
    hits = [hit(f"e{i}", f"Pythia 8.2{i:02d}") for i in range(15)]
    review = C.judge_candidates(replying(verdicts_for(15, C.EXACT)),
                                "which analyses use Pythia 8?", "Pythia", hits)
    assert len(review.kept_ids) == 15
    assert review.tally[C.UNRELATED] == 0


def test_it_can_drop_everything():
    """The mirror. Ranked order puts the weakest candidates in the last chunk
    together, so a per-chunk keep quota would inflate every count."""
    hits = [hit(f"e{i}", f"single top {i}") for i in range(15)]
    review = C.judge_candidates(replying(verdicts_for(15, C.UNRELATED)),
                                "which analyses use Pythia 8?", "Pythia", hits)
    assert review.kept_ids == []
    assert review.tally[C.UNRELATED] == 15


# -- the asymmetry ---------------------------------------------------------

def test_a_missing_verdict_is_kept_not_dropped():
    """47 verdicts for 60 candidates must not become 13 silent drops.

    Flagging is recoverable: the candidate stays in the set and in the trace.
    A drop the model never actually made is invisible, which is the failure this
    project keeps paying for.
    """
    hits = [hit(f"e{i}", f"thing {i}") for i in range(10)]
    review = C.judge_candidates(replying(verdicts_for(4)), "q", "s", hits)
    assert len(review.kept_ids) == 10, "the six unjudged must survive"
    assert review.defaulted == 6
    assert all(v.defaulted for v in review.verdicts[4:])


def test_an_unparseable_reply_keeps_everything():
    """A broken chunk loses no candidates and is counted as an error."""
    hits = [hit(f"e{i}", f"thing {i}") for i in range(5)]
    review = C.judge_candidates(replying("the model rambled"), "q", "s", hits)
    assert len(review.kept_ids) == 5 and review.defaulted == 5


def test_one_bad_chunk_does_not_lose_the_others():
    def chat(messages):
        # Prompt indices restart at 1 in every chunk, so the second chunk is
        # identified by its content, not by its numbering.
        if "thing 7" in messages[0]["content"]:
            raise RuntimeError("endpoint fell over")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(verdicts_for(5, C.UNRELATED))))],
            usage=None)

    hits = [hit(f"e{i}", f"thing {i}") for i in range(10)]
    review = C.judge_candidates(chat, "q", "s", hits, chunk=5)
    assert review.errors == 1
    assert review.tally[C.UNRELATED] == 5, "the good chunk still counts"
    assert review.defaulted == 5, "the failed chunk defaults to kept"


# -- rungs -----------------------------------------------------------------

def test_broader_is_kept_but_recorded_separately():
    """`broader` is the right answer to some questions (S-68), so it survives --
    but a binary flag would lose the direction of the error, which is the
    diagnosis: Tier B errs too coarse, the supervisor's Q5 too fine."""
    hits = [hit("e1", "Pythia 8.230"), hit("e2", "Pythia"), hit("e3", "Herwig 7")]
    review = C.judge_candidates(replying({"verdicts": [
        {"i": 1, "why": "the exact version", "rung": "exact"},
        {"i": 2, "why": "no version given", "rung": "broader"},
        {"i": 3, "why": "a different generator", "rung": "unrelated"},
    ]}), "which analyses use Pythia 8?", "Pythia", hits)
    assert review.kept_ids == ["e1", "e2"]
    assert review.tally == {C.EXACT: 1, C.BROADER: 1, C.UNRELATED: 1}


def test_an_unknown_rung_is_not_a_drop():
    hits = [hit("e1", "x")]
    review = C.judge_candidates(
        replying({"verdicts": [{"i": 1, "rung": "maybe", "why": ""}]}),
        "q", "s", hits)
    assert review.kept_ids == ["e1"] and review.defaulted == 1


# -- what the model is shown -----------------------------------------------

def test_the_question_and_the_search_text_are_both_shown():
    """They differ, and the difference is where `broader` lives: the planner
    searches the wider "Pythia" on purpose, so `Pythia 6` matches the search and
    not the question."""
    chat = replying(verdicts_for(1))
    C.judge_candidates(chat, "how many analyses use a Pythia 8 shower?",
                       "Pythia", [hit("e1", "Pythia 6.428")])
    prompt = chat.state["prompts"][0]
    assert "Pythia 8 shower" in prompt, "the question defines relevance"
    assert "Pythia" in prompt, "the search text explains why the junk is here"
    assert "each on its own" in prompt, "independent judgement, not a ranking"


def test_facets_are_shown_when_present():
    chat = replying(verdicts_for(1))
    C.judge_candidates(chat, "q", "s", [hit("e1", "b-tagged jet", "detector_object",
                                            facets=["BJet"])])
    assert "BJet" in chat.state["prompts"][0]


def test_chunking_splits_the_calls():
    hits = [hit(f"e{i}", f"thing {i}") for i in range(60)]
    chat = replying(verdicts_for(15))
    review = C.judge_candidates(chat, "q", "s", hits, chunk=15)
    assert review.calls == 4
    assert len(review.verdicts) == 60


# -- ordering --------------------------------------------------------------

def test_verdicts_come_back_in_retrieval_order_even_when_shuffled():
    """Only what the MODEL saw depends on the seed. The returned order never
    does, so a ranked run and a shuffled run are directly comparable."""
    hits = [hit(f"e{i}", f"thing {i}") for i in range(10)]
    payload = {"verdicts": [{"i": i, "why": "", "rung": "exact"}
                            for i in range(1, 11)]}
    ranked = C.judge_candidates(replying(payload), "q", "s", hits)
    shuffled = C.judge_candidates(replying(payload), "q", "s", hits, seed=7)
    ids = [f"e{i}" for i in range(10)]
    assert [v.entity_id for v in ranked.verdicts] == ids
    assert [v.entity_id for v in shuffled.verdicts] == ids


def test_a_shuffle_actually_reorders_what_the_model_sees():
    hits = [hit(f"e{i}", f"label-{i}") for i in range(10)]
    ranked = replying(verdicts_for(10))
    shuffled = replying(verdicts_for(10))
    C.judge_candidates(ranked, "q", "s", hits)
    C.judge_candidates(shuffled, "q", "s", hits, seed=3)
    assert ranked.state["prompts"][0] != shuffled.state["prompts"][0]


def test_a_shuffled_verdict_lands_on_the_candidate_the_model_meant():
    """The bookkeeping that makes the bias measurement trustworthy: verdict `i`
    refers to the shuffled position, not the retrieval position."""
    hits = [hit(f"e{i}", f"thing {i}") for i in range(4)]
    seen = {}

    def chat(messages):
        for line in messages[0]["content"].splitlines():
            index, _, rest = line.partition(". ")
            if index.isdigit() and "thing" in rest:
                seen[int(index)] = rest.split("[")[0].strip()
        first = min(seen)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"verdicts": [
                {"i": first, "why": "", "rung": "unrelated"}]})))], usage=None)

    review = C.judge_candidates(chat, "q", "s", hits, seed=11)
    dropped = [v.entity_id for v in review.verdicts if not v.kept]
    assert len(dropped) == 1
    # whichever label sat at prompt position 1 is the one that got dropped
    assert seen[1].replace("thing ", "e") == dropped[0]


# -- the stopping rule -----------------------------------------------------

def test_a_full_list_that_is_still_relevant_at_the_tail_widens():
    """The case a keep-count rule gets exactly backwards: every hit relevant and
    the list came back full is the MOST truncated case there is."""
    hits = [hit(f"e{i}", f"jet energy scale {i}") for i in range(60)]
    review = C.judge_candidates(replying(verdicts_for(15, C.EXACT)),
                                "q", "jet energy scale", hits, chunk=15)
    assert review.tail_keep_rate() == 1.0
    assert C.should_widen(review, hits_returned=60, limit=60) is True


def test_a_list_whose_tail_is_junk_stops():
    """`top squark` is already "single top" by rank 20; rank 61 is worse."""
    hits = [hit(f"e{i}", f"thing {i}") for i in range(30)]
    review = C.judge_candidates(
        replying(verdicts_for(15, C.EXACT), verdicts_for(15, C.UNRELATED)),
        "q", "top squark", hits, chunk=15)
    assert review.tail_keep_rate() == 0.0
    assert C.should_widen(review, hits_returned=30, limit=30) is False


def test_a_list_that_did_not_fill_never_widens():
    """43 hits against a limit of 60 means there is no rank 61 to fetch, however
    relevant the tail is."""
    hits = [hit(f"e{i}", f"thing {i}") for i in range(43)]
    review = C.judge_candidates(replying(verdicts_for(15, C.EXACT)),
                                "q", "s", hits, chunk=15)
    assert C.should_widen(review, hits_returned=43, limit=60) is False


def test_no_candidates_is_not_an_error():
    review = C.judge_candidates(replying(verdicts_for(1)), "q", "s", [])
    assert review.verdicts == [] and review.calls == 0
    assert review.tail_keep_rate() is None


# -- the facets site: a label reader, never a filter ------------------------

ABCD_ROWS = [
    {"paper_id": "2004.01678", "matched": "ABCD",
     "evidence_labels": "['Modified ABCD estimate']"},
    {"paper_id": "2011.07812", "matched": "ABCD",
     "evidence_labels": "['ABCD data-driven background estimation method']"},
    {"paper_id": "2012.01581", "matched": "ABCD",
     "evidence_labels": "['ABCD-style ratio method using eight regions (A-H)']"},
]


def test_the_facet_reader_shows_the_papers_own_words():
    """The tag is a family; the label is what the paper did. Six papers tagged
    ABCD describe six different methods, and the question decides which count."""
    chat = replying(verdicts_for(3))
    C.read_facet_labels(chat, "which analyses use the standard four-region ABCD?",
                        "background_methods", ["ABCD"], ABCD_ROWS)
    prompt = chat.state["prompts"][0]
    assert "eight regions" in prompt, "the distinguishing detail must be shown"
    assert "four-region" in prompt, "and the question that discriminates on it"


def test_the_facet_reader_never_narrows_the_result():
    """Unlike the search site. Every match here is genuine -- the tag is right
    and only the variant differs -- and for a coverage map those variants ARE
    the finding."""
    review = C.read_facet_labels(
        replying({"verdicts": [
            {"i": 1, "why": "modified, not standard", "rung": "broader"},
            {"i": 2, "why": "the standard method", "rung": "exact"},
            {"i": 3, "why": "eight regions, not four", "rung": "broader"},
        ]}), "q", "background_methods", ["ABCD"], ABCD_ROWS)
    summary = C.facet_summary(review)
    assert "3 papers carry this tag" in summary
    assert "1 match the question as asked" in summary
    assert "No paper has been removed" in summary


def test_a_broader_verdict_is_a_lead_not_a_demotion():
    """It names HOW the paper differs, which is the next query."""
    review = C.read_facet_labels(
        replying({"verdicts": [
            {"i": 1, "why": "eight regions, not four", "rung": "broader"},
            {"i": 2, "why": "standard", "rung": "exact"},
            {"i": 3, "why": "two-dimensional sideband", "rung": "broader"},
        ]}), "q", "background_methods", ["ABCD"], ABCD_ROWS)
    summary = C.facet_summary(review)
    assert "eight regions, not four" in summary
    assert "coverage information" in summary
    assert "facet_entities" in summary, "it says where to look next"


def test_the_facet_reader_defaults_missing_verdicts_the_same_way():
    review = C.read_facet_labels(replying("nonsense"), "q",
                                 "background_methods", ["ABCD"], ABCD_ROWS)
    assert review.defaulted == 3
    assert len(review.verdicts) == 3


def test_no_rows_is_not_a_call():
    review = C.read_facet_labels(replying(verdicts_for(1)), "q", "f", ["v"], [])
    assert review.calls == 0 and review.verdicts == []
