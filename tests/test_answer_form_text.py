"""answer_form counts ids in the text, not a system-filled field (D-145)."""
from types import SimpleNamespace
from hepcoveragekg.eval import scoring


def test_print_rate_reads_the_text():
    q = SimpleNamespace(shape="set")
    a = SimpleNamespace(text="The analyses are 2001.00001 and 2002.00002.", named_ids=0, cited="")
    out = scoring.answer_form(q, a)
    assert out["named_ids"] == 2 and out["answer_names_papers"] == 1.0


def test_no_ids_anywhere_is_not_a_print():
    q = SimpleNamespace(shape="set")
    a = SimpleNamespace(text="They use b-jets.", named_ids=0, cited="")
    assert scoring.answer_form(q, a)["answer_names_papers"] == 0.0
