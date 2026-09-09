"""Constrained selection on the free-SQL side (CONSTRAINED_IDS=1)."""
import json
from types import SimpleNamespace
from hepcoveragekg.eval import free_sql as F


class _Client:
    def __init__(self, reply): self.reply = reply; self.calls = []
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.calls.append(kw); return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_constrained_papers_selects_only_touched_ids():
    cl = _Client(json.dumps({"papers": ["2001.00001", "2099.09999"]}))
    picked, mode = F.constrained_papers(cl, "m", "q", "draft", {"2001.00001", "2002.00002"}, {"2001.00001": "SELECT ..."})
    assert picked == ["2001.00001"] and mode == "guided_json"
    assert cl.calls[0]["extra_body"]["guided_json"]["properties"]["papers"]["items"]["enum"] == ["2001.00001", "2002.00002"]


def test_no_candidates_no_call():
    cl = _Client("{}")
    assert F.constrained_papers(cl, "m", "q", "d", set()) == ([], "") and not cl.calls
