"""One worked call per tool, mined from runs that answered correctly.

NOT ONE of the twelve tool descriptions carried an example. They say what a tool
IS -- "papers whose analysis card carries these closed-vocabulary values" -- and
leave the model to infer when to reach for it and what to put in it. The
measured failures are exactly there:

  gf-08   searched `kind="selection_requirement"` and hopped through
          `region_requires_object`, which relates detector_objects. Empty by
          construction, and all six arms answered "the graph does not record
          any" about thirteen papers that do.
  Qwen    makes exactly one search on 135 of 207 records and none on 69. It
          never reaches for `facets`, which answers gf-01 in a single call --
          sol found that route in round 1.

WHY MINED RATHER THAN WRITTEN. An invented example shows what the author thinks
the tool is for. These are real calls from records that scored >= 0.75, so each
one demonstrably led somewhere. `facets(field="statistical_methods",
values=["Unfolding"])` returned 11 rows on a question that scored 0.95; that is
a fact about the graph, not a hope about the API.

NOT CHEAPER THAN FEW-SHOT, which was the first claim made for it and was wrong:
this block is 1,679 characters against the few-shot-with-plan block's 1,474, and
every character is re-sent on every round. Its case rests on two other things.

  WHERE it lands. A few-shot block shows three whole questions and hopes the
  shape transfers. This names, for each tool, one call that worked -- so the
  guidance is present at the moment the model is choosing that tool, and covers
  nine tools rather than whichever three the exemplars happened to use.

  NO CONTAMINATION AT ALL. These are calls, not answers. There is no gold to
  leak, so nothing has to be held disjoint from the evaluation set and no
  distillation caveat attaches to the result.
"""
from __future__ import annotations

#: tool -> (example call, what it got, when to reach for it)
EXAMPLES: dict[str, tuple[str, str, str]] = {
    "search": (
        'search(text="PDF modelling uncertainty (CT14, MMHT2014 vs NNPDF3.0)", kind="systematic_uncertainty")',
        "185 entities",
        "you know the CONCEPT but not how the papers spell it. Leave `kind` out "
        "when unsure -- one concept lives under several kinds"),
    "facets": (
        'facets(field="statistical_methods", values=["Unfolding"], mode="any")',
        "11 papers",
        "the question names something in a closed vocabulary. ONE call, no "
        "search first -- this is the shortest route to a paper set and it is "
        "the one most often missed"),
    "facet_entities": (
        'facet_entities(field="statistical_methods")',
        "16 distinct things",
        "you want to know WHICH values exist before filtering on one"),
    "subjects_of": (
        'subjects_of(predicate="sample_uses_generator", object_set="set_1")',
        "453 rows",
        "walking backwards: which things point AT what you hold. Check the "
        "predicate accepts your kind -- an impossible hop returns 0 and says so"),
    "describe": (
        'describe(entity_ids=[...], predicate="region_requires_object")',
        "54 rows",
        "walking forwards from entities you already have"),
    "papers_of": (
        'papers_of(entity_ids=["hepkg:systematic:l1_prefiring", ...])',
        "31 papers",
        "the terminal step: you hold the right entities and want their papers"),
    "contents_of": (
        'contents_of(paper_ids=["2006.05880"])',
        "231 rows",
        "the question names a paper. Do NOT search for a paper id"),
    "count": (
        'count(predicate="result_uses_sample", object_set="set_1")',
        "the number",
        "a counting question. Count over a JUDGED set, not a raw search"),
    "quotes": (
        'quotes(assertion_ids=[...])',
        "the verbatim sentence",
        "you need the paper\'s own words behind a fact"),
}


def render(tools: list[dict]) -> str:
    """A block naming, for each tool offered, one call that worked."""
    lines = ["", "WHEN TO REACH FOR EACH TOOL -- real calls from runs that answered correctly."]
    for spec in tools:
        name = spec.get("name") or (spec.get("function") or {}).get("name")
        item = EXAMPLES.get(name)
        if not item:
            continue
        call, got, when = item
        lines.append(f"  {call}")
        lines.append(f"      -> {got}. Use when {when}.")
    return "\n".join(lines) if len(lines) > 2 else ""
