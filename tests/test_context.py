# =============================================================================
# Entity context for adjudication: every wording plus the defining quotes.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import context


def test_section_priority_is_learned_per_kind(monkeypatch):
    """A single global ordering was wrong: physics_process takes 30% of its
    quotes from the Introduction and 2% from reconstruction, while
    detector_object is the reverse. Priority is measured per kind."""
    monkeypatch.setattr(context, "_SECTION_PRIORS", {
        "detector_object": {"object reconstruction": 500, "introduction": 5},
        "physics_process": {"object reconstruction": 5, "introduction": 400},
    })
    # same two sections, opposite verdicts depending on the kind
    assert context.section_rank("Object reconstruction", "detector_object") < \
           context.section_rank("Introduction", "detector_object")
    assert context.section_rank("Introduction", "physics_process") < \
           context.section_rank("Object reconstruction", "physics_process")


def test_unseen_sections_rank_neutral_not_last(monkeypatch):
    """An unusual heading is not evidence of being uninformative."""
    monkeypatch.setattr(context, "_SECTION_PRIORS",
                        {"detector_object": {"results": 100, "introduction": 50}})
    unseen = context.section_rank("Three-body event selection", "detector_object")
    assert unseen == 0
    assert unseen > context.section_rank("Results", "detector_object")   # known-good still wins
    assert context.section_rank(None, "detector_object") == 0


def test_render_omits_empty_sections():
    c = context.EntityContext("e1", "detector_object", labels=["Electron"])
    text = c.render("ENTITY A")
    assert "ENTITY A:" in text and 'primary name: "Electron"' in text
    for absent in ("also written as", "aliases:", "attributes:", "quotes from the papers"):
        assert absent not in text


def test_render_shows_every_wording_and_labels_the_section():
    c = context.EntityContext(
        "e1", "background",
        labels=["Diboson background", "Diboson ($VV$) background"],
        aliases=["diboson"],
        attributes={"level": "reconstructed"},
        papers=8,
        quotes=[("Event selection", "Diboson production is estimated from simulation.")],
    )
    text = c.render("ENTITY B")
    assert 'also written as: "Diboson ($VV$) background"' in text
    assert "appears in 8 paper(s)" in text
    assert "quotes from the papers:" in text
    assert '- [Event selection] "Diboson production is estimated from simulation."' in text


def test_long_quotes_are_truncated_not_dropped():
    long_quote = "x" * (context.QUOTE_CHARS + 200)
    out = context._truncate(long_quote)
    assert len(out) <= context.QUOTE_CHARS
    assert out.endswith("…")


def test_whitespace_is_collapsed():
    assert context._truncate("a\n\n  b\tc") == "a b c"


def test_latex_typography_is_stripped_but_notation_survives():
    """~10% of quote characters are markup and the tail is far worse, so the
    320-char budget was buying almost no sentence on exactly the entities where
    quotes matter most."""
    out = context.clean_latex(r"$E_{\text{T}}^{\text{miss}}$ > 30\,\text{GeV}")
    assert out == "E_T^miss > 30 GeV"
    assert context.clean_latex(r"Jet (anti-$k_{t}$, $R=0.4$)") == "Jet (anti-k_t, R=0.4)"


def test_a_bar_is_meaning_not_typography():
    """`t\\bar{t}` is ttbar and a physicist writes it that way; dropping the bar
    silently would change the particle."""
    assert "ttbar" in context.clean_latex(r"production of $t\bar{t}$ events")


def test_deeply_nested_bars_survive_the_corpus_nesting():
    """Order caught this: running the bar rule before the font commands left a
    literal `\\overlinet` in the output."""
    ugly = r"${\mathup{{{t}}}}{}{\mathup{{\overline{{{\mathup{{{t}}}}}}}}}$"
    assert "overline" not in context.clean_latex(ugly)
    assert "ttbar" in context.clean_latex(ugly)


def test_cleaning_happens_before_truncation():
    """Otherwise the budget is spent on markup that is then thrown away."""
    noisy = r"\mathrm{" * 30 + "the actual sentence content" + "}" * 30
    assert "the actual sentence content" in context._truncate(noisy, limit=60)


def test_empty_embed_model_env_falls_back_to_default():
    """An empty ALIASES_EMBED_MODEL must mean UNSET, not "load a model named ''".

    Slurm's `--export=ALL,VAR=` sets the variable to blank. `os.environ.get`
    with a default does not catch that, so `SentenceTransformer("")` builds an
    object whose first module is None and the run dies much later with
    `'NoneType' object has no attribute 'tokenize'` -- naming neither the
    variable nor the model. Four dev-200 jobs died this way (2026-09-02).
    """
    import os
    from hepcoveragekg.aliases import semantics

    original = os.environ.get("ALIASES_EMBED_MODEL")
    try:
        for blank in ("", "   ", "\t"):
            os.environ["ALIASES_EMBED_MODEL"] = blank
            assert semantics._model_name_from_env() == semantics.DEFAULT_MODEL
        os.environ.pop("ALIASES_EMBED_MODEL", None)
        assert semantics._model_name_from_env() == semantics.DEFAULT_MODEL
        os.environ["ALIASES_EMBED_MODEL"] = "some/other-model"
        assert semantics._model_name_from_env() == "some/other-model"
    finally:
        os.environ.pop("ALIASES_EMBED_MODEL", None)
        if original is not None:
            os.environ["ALIASES_EMBED_MODEL"] = original


def test_env_config_records_the_model_actually_used():
    """Provenance must not record "" when the code fell back to the default."""
    import os
    from hepcoveragekg.eval.systems import env_config

    original = os.environ.get("ALIASES_EMBED_MODEL")
    try:
        os.environ["ALIASES_EMBED_MODEL"] = ""
        assert env_config()["env.ALIASES_EMBED_MODEL"] == "BAAI/bge-base-en-v1.5"
        os.environ["ALIASES_EMBED_MODEL"] = "thellert/physbert_cased"
        assert env_config()["env.ALIASES_EMBED_MODEL"] == "thellert/physbert_cased"
    finally:
        os.environ.pop("ALIASES_EMBED_MODEL", None)
        if original is not None:
            os.environ["ALIASES_EMBED_MODEL"] = original


def test_cache_path_separates_encoders_and_index_contents():
    """Two encoders must never share one embedding cache file.

    Keyed only on index contents, a chATLAS arm and a bge-base arm pointed at
    the same .npz, each read the other's stored model name, each logged
    "discarding" and re-encoded ~14k strings. Two concurrent arms would thrash
    forever -- and that is exactly the shape of the encoder x search-sets
    experiment (D-089).
    """
    import os
    from hepcoveragekg.query import retrieve

    original = os.environ.get("ALIASES_EMBED_MODEL")
    try:
        os.environ.pop("ALIASES_EMBED_MODEL", None)
        base = retrieve.cache_path()
        os.environ["ALIASES_EMBED_MODEL"] = "kipark/all-mpnet-base-v2-combined_4400-400vs1000"
        other = retrieve.cache_path()
        assert base != other, "different encoders must not share a cache file"
        # the model name must not leak into the filename: it contains a "/"
        assert "/" not in os.path.basename(other)
        # index contents still separate caches, per encoder
        assert retrieve.cache_path(include_values=True) != retrieve.cache_path()
        assert retrieve.cache_path(include_quotes=True) != retrieve.cache_path()
        assert (retrieve.cache_path(include_values=True, include_quotes=True)
                != retrieve.cache_path(include_values=True))
        # and it is stable for the same configuration
        assert retrieve.cache_path() == retrieve.cache_path()
    finally:
        os.environ.pop("ALIASES_EMBED_MODEL", None)
        if original is not None:
            os.environ["ALIASES_EMBED_MODEL"] = original


def test_embedding_cache_write_is_atomic(tmp_path):
    """A reader must never see a half-written cache.

    Concurrent Slurm arms share a cache path; `np.savez` direct to that path is
    a multi-second non-atomic write, so a reader arriving mid-write gets a
    truncated npz and the run dies hours in.
    """
    import numpy as np
    from hepcoveragekg.aliases import semantics

    cache = tmp_path / "idx.npz"
    semantics.embed_cached(["alpha", "beta"], cache)
    assert cache.exists()
    # no temp files left behind
    assert not list(tmp_path.glob("*.tmp.npz"))
    blob = np.load(cache, allow_pickle=False)
    assert set(blob["texts"]) == {"alpha", "beta"}
    # a second call reuses and still leaves the directory clean
    semantics.embed_cached(["alpha", "beta", "gamma"], cache)
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_planner_temperature_defaults_to_greedy():
    """0.0 unless explicitly asked. Every arm measured before 2026-09-04 assumed
    greedy decoding; a non-zero default would silently move every baseline."""
    import os
    from hepcoveragekg.query import planner
    original = os.environ.get("PLANNER_TEMPERATURE")
    try:
        os.environ.pop("PLANNER_TEMPERATURE", None)
        assert planner.planner_temperature() == 0.0
        os.environ["PLANNER_TEMPERATURE"] = ""
        assert planner.planner_temperature() == 0.0
    finally:
        os.environ.pop("PLANNER_TEMPERATURE", None)
        if original is not None:
            os.environ["PLANNER_TEMPERATURE"] = original


def test_planner_temperature_is_read_fresh_not_captured_at_import():
    """A job script exporting this AFTER the module is imported must still win --
    the same failure `completion_cap` was fixed for (D-084)."""
    import os
    from hepcoveragekg.query import planner
    original = os.environ.get("PLANNER_TEMPERATURE")
    try:
        os.environ["PLANNER_TEMPERATURE"] = "0.7"
        assert planner.planner_temperature() == 0.7
        os.environ["PLANNER_TEMPERATURE"] = "0.3"
        assert planner.planner_temperature() == 0.3
    finally:
        os.environ.pop("PLANNER_TEMPERATURE", None)
        if original is not None:
            os.environ["PLANNER_TEMPERATURE"] = original


def test_planner_temperature_rejects_nonsense_rather_than_crashing_a_run():
    import os
    from hepcoveragekg.query import planner
    original = os.environ.get("PLANNER_TEMPERATURE")
    try:
        for bad in ("hot", "-1", "9"):
            os.environ["PLANNER_TEMPERATURE"] = bad
            assert planner.planner_temperature() == 0.0, bad
    finally:
        os.environ.pop("PLANNER_TEMPERATURE", None)
        if original is not None:
            os.environ["PLANNER_TEMPERATURE"] = original


def test_a_reasoning_critic_gets_the_reasoning_token_budget():
    """D-084 repeated in the critic path (found 2026-09-05).

    The critic call hardcoded MAX_COMPLETION_TOKENS=800. A reasoning judge
    spends that on its chain of thought, never emits the verdict JSON, the
    parser returns {} and EVERY candidate defaults to kept -- silently, because
    a default is not an error. Measured across four 164-question runs with
    Qwen3.5-9B: 33,836 candidates, 100% defaulted, zero drops.
    """
    from hepcoveragekg.query.planner import completion_cap, MAX_COMPLETION_TOKENS
    assert completion_cap("Qwen/Qwen3.5-9B") > MAX_COMPLETION_TOKENS
    assert completion_cap("Qwen/QwQ-32B-AWQ") > MAX_COMPLETION_TOKENS
    # a non-reasoning judge is unaffected
    assert completion_cap("NousResearch/Meta-Llama-3.1-8B-Instruct") == MAX_COMPLETION_TOKENS
