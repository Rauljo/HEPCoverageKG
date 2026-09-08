from hepcoveragekg.eval.scoring import _label_fuzzy_hit, _normalise_text

def test_paraphrase_counts_and_unrelated_does_not():
    ans = _normalise_text("It estimates Drell-Yan, nonprompt lepton from Z+jets and ttbar, triboson and ZZ.")
    assert _label_fuzzy_hit("Drell-Yan background", ans)
    assert _label_fuzzy_hit("Triboson (VVV) background", ans)
    assert not _label_fuzzy_hit("W+jets background", ans)
    assert not _label_fuzzy_hit("Charge-flip electrons", ans)
