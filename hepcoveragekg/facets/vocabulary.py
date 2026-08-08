"""Controlled vocabularies and the canonical entity map (lab repair B2; v2).

The prompt-R&D lab measured that a closed object vocabulary gives high
expectation precision (84%/93%) with zero invented types, while free-text
labels carry all the granularity variance. The acquisition contract keeps
free-text labels as the evidence-true entity identity and derives the
canonical name; nothing is ever stored *instead of* the label.

objects-v2 (2026-07-27): the fleet showed the v1 exact-label table matched
only 21% of the 471 detector_object entities the pilot actually produced.
v2 folds LaTeX markup, parentheticals ("b-tagged jet (MV2c10, 77%)"), and
qualifier words before matching, and the enum gained the members the fleet
demonstrably needs (MET, vertices, c-/track-jets, H/Z candidates, tracks).
MET's lab-era exclusion ("a quantity, not a countable object") survives as
NON_COUNTABLE, not as absence from the map.

generators-v1: generator labels are family + version ("Pythia 8.230",
"MG5_aMC@NLO 2.6.2"); the canonical name is the family, the version is a
derived attribute.

The map is deterministic and unique: patterns are tried in fixed order and
the first match wins; `tests/test_vocabulary.py` locks the fleet-level
coverage and uniqueness.
"""

from __future__ import annotations

import re

from .models import DetectorObjectName

OBJECT_VOCABULARY_VERSION = "objects-v2"
GENERATOR_VOCABULARY_VERSION = "generators-v1"

# Canonical identities that are never countable (no object_count
# expectations): the lab schema decision about MET, kept out of the
# counting layer rather than out of the identity map.
NON_COUNTABLE = frozenset({
    DetectorObjectName.MET,
    DetectorObjectName.PRIMARY_VERTEX,
})

# Qualifier words that never change which canonical object a label denotes.
# "primary"/"secondary" are NOT here: they distinguish vertices.
_QUALIFIERS = (
    "prompt", "isolated", "signal", "baseline", "loose", "medium", "tight",
    "reconstructed", "selected", "control", "veto", "calorimeter",
    "well identified", "high purity", "light flavour", "leading",
    "subleading", "sub leading",
)
_QUALIFIER_RE = re.compile(r"\b(" + "|".join(_QUALIFIERS) + r")\b")

# Ordered: specific before generic (b/c/track/large-R jets before "jet").
_SYNONYMS: list[tuple[str, DetectorObjectName]] = [
    (r"^electron$", DetectorObjectName.ELECTRON),
    (r"^muon$", DetectorObjectName.MUON),
    (r"^(hadronically decaying |hadronic )?tau( lepton)?$", DetectorObjectName.TAU),
    (r"^tau ?had( vi[sz]?)?$", DetectorObjectName.TAU),
    (r"^(converted |unconverted |di|fsr |final state radiation )?photon$", DetectorObjectName.PHOTON),
    (r"^(soft )?(b|bottom)?( tagged)? ?track jet$", DetectorObjectName.TRACK_JET),
    (r"^(b|bottom)( quark)?( tagged)? jet$", DetectorObjectName.BJET),
    (r"^(c|charm)( quark)?( tagged)? jet$", DetectorObjectName.CJET),
    (r"^((reclustered )?large (radius|r) jet|fat jet|ak8( jet)?|reclustered r ?[0-9.]+ jet)$",
     DetectorObjectName.LARGE_R_JET),
    (r"^(small radius |ak4 |anti kt? |isr |vbf |forward |additional |associated |hadronic )*jet$",
     DetectorObjectName.JET),
    (r"^(primary |secondary )?(hadronic )?top( quark)?( tagged)?( jet)?$",
     DetectorObjectName.TOP_CANDIDATE),
    (r"^(w( z)?( boson)?|w tagged( jet)?)$", DetectorObjectName.W_CANDIDATE),
    (r"^z( boson)?( dilepton)?$", DetectorObjectName.Z_CANDIDATE),
    (r"^h(iggs)?( boson)?$", DetectorObjectName.HIGGS_CANDIDATE),
    (r"^(displaced|secondary) vert(ex|ice)$", DetectorObjectName.DISPLACED_VERTEX),
    (r"^(diphoton )?(primary|interaction)( interaction)?( collision)? vertex$|^pv$",
     DetectorObjectName.PRIMARY_VERTEX),
    (r"^(track based )?missing transverse (momentum|energy)$|^met$|^etmiss$|^ptmiss$",
     DetectorObjectName.MET),
    (r"^(ghost associated |inner detector |id )?track$", DetectorObjectName.TRACK),
    (r"^(light |charged )?lepton$", DetectorObjectName.LEPTON),
]

# Generator families: pattern on the folded label prefix -> canonical name.
_GENERATOR_FAMILIES: list[tuple[str, str]] = [
    (r"pythia ?6", "Pythia6"),
    (r"pythia", "Pythia8"),
    (r"sherpa", "Sherpa"),
    (r"powheg", "PowhegBox"),
    (r"(madgraph|mg)5? ?a?mc(@| at )?nlo|^madgraph", "MadGraph5_aMC@NLO"),
    (r"herwig ?\+\+|herwigpp", "Herwig++"),
    (r"herwig", "Herwig7"),
    (r"evtgen", "EvtGen"),
    (r"madspin", "MadSpin"),
    (r"mcfm", "MCFM"),
    (r"openloops?", "OpenLoops"),
    (r"nnlops?", "NNLOPS"),
    (r"jhugen", "JHUGen"),
    (r"gosam", "GoSam"),
    (r"minlo", "MiNLO"),
    (r"fewz", "FEWZ"),
    (r"top\+\+|toppp", "Top++"),
    (r"resummino", "Resummino"),
    (r"lpair", "LPAIR"),
    (r"jetset", "JETSET"),
    (r"photos", "Photos"),
    (r"tauola", "Tauola"),
    (r"geant ?4?", "Geant4"),
    (r"amc@nlo|amcatnlo", "MadGraph5_aMC@NLO"),
]
# A version needs a dot: "8.230" yes, the "8" in "Pythia8" no.
_VERSION_RE = re.compile(r"v?(\d+(?:\.\d+)+)")


def _fold(label: str) -> str:
    text = label.strip().lower()
    text = re.sub(r"\(.*?\)", " ", text)             # parentheticals are detail, not identity
    text = re.sub(r"\\(tau|gamma|mu|ell)\b", r" \1 ", text)  # greek symbols are words
    text = re.sub(r"\\[a-z]+", " ", text)            # latex commands
    text = re.sub(r"[${}^]", "", text)               # latex markup remnants
    text = re.sub(r"[_\-/=,;:]+", " ", text)         # punctuation to spaces
    text = re.sub(r"[→>]+", " ", text)
    text = _QUALIFIER_RE.sub(" ", text)
    text = re.sub(r"\bcandidates?\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[a-z])s$", "", text)         # naive trailing singularization
    return text


def normalize_object_label(label: str) -> tuple[str, bool]:
    """Map a free-text detector-object label onto the canonical vocabulary.

    Returns (canonical_or_original, matched). The original label is returned
    unchanged when nothing matches — callers flag, never reject.
    """
    folded = _fold(label)
    for pattern, canonical in _SYNONYMS:
        if re.match(pattern, folded):
            return canonical.value, True
    return label, False


def canonical_generator(label: str) -> tuple[str | None, str | None]:
    """Return (family, version) for a generator label, (None, None) if unknown."""
    folded = _fold(label)
    for pattern, family in _GENERATOR_FAMILIES:
        if re.search(pattern, folded):
            match = _VERSION_RE.search(label)
            return family, match.group(1) if match else None
    return None, None


def canonicalize_entity(kind: str, label: str) -> dict | None:
    """Derived canonical annotation for an entity, or None when the kind has
    no controlled vocabulary or the label falls outside it. Never a
    substitute for the label — a versioned, recomputable grouping key."""
    if kind == "detector_object":
        canonical, matched = normalize_object_label(label)
        if not matched:
            return None
        return {"canonical": canonical, "vocabulary": OBJECT_VOCABULARY_VERSION}
    if kind == "generator":
        family, version = canonical_generator(label)
        if family is None:
            return None
        result = {"canonical": family, "vocabulary": GENERATOR_VOCABULARY_VERSION}
        if version:
            result["version"] = version
        return result
    return None


# ---------------------------------------------------------------------------
# Facet tags (facets-v1): enum navigation over descriptive entity kinds.
#
# For background_method / statistical_method / systematic_uncertainty /
# background / physics_process / bsm_model, fleet labels are near-unique
# descriptions (213/213 distinct background_method labels), so a 1:1
# synonym map is the wrong tool. Instead each label collects every matching
# technique/family tag from a closed per-kind list: multi-label, keyword
# driven, deterministic. Labels stay the identity; tags are the cheap
# navigation index ("all analyses using an ABCD estimate").
# ---------------------------------------------------------------------------

FACET_VOCABULARY_VERSION = "facets-v1"

_PROCESS_FAMILY = [
    (r"\btt ?z\b", "TTV"),
    (r"\btt ?w\b", "TTV"),
    (r"\btwz\b|\btzq?\b|\bttt t?\b|\bttww\b|rare top", "TTV"),
    (r"\bt ?t ?h\b|association with a top quark", "HiggsTTH"),
    (r"\bttbar\b|\btt\b(?! ?[zwh]\b)|top quark antiquark|top quark pair|top pair", "TTbar"),
    (r"single top|\btw\b|s channel|t channel", "SingleTop"),
    (r"\bw jets?\b|w boson production in association|w boson and a charm|\bw (c|cc|bb|charm|heavy)\b|\bw [ℓl]?ν\S* jets?", "WJets"),
    (r"\bz jets?\b|drell yan|\bdy\b|z boson production in association|z νν|z nu nu|\bzjj\b|z ?γ ?\*|\bz [ℓl]{2}\b", "ZJets"),
    (r"\b[wzv] ?γ\b(?! ?\*)|[wzv] boson \+? ?photon", "VGamma"),
    (r"minimum bias|pile ?up overlay", "MinimumBias"),
    (r"diboson|di boson|\bww\b|\bwz\b|\bzz\b", "Diboson"),
    (r"multi ?boson|triboson|\bvvv\b|\bwwz\b|\bwww\b", "Multiboson"),
    (r"multi ?jet|\bqcd\b|all hadronic", "Multijet"),
    (r"fake|non ?prompt|\bfnp\b|misidentif", "FakeNonPrompt"),
    (r"cosmic|beam induced|beam halo|non collision", "NonCollision"),
    (r"\bggf\b|\bggh\b|gluon( gluon)? fusion", "HiggsGGF"),
    (r"\bvbf\b|vector boson fusion", "HiggsVBF"),
    (r"\bvh\b|\bwh\b|\bzh\b|\bggzh\b|higgsstrahlung|association with a (vector boson|w or z)", "HiggsVH"),
    (r"\bth(qb?|w)?\b|single top.*higgs", "HiggsTH"),
    (r"\bbbh\b|higgs boson production", "HiggsOther"),
    (r"^h \S|higgs( boson)? decay|h decay", "HiggsDecay"),
    (r"\bhh\b|di ?higgs|double higgs", "DiHiggs"),
    (r"diphoton|gamma gamma|γγ", "Diphoton"),
    (r"(γ|gamma|photon) jets?\b", "GammaJets"),
    (r"supersymmet|\bsusy\b|s?top squark|squark|gluino|neutralino|chargino|slepton|sbottom|electroweakino|\bgmsb\b|χ|\bchi ?\d|msugra|cmssm|\bmssm\b", "SUSY"),
    (r"leptoquark", "Leptoquark"),
    (r"dark matter|\bwimp\b|axion|\balps?\b|dark photon|dark higgs|invisible|dark sector|dark energy|hidden valley", "DarkMatter"),
    (r"z ?[′']|w ?[′']|heavy resonance|heavy (gauge|vector) boson|heavy vector triplet|\bhvt\b|sequential standard model|graviton", "HeavyResonance"),
    (r"heavy neutral lepton|\bhnl\b|majorana|right handed neutrino|sterile neutrino|seesaw", "HeavyNeutrino"),
    (r"long lived|displaced|\bllp\b", "LongLived"),
    (r"2hdm|two higgs doublet|charged higgs|pseudoscalar|extended higgs|mssm higgs", "ExtendedHiggs"),
    (r"vector like quark|\bvlq\b", "VLQ"),
    (r"vector like lepton|\bvll\b", "VLL"),
    (r"extra dimension|kaluza|randall", "ExtraDimensions"),
    (r"\beft\b|wilson coefficient|\bsmeft\b|anomalous coupling|effective field|kappa framework|κ framework|coupling modifier|pseudo observable|hvv coupling|yukawa coupling", "EFT"),
    (r"technicolor|little higgs|composite|neutral naturalness", "CompositeModels"),
    (r"elastic|forward proton|photon induced|ultraperipheral|diffract", "ForwardElastic"),
    (r"standard model higgs|\bsm higgs\b|^(the )?standard model( sm)?$", "StandardModel"),
    (r"beyond the (standard model|sm)|\bbsm\b", "GenericBSM"),
]

_FACET_TABLES: dict[str, list[tuple[str, str]]] = {
    "background_method": [
        (r"control region|\bcr[a-z0-9]*\b|normali[sz]", "ControlRegionNormalisation"),
        (r"abcd", "ABCD"),
        (r"fake|matrix method|tight loose|\bfnp\b|misidentif|non ?prompt", "FakeEstimate"),
        (r"jet smearing|resolution function", "JetSmearing"),
        (r"template", "TemplateFit"),
        (r"parametric|functional fit|analytic|smooth function|\bbpdf\b|\bbwz|background function|core pdf", "ParametricFit"),
        (r"simulat|\bmc\b|monte carlo", "SimulationBased"),
        (r"embedd", "Embedding"),
        (r"mixed data|event mixing", "EventMixing"),
        (r"transfer factor|extrapolat|kappa", "TransferFactor"),
        (r"reweight", "Reweighting"),
        (r"side ?band", "Sideband"),
        (r"data driven", "DataDriven"),
    ],
    "statistical_method": [
        (r"profile likelihood|profiled", "ProfileLikelihood"),
        (r"\bcls?\b|cl s\b", "CLs"),
        (r"maximum likelihood|likelihood fit|\bml fit\b", "MaximumLikelihoodFit"),
        (r"asymptotic", "AsymptoticFormulae"),
        (r"pseudo experiment|\btoys?\b|bootstrap", "PseudoExperiments"),
        (r"unfold", "Unfolding"),
        (r"template", "TemplateFit"),
        (r"tag and probe", "TagAndProbe"),
        (r"histfitter", "HistFitter"),
        (r"cms combine|combine (statistical|tool)", "Combine"),
        (r"beeston|barlow", "MCStatTreatment"),
        (r"shape fit", "ShapeFit"),
        (r"background only", "BackgroundOnlyFit"),
        (r"asimov", "AsimovDataset"),
        (r"f test|fisher test|discrete profiling|polynomial order|functional form|spurious signal", "FunctionalFormSelection"),
        (r"log normal|gaussian constrain|poisson|nuisance parameter", "LikelihoodModelling"),
        (r"punzi|significance|likelihood ratio test|\bllr\b|wilks|hypothesis", "HypothesisTesting"),
        (r"chi squared|chi2|χ2|least squares", "ChiSquareFit"),
        (r"morphing|interpolat", "Morphing"),
        (r"replica|bayesian reweight", "Replicas"),
        (r"clopper|confidence interval|neyman", "ConfidenceIntervals"),
        (r"smoothing|pruning|symmetri[sz]", "SystematicsProcessing"),
    ],
    "systematic_uncertainty": [
        (r"jet energy scale|\bjes\b", "JES"),
        (r"jet energy resolution|\bjer\b", "JER"),
        (r"(b|c|flavou?r) tag|b veto|mistag", "FlavourTagging"),
        (r"(lepton|electron|muon).*(efficien|identific|reconstruc|isolat|scale factor)|(electron|muon) (energy|momentum)", "LeptonPerformance"),
        (r"\btau\b", "TauPerformance"),
        (r"photon", "PhotonPerformance"),
        (r"\bmet\b|missing transverse|soft term", "MET"),
        (r"trigger", "Trigger"),
        (r"track", "Tracking"),
        (r"pile ?up", "Pileup"),
        (r"luminosit", "Luminosity"),
        (r"\bpdf\b|parton distribution|alpha ?_?s\b|αs", "PDF"),
        (r"renormali[sz]ation|factori[sz]ation|qcd scale|scale variation", "QCDScale"),
        (r"parton shower|hadroni[sz]ation|fragmentation|underlying event|\btune\b|\bisr\b|\bfsr\b|gluon splitting", "PartonShower"),
        (r"modell?ing|generator", "Modelling"),
        (r"statistic", "Statistics"),
        (r"cross section", "CrossSection"),
    ],
    "background": _PROCESS_FAMILY,
    "physics_process": _PROCESS_FAMILY,
    "bsm_model": _PROCESS_FAMILY,
    # A sample is a process (plus a generator) in a wrapper: the process
    # family list applies directly ("ttbar MC sample" -> TTbar).
    "sample": _PROCESS_FAMILY,
    "observable": [
        (r"upper limit|exclusion limit|excluded|95 ?% ?cl|confidence level limit", "ExclusionLimit"),
        (r"differential|\bvs\b|as a function of|spectrum", "Differential"),
        (r"fiducial", "Fiducial"),
        (r"cross section ratio|ratio", "Ratio"),
        (r"cross section", "CrossSection"),
        (r"signal strength|\bmu (ggh|vbf|vh|wh|zh|tth)|signal strength modifier", "SignalStrength"),
        (r"branching (ratio|fraction)", "BranchingFraction"),
        (r"asymmetry", "Asymmetry"),
        (r"significance|p value|local z\b", "Significance"),
        (r"coupling|wilson|\bsmeft\b|kappa|κ|yukawa|anomalous", "CouplingConstraint"),
        (r"mass limit|limit on .* mass|mass exclusion", "MassLimit"),
        (r"transverse momentum|\bpt?\b|invariant mass|\bm\b|rapidity|\by\b|\beta\b|angular|delta phi|δφ|angle|\bht\b|multiplicity|n jets?\b|\bmet\b|missing transverse", "KinematicDistribution"),
        (r"polari[sz]ation|helicity|spin", "Polarisation"),
        (r"width", "Width"),
        (r"yield|number of events", "EventYield"),
    ],
}


def _fold_light(label: str) -> str:
    """Facet fold: like _fold but parenthetical CONTENT is kept — for
    descriptive labels the parenthetical often carries the technique
    ("(ggH category)", "(PDF, alpha_s, scale)")."""
    import unicodedata
    text = unicodedata.normalize("NFKD", label)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\\(tau|gamma|mu|ell|chi|nu|bar|tilde)\b", r" \1 ", text)
    text = re.sub(r"\\[a-z]+", " ", text)
    text = re.sub(r"[${}()^\[\]]", " ", text)
    text = re.sub(r"\b(bar|tilde)\b", "", text)
    text = re.sub(r"\bt +t\b", "tt", text)  # "$t\bar{t}$" folds to "t t"
    text = re.sub(r"[_\-/=,;:+.'’′]+", " ", text)
    text = re.sub(r"[→>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def facet_tags(kind: str, label: str) -> list[str]:
    """All facet tags matching a label, in table order; [] when the kind has
    no facet table or nothing matches. Deterministic and multi-label."""
    table = _FACET_TABLES.get(kind)
    if not table:
        return []
    folded = _fold_light(label)
    tags: list[str] = []
    for pattern, tag in table:
        if tag not in tags and re.search(pattern, folded):
            tags.append(tag)
    return tags
