# Entity kinds — what each one means

**For Gabriel.** The graph types every entity as one of 22 kinds. Until now the
extraction schema listed the 22 NAMES with no definitions at all, so the
extractor was choosing between them unguided — and the query agents had no way
to use the type as a filter.

These are first drafts, written from what the entities actually look like. The
distinction that matters most in practice is **`detector_object` vs
`physics_process`**: one question in the last batch asks which analyses
*reconstruct a Higgs candidate as a physical object they select on, rather than
merely studying Higgs production* — and that difference is the kind, not the
label, since both are called "Higgs".

Corrections welcome on any of them, and especially on the ones marked ⚠, where
the boundary looks genuinely ambiguous to me.

| kind | n | draft meaning | examples from the graph |
|---|---|---|---|
| **systematic_uncertainty** | 803 | A source of systematic uncertainty. | `PDF uncertainty`, `PDF uncertainty on R_W` |
| **event_region** | 771 | A region of phase space the analysis defines and uses. | `SRC1`, `SRC2` |
| **sample** | 551 | A simulated or data sample used by the analysis. | `tZ MC sample`, `tW MC sample` |
| **observable** | 422 | A quantity that is measured or reported. | `σ(pp→H)×B(H→Zη_c)`, `σ(pp→H)×B(H→ZJ/ψ)` |
| **result** | 272 | One specific finding inside a paper -- the headline analysis, or a single number such as a normalisation factor or a per-region measurement. | `Data deficit: SR^WZ-5`, `Data excess: SR^Wh_DFOS-1` |
| **statistical_method** | 244 | A statistical method or framework used for the fit, limits or significance. | `CLs technique`, `sPlot technique` |
| **bsm_model** | 237 | A theory beyond the Standard Model that the analysis targets, constrains or interprets. | `cMSSM`, `mSUGRA` |
| **physics_process** | 229 | A process being studied, targeted or produced -- as opposed to something reconstructed and cut on. | `H → Z η_c`, `H → Z J/ψ` |
| **detector_object** | 227 | A reconstructed physics object the analysis SELECTS ON: electrons, jets, b-tagged jets, missing transverse momentum, a Higgs candidate it builds and cuts on. | `Jet`, `Muon` |
| **background** | 223 | A background process the analysis estimates. | `$WZ$ background`, `$WW$ background` |
| **generator** | 223 | The Monte Carlo generator (and tune) that produced a sample. | `FEWZ`, `EPOS` |
| **channel** | 208 | A final state or decay channel. | `H→γγ`, `H→ττ` |
| **background_method** | 203 | The method used to estimate a background. | `Estimated from data`, `Modified ABCD estimate` |
| **model_parameter** | 154 | A parameter of such a model. | `c_g`, `c_HG` |
| **benchmark** ⚠ | 126 | A specific parameter point or scenario used as a reference. | `CT18 PDF set`, `CT18Z PDF set` |
| **dataset** | 72 | A dataset the analysis runs over, usually a data-taking period and luminosity. | `CMS pPb collision dataset`, `CMS PbPb collision dataset` |
| **paper** | 60 | One published article. | — |
| **selection_requirement** ⚠ | 37 | A named requirement, where the paper gives it an identity of its own. | `Photon track isolation`, `1L channel muon selection` |
| **collision_system** | 24 | The colliding beams and energy. | `pp`, `pPb collisions` |
| **object_definition** ⚠ | 24 | The definition of such an object, where the paper states it separately from the object itself. | `Light jet`, `Particle-level jet` |
| **result_quantity** ⚠ | 4 | A specific numeric quantity attached to a result. | — |

## The two I am least sure about

**`detector_object` vs `object_definition`** — 227 vs 24. The split looks
inconsistent: some papers file the object and its definition separately, most
do not. It may not be a real distinction.

**`selection_requirement`** — only 37 entities, while 1,335 selection facts are
stored as free text hanging off `detector_object`. So the kind exists but is
almost unused, and the information lives somewhere else.
