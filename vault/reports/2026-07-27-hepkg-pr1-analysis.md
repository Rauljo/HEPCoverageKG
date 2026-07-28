# HEPKG_promopt_tests PR #1 — analysis (2026-07-27)

Gabriel's PR #1 (`gfacini/HEPKG_promopt_tests#1`): *"fully labeled set of characteristics; a
single set of enums for the particles to be mapped to, with quotes back to original text for
traceability."* Fetched the branch (`pr-1`), read the diff (30 files, +3,722). **It validates most
of our findings and independently builds a canonicalization + facet-navigation layer that overlaps
our aliases layer — and it corrects one part of our importer.** Bottom line up top:

- **No schema break.** Bundles are still `hepkg-acquisition-v0.2`; raw entities are unchanged
  (`aliases, attributes, entity_id, external_ids, kind, label`). **Our importer's schema gate still
  works on the new bundles.** `canonical`/`facets` are added at **compile/export** time only.
- **One real correction to us**: entity node identity is **(bundle_id, entity_id)**, *never* bare
  entity_id — see impact §2.

## What's new

1. **Canonical entity map** (`vocabulary.py`, +329). `canonicalize_entity(kind, label) -> {"canonical":
   "BJet", "vocabulary": "objects-v2"}` (generators also get `version`; `null` outside the vocab).
   Detector objects → a closed enum `DetectorObjectName` (**objects-v2**, expanded with CJet,
   TrackJet, ZCandidate, HiggsCandidate, PrimaryVertex, Track, MET). Generators →
   `canonical_generator(label) -> (family, version)` (**generators-v1**: `_GENERATOR_FAMILIES` +
   `_VERSION_RE`). `NON_COUNTABLE` = {MET, vertices} (canonical identity but never counted).
   **This is the supervisor's version of our aliases-layer Tier 1 — and it's the clean-enum
   "canonical standardization" Raul asked for** (BJet, not a picked existing id).
2. **Facet layer** (`facets-v1`). `facet_tags(kind, label) -> [tags]` from closed per-kind lists
   (`_FACET_TABLES`, `_PROCESS_FAMILY`). Descriptive kinds (background_method, statistical_method,
   systematic_uncertainty; background/physics_process/bsm_model share one ProcessFamily list) get
   **multi-label** tags instead of a 1:1 canonical ("Data-driven fake-factor…" → `["FakeEstimate",
   "DataDriven"]`). Fleet coverage 70–91%/kind; untagged tail is honest, not an error.
3. **`analysis_facets.jsonl`** — a NEW export: **one row per paper** = the analysis card, sets of
   closed-vocab values (`objects`, `generators`, `process_families`, `background_methods`,
   `statistical_methods`, `systematic_sources`, `observable_types`, `category`, `experiments`),
   computed over **accepted** entities. All 60 pilot cards committed (`pilot/analysis_facets.jsonl`).
   On-demand for any bundle via `hepkg-acquire facets <bundle>` / `compiler.bundle_facets`. Example
   (2001.06899): objects `[BJet, CJet, Electron, Jet, MET, Muon, ZCandidate]`, generators
   `[FEWZ, Herwig++, MadGraph5_aMC@NLO, PowhegBox, Pythia8, Top++]` — note **generators collapsed to
   family** (all pythia 8.x → `Pythia8`), the exact fragmentation our aliases layer was fixing.
4. **Export entity rows** now carry `canonical` + `facets` (see `entities.jsonl`).
5. **Expert review is being recorded**: 2001.06899 (122 accepts, already on main HEAD),
   2006.05880 (287 decisions: 149 accept / 130 reject / 8 correct) staged in
   `pilot/runs/2006.05880/decisions.json` (+2,657) — **not yet compiled into the bundle** (that
   bundle still has 0 expert_decisions, statuses machine_verified/quarantined).
6. Pipeline robustness: `reevaluate` command, coercion/runtime repairs, region-lexicon kind guard,
   VC-1 "twin-match before rejecting representation duplicates", verification-case-registry doc.

## What it CONFIRMS of our findings (nice validation)

- **Vocabulary now canonicalizes, not just flags** — exactly our #1 finding (it was QA-only). Now
  `canonicalize_entity` produces a name that's *used*.
- **Id collisions**: the contract now states **"347 shared ids, 752 with conflicting kind/label
  under the same id"** and that id-equality across bundles is NOT entity resolution. Matches our
  347 shared / 334 divergent.
- **Signatures empty (0/14,188)** — confirmed; guidance: implement the slot, read OR/count from
  **qualifiers** (`count: ">=2"`, `subchannels`, and a new `column`). Validates our
  compile-from-qualifiers plan (add `column`).
- **Labels are evidence-true, never rewritten** — matches our non-destructive stance.

## IMPACT ON OUR WORK (actionable)

1. **★ Importer entity identity — correct it (revisits D-024/D-029).** We built a merged `entity`
   table keyed by **bare entity_id** (union across papers). The contract now says that is **wrong**:
   cross-paper same-id is *accidental collision* (e.g. `hepkg:object:muon` in 42 papers, sometimes
   different meanings) → **key nodes by (bundle_id, entity_id)**. Our `entity_occurrence` already has
   the right grain; the merged `entity` node over-trusts bare-id equality. **Action**: demote the
   merged `entity` table from "identity" to "convenience", make (bundle_id, entity_id) the node key,
   and do cross-paper grouping via `canonical` + explicit resolution — not bare id. (Our aliases
   `same_as` is already an explicit-resolution layer, so it's aligned; the importer merge is the bit
   to fix.)
2. **★ Aliases layer — reposition, don't scrap.** Gabriel's `canonical` map now does the clean Tier-1
   job for **detector objects + generators** (better than ours: enum, not picked id). But it's
   per-kind, 70–91% coverage, `null` for the tail, and 1:1 only for object/generator kinds (others
   get facets). The contract explicitly frames anything beyond it as a separate **`resolves_to`**
   layer — **that is our aliases layer.** Reposition: (a) *consume* `canonicalize_entity` /
   `facet_tags` as a strong prior, (b) own the residual (the null tail, cross-paper node resolution,
   kinds without a canonical), (c) emit `resolves_to`-style edges (our `same_as` ≈ this). Our
   Tier-1 spelling/dot work is now partly redundant *for the covered kinds* — still needed elsewhere.
3. **Consume the new derived data.** `canonical`/`facets` per entity + `analysis_facets.jsonl` are
   pure gold for coverage/queries (the facet card is literally a coverage backbone). Two options:
   (a) run `canonicalize_entity`/`facet_tags` ourselves over imported entities (port ~vocabulary.py),
   or (b) import from the **compiled export** instead of raw bundles. Leaning (a) for now (keeps us on
   the faithful bundle path; the vocab is versioned and portable).
4. **Export step (step 10) will need updating** if we do the byte-exact `expected-export/` match:
   entities.jsonl now has `canonical`+`facets`, and there's a new `analysis_facets.jsonl`.
5. **M4 becomes testable on real data (soon).** Review decisions now exist (2001.06899 accepted;
   2006.05880 staged). Once a *reviewed* bundle is compiled (statuses promoted + expert_decisions
   populated), the re-import reconciler has a real target beyond the `corrected_bundle` fixture.

## Open questions / decisions for Raul

- Adopt (bundle_id, entity_id) as node identity now, or keep the merged table with a caveat? (I'd
  correct it — it's the contract, and it's a small change since `entity_occurrence` exists.)
- Consume Gabriel's vocab by **porting** `vocabulary.py` (option 3a) vs importing the compiled
  export (3b)? Porting keeps us bundle-native and versioned.
- Aliases layer: relabel `same_as`/`entity_canonical` as the `resolves_to` layer and have it *build
  on* `canonical` (only resolve the residual), rather than re-deriving object/generator canonicals.
- Ask Gabriel: is `objects-v2`/`facets-v1` frozen, and will future bundles carry `canonical`/`facets`
  *in the bundle* (which would bump the schema and need our gate updated), or stay export-only?
