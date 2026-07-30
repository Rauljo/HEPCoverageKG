# Final-state representation

**Status**: discussed — **rewritten 2026-07-29 after measuring the database**. The previous version
described a problem the data does not have, and prescribed a fix the data cannot support. Blocks M3.
**Discussed**: 2026-07-06 (schema review), 2026-07-09, **2026-07-29 (measured; direction changed)**.

## What the earlier version got wrong

It claimed `result_has_final_state` fans a signature out into separate object edges
(`electron`, `small_r_jet`, `missing_transverse_momentum`), losing both the multiplicities and the
fact that the objects form one combination — and that the fix was to *compile* signatures back
together from `count` / `subchannel` qualifiers, which were said to exist.

Both halves are false, and the second one shaped the M3 plan for three weeks:

- There is **no fan-out**. The whole final state is stored on **one** node, as prose.
- There is **no `count` qualifier**. Zero of 161. The multiplicity that did survive into a
  qualifier sits under ~6 invented key names (`lepton_multiplicity` 4, `n_leptons` 4,
  `object_multiplicity` 3, `b_tag_multiplicity` 5, `lepton_flavors`, `subchannels`) — ~15
  assertions in total. **54 of 161 (34%) have no qualifiers at all.**

So there was never anything to compile. This is why the "the ingredients aren't empty" note in
`backlog.md` was wrong.

## What is actually there (measured 2026-07-29, `data/processed/hepkg.db`)

| | |
|---|---|
| `result_has_final_state` assertions | **161**, spread over **all 60 papers** |
| pointing at a node (`kind = channel`) | 138 |
| carrying a bare `object_value` string instead | 23 ← inconsistent representation, own problem |
| **distinct labels among the 138** | **126** |
| labels shared by more than one paper | **5** |
| `assertion.signature` populated | **0** |

Real examples:

```
"Z(→e+e-/μ+μ-) + jet(s) final state"          {"level":"reconstructed","subchannels":[...]}
"Exactly one lepton plus ≥4 b-tagged jets"    {"region_role":"signal"}
"0-lepton channel: ZH→νν b b̄"                 {"lepton_multiplicity":0,"level":"reconstructed"}
"two isolated photons plus top-quark pair"    {"reconstruction_level":"reconstructed"}
"same-flavor lepton pair (e or μ)"            {"reconstruction_level":"reconstructed"}
```

## The real problem

The combination is **not** lost and the multiplicity is **not** lost. Both are sitting in the label,
in English. What is missing is **structure, and agreement on wording**: 126 distinct phrasings for a
set of final states that is certainly far smaller. Even the 5 shared labels understate the overlap —
two of them ("Diphoton (two photon) final state" and "H -> gamma gamma (diphoton) final state") are
the same physics named twice.

This is the **aliases problem again, one level up**: not string variants of one entity, but
different English descriptions of one signature. Same shape as M1's "487 ids → 223 concepts".

Consequence: coverage counting — the project's central claim, and the place where the KG beats
document retrieval (2026-07-29 discussion) — is impossible today. "Which final states have been
measured at 13 TeV" returns 126 near-unique strings.

## The fix (direction agreed 2026-07-29)

Two steps, deliberately split at the point where consistency starts to matter.

**1. Parse — LLM.** Read each label (plus its evidence quote and qualifiers) and emit *structure*:
`"Exactly one lepton plus ≥4 b-tagged jets"` → `[{lepton, 1, exactly}, {b_jet, 4, at_least}]`.
Prose is the only source, so this can only be a reading job. **161 items** — small enough to run
several times and use the agreement rate as a free reliability number.

**2. Canonicalise — plain code.** Sort the structure and serialise it to the id (D-015 already
settles that identical compositions must collapse to one id). **The LLM must never write the id
itself**: it would emit `1L+4b` once and `1lep_4bjet` the next time, silently splitting the counts
the whole coverage claim rests on. Model reads; code names.

**3. Group — LLM (separate, later).** `2e+MET` and `2μ+MET` are different signatures that are both
"two same-flavour leptons + MET"; a 2-lepton search *covers* the 2-electron case. That is genuine
semantic judgement and belongs to the grouping layer, not to the parse. The corpus is already full
of it: "same-flavor lepton pair (e or μ)", "0/1/2-lepton channel", "Z(→e+e-/μ+μ-)".

## Still open

- **Composite node vs reconstruct at query time.** Recommended 2026-07-29: **store the signature as
  a node**. The 2026-07-09 position (defer, rebuild later) was reasonable when a human wrote each
  query; it is worse now that the query system is the front door — a stored node is one lookup, is
  something the router can point at and the grouping layer can cluster, whereas a rebuild rule has
  to be re-derived correctly by an agent on every question. **Not signed off.**
- **The 23 `object_value` rows.** Same predicate, no node. Decide whether to promote them.
- **Flavour hierarchy** (unchanged, still the landmine): "2 leptons" ≠ "2 electrons", but
  lepton ⊃ electron, muon. Conflating them merges distinct searches; ignoring the relation loses
  real coverage. Needs an explicit small hierarchy, and it is step 3's job, not step 1's.

Cross-links: [[open-vocab-reconciliation]] (same problem, entity level) · [[gap-hypothesis-system]]
(consumes the signatures) · [[grounding-and-evaluation]].
