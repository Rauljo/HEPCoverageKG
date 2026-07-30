# HEP notation cheat sheet

*For reading the labels, quotes and final states in the database. Written 2026-07-30.*
*Physics reference, not a project decision — no D-nnn / S-nn number.*

## The ones that actually appear in our data

| Symbol | Name | What it means here |
|---|---|---|
| **ℓ** | (script L — **not Greek**) | **lepton**, generic: electron *or* muon, sometimes tau too. The single most common symbol in our final states |
| **ν** | nu | **neutrino** — invisible, shows up as missing energy |
| **γ** | gamma | **photon** |
| **μ** | mu | **muon**. Also, confusingly, **signal strength** μ in results |
| **τ** | tau | **tau lepton** (heavy, decays fast). Also lifetime |
| **σ** | sigma | **cross-section** — the "how often does this happen" number. Also standard deviation / significance ("5σ discovery") |
| **η** | eta | **pseudorapidity** — an angle-like coordinate, how forward/central a particle is |
| **φ** (ϕ) | phi | **azimuthal angle** — around the beam pipe |
| **χ̃** | chi-tilde | **neutralino / chargino** — SUSY particles. Plain **χ²** is chi-squared, a fit statistic |
| **Δ** | Delta | **difference** — ΔR (angular separation), Δφ, ΔE |
| **Λ** | Lambda | Lambda baryon; also an **energy scale** (Λ for new physics) |
| **κ** | kappa | **coupling modifier** — κ_t, κ_λ in Higgs measurements |
| **λ** | lambda | Higgs **self-coupling** |
| **α, β** | alpha, beta | couplings / angles. α_s = strong coupling |
| **π** | pi | **pion**, the lightest hadron (also just 3.14…) |
| **ψ, Υ** | psi, Upsilon | **J/ψ** and **Υ** mesons — charm and bottom bound states |

## Non-Greek notation that matters just as much

| Notation | Meaning |
|---|---|
| **b̄** (bar over anything) | **anti**particle. `t t̄` = top + antitop pair |
| **→** | **decays to**. `W → ℓν` = a W boson decays to a lepton and a neutrino |
| **e⁺e⁻**, **μ⁺μ⁻** | charge. A **same-flavour opposite-sign** pair — very common selection |
| **ℓ±** | either charge |
| **SF / OS / SS** | same-flavour / opposite-sign / same-sign (lepton pairs) |
| **p_T** | **transverse momentum** — momentum sideways to the beam |
| **E_T^miss**, **MET** | **missing transverse energy** — what the neutrinos (or invisible new particles) carried away |
| **b-jet / b-tagged** | a jet identified as coming from a bottom quark |
| **large-R jet** | a wide jet, used when a heavy particle decays into something collimated |
| **fb⁻¹, ab⁻¹** | inverse femtobarns / attobarns — **how much data** was collected |
| **TeV** | collision energy. LHC Run 2 = 13 TeV, Run 3 = 13.6 TeV |

## Reading a real final state from our database

```
"1-lepton channel: WH→ℓν b b̄"
```

- **1-lepton channel** — the analysis looked at events with exactly one lepton
- **WH** — a W boson produced together with a Higgs boson
- **→** — which decay as follows:
- **ℓν** — the W becomes a lepton plus a neutrino (that's the 1 lepton, plus missing energy)
- **b b̄** — the Higgs becomes a bottom quark and its antiparticle (two b-jets)

So in structured form: `[{lepton, 1, exactly}, {b-jet, 2, exactly}, {MET, present}]`
— which is exactly what the M3 parse has to produce (D-038).

## Traps

- **μ is overloaded** — muon, signal strength, and pile-up all use it. Context decides.
- **σ is overloaded** — cross-section vs significance.
- **ℓ is not Greek and not a number 1.** It is a stylised lowercase L.
- **"2 leptons" ≠ "2 electrons".** A lepton is an electron *or* a muon, so a 2-lepton search
  covers more than a 2-electron one. This is the flavour-hierarchy landmine in
  [[final-state-representation]].
- **γγ vs H→γγ** — the first is a final state, the second says which particle produced it. Not the
  same claim, and easy to merge by mistake.
