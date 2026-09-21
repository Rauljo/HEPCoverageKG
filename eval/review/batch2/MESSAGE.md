Subject: HEPCoverageKG — second review batch (104 rows), and what your first one changed

Hi Gabriel,

Second batch attached — **104 rows**, and it is deliberately a different kind of
question from the first. Open `gabriel-splits-app.html` in a browser; it saves as
you go, so you can stop and come back. `gabriel-splits.tsv` is the same content
if you would rather work in a spreadsheet.

**Your first 199 verdicts changed three things**, and it is worth saying what,
because two of them were you catching our mistakes rather than the system's.

1. **You flagged corrupted LaTeX** — `$b\bar{b}$` showing as `$b<backspace>ar{b}$`.
   That was a bug in how we parsed the model's replies: we were rescuing
   backslashes from JSON but exempting exactly the letters HEP notation uses most
   (`\bar`, `\tau`, `\text`, `\nu`). 1,297 of them across 18 files. The graph
   itself was never wrong. Fixed, and this batch is clean.

2. **You wrote "Not a yes/no question. What to do here?"** on gf-10 through gf-15,
   and then understandably stopped. You were right — we were showing you a
   sentence and a yes/no box beside a question asking "what signal efficiency
   does the cut retain?". Those seven rows now show **our answer** and ask
   whether it is correct, with a notes box for the right value. That is a
   question you can actually answer.

3. **Your row 1 note** — that the answer is yes for a validation region and no
   for a signal region, and "this distinction is lost in the current schema" —
   turned out to be a normalisation problem rather than a missing concept. The
   role was already recorded on 580 of 771 regions, under three different keys
   with 33 spellings (`SR`, `signal_region`, `signal region`, `signal`…). It is
   normalised now: signal / control / validation / fiducial / preselection, at
   82% coverage. Thirteen papers have all three of SR, CR and VR.

**What is in this batch**

- **35 rows on gf-01, asked ONE CONDITION AT A TIME.** Last time we gave you the
  whole three-part question — search AND b-jets AND missing transverse momentum —
  with a single sentence as evidence. That was unfair and it is the failure mode
  we were supposed to be measuring: precision fell from 0.90 on single-condition
  questions to 0.30 on that one. These rows ask about one condition each, with
  its own quote.
- **60 rows where three independent reads of the same passage disagreed with
  each other.** Our own machinery could not settle these, which is exactly what
  makes them worth your time — the first batch mostly measured the easy cases.
- **7 value questions**, reframed as above.

**One thing I should be upfront about.** We are using your first batch to choose
between system configurations, which means it can no longer serve as an unbiased
measure of how good the system is. So this batch is the held-out one: I would
rather not look at it until the configuration is frozen, and then report it as
the headline number. If that changes what you would want to spend time on, say
so.

No rush, and partial is genuinely useful — the rows are ordered so the most
informative come first.

Thanks.
