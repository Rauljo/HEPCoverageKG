# =============================================================================
# HEPCoverageKG aliases: orchestration
#
# Two entry points, deliberately kept separate:
#
#   build(conn)                 -- Tiers 1 + 1.5. Deterministic, offline, cheap.
#                                  No model, no network. Writes same_as proposals.
#   propose_deep_semantics(...) -- Tiers 2 / 2.5 / 3. Loads an embedding model and
#                                  calls an LLM endpoint. Writes a review JSON only.
#
# They are NOT chained. build() used to call the deep pass, which meant the cheap
# deterministic tier could not run without a GPU and a live LLM server, and made
# the test suite load a model and hit the network.
#
# propose_deep_semantics takes its output path as a REQUIRED argument. It used to
# be hardcoded relative to the working directory, so running pytest from the repo
# root silently overwrote real pipeline results.
# =============================================================================
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from pathlib import Path

from hepcoveragekg.aliases import guards, normalize, spelling, store
from hepcoveragekg.aliases.cluster import connected_components

# `semantics` (sentence-transformers) and `adjudicate` (openai) are imported inside
# propose_deep_semantics, not here -- importing this module must stay cheap so that
# build() and the test suite never pull in a model or an HTTP client.

logger = logging.getLogger(__name__)

# The CLI's default. Never applied implicitly inside the library.
DEFAULT_DEEP_OUT = Path("data/processed/aliases_proposed.json")

# In-flight LLM requests. Low by default so pointing LLM_BASE_URL at a shared,
# rate-limited API is safe out of the box; DIAS overrides it upward.
DEFAULT_CONCURRENCY = 8

# A run with more failures than this is reported as suspect rather than trusted.
ERROR_RATE_WARN = 0.02


def build(conn) -> dict[str, int]:
    """Tiers 1 + 1.5 only. Deterministic, offline, non-destructive.

    Writes same_as proposals; nothing resolves until `aliases confirm` runs.
    Tiers 2/2.5/3 are a separate call -- see propose_deep_semantics().
    """
    ents = store.entities_with_kind(conn)
    labelled = store.entities_with_label(conn)
    results: dict[str, int] = {}

    # Tier 1 — deterministic normalization (separators, case, version dot)
    results["normalize"] = store.write_proposals(conn, normalize.propose(ents))

    # Tier 1 (label) — the same folding applied to labels rather than ids. Two
    # entities whose labels are byte-identical are the same concept whatever
    # their ids look like, and the id key cannot see that (D-050).
    label_props = normalize.propose_labels(labelled)
    for method in ("label_exact", "label_norm"):
        results[method] = store.write_proposals(
            conn, [p for p in label_props if p.method == method]
        )

    # Tier 1.5 — deterministic US/UK spelling + data-driven plural
    results["spelling"] = store.write_proposals(conn, spelling.propose(ents))

    return results


def propose_deep_semantics(
    conn,
    out_path: Path | str,
    concurrency: int | None = None,
    dry_run: bool = False,
    stability_sample: int = 0,
    stability_repeats: int = 3,
) -> dict[str, int]:
    """Tiers 2 (candidates), 2.5 (guards) and 3 (LLM adjudication) over entity labels.

    Requires an embedding model and a reachable LLM endpoint. Results are written
    to `out_path` as JSON for human review -- deliberately NOT into same_as yet.

    `concurrency` caps in-flight LLM requests (env LLM_CONCURRENCY, default 8).
    The default is deliberately low: it is safe against a shared, rate-limited
    API. Raise it for a dedicated vLLM server you own.

    `dry_run` stops after the guards and reports how many pairs WOULD be sent,
    per kind. No LLM is contacted and nothing is written -- run this before
    committing to a long job, since a threshold change can move the candidate
    count by an order of magnitude.

    `stability_sample` (pairs per kind, 0 = off) re-adjudicates a sample
    `stability_repeats` times at the end and reports how often the verdict holds.
    Temperature is 0.0 but vLLM is not bit-deterministic, and an unmeasured noise
    floor makes any later precision comparison unreadable.
    """
    # Deferred so that importing this module (and therefore build(), and the test
    # suite) never loads sentence-transformers or an HTTP client.
    from hepcoveragekg.aliases import adjudicate, context, semantics

    if concurrency is None:
        concurrency = int(os.environ.get("LLM_CONCURRENCY", DEFAULT_CONCURRENCY))
    out_path = Path(out_path)
    logger.info("Starting Deep Semantic Pipeline (Tiers 2-3)...")

    cursor = conn.execute("SELECT entity_id, label, kind FROM entity")
    kind_to_labels = collections.defaultdict(set)
    label_to_ids = collections.defaultdict(list)

    for row in cursor:
        eid, label, kind = row["entity_id"], row["label"], row["kind"]
        label = label or ""
        kind_to_labels[kind].add(label)
        label_to_ids[label].append(eid)

    all_candidates = []
    per_kind: dict[str, dict[str, int]] = {}

    # Phase A: Generate Candidates
    for kind, labels in kind_to_labels.items():
        if len(labels) < 2:
            continue
        candidates = semantics.generate_candidates(list(labels))
        for a, b in candidates:
            all_candidates.append((a, b, kind))
        per_kind[kind] = {
            "labels": len(labels),
            "pairs": len(labels) * (len(labels) - 1) // 2,
            "candidates": len(candidates),
            "after_guards": 0,
        }

    logger.info(f"Phase A (Embeddings + Jaccard) proposed {len(all_candidates)} candidates.")

    # Phase B: Semantic Guards
    surviving_candidates = []
    for a, b, kind in all_candidates:
        passed, reason = guards.passes_semantic_guards(a, b)
        if passed:
            surviving_candidates.append((a, b, kind))
            per_kind[kind]["after_guards"] += 1
        else:
            logger.debug(f"Vetoed '{a}' vs '{b}': {reason}")

    logger.info(f"Phase B (Semantic Guards) surviving candidates: {len(surviving_candidates)}")

    if dry_run:
        # Stop before spending anything. Report per kind, because the totals hide
        # where the candidates actually come from.
        logger.info("DRY RUN -- no LLM calls, nothing written")
        header = f"{'kind':26s} {'labels':>7s} {'pairs':>10s} {'cands':>8s} {'post-guard':>11s}"
        logger.info(header)
        for kind, s in sorted(per_kind.items(), key=lambda kv: -kv[1]["candidates"]):
            logger.info(
                f"{kind:26s} {s['labels']:7d} {s['pairs']:10,d} "
                f"{s['candidates']:8,d} {s['after_guards']:11,d}"
            )
        stats = {
            "dry_run": 1,
            "candidates": len(all_candidates),
            "after_guards": len(surviving_candidates),
            "vetoed": len(all_candidates) - len(surviving_candidates),
        }
        stats.update({f"kind_{k}": v["after_guards"] for k, v in per_kind.items()})
        return stats

    # Phase B.5: labels -> entity ids, and the graph context for each.
    #
    # Two defects fixed here together, both recorded on 2026-07-28 and still open
    # on 2026-08-02:
    #
    #   keyed by label, not entity_id.  `label_to_ids` was built and never read,
    #       and **144 labels map to more than one entity_id** -- so a verdict on
    #       a label could not be written back to `same_as` at all. Expanding to
    #       id pairs costs almost nothing (mean 1.04 ids per label, ~1.08x pairs)
    #       and makes the output writable.
    #
    #   context built but never sent.  `aliases/context.py` was wired into
    #       `evaluate.py` (the trial set) and NOT into this pipeline, so quality
    #       was measured on one system and proposals generated by another. It is
    #       the fix aimed squarely at the observed errors: `|eta| of LEADING jet`
    #       vs `|eta| of SUBLEADING jet` are near-identical as strings and have
    #       completely different defining sentences.
    #
    # Context is read from SQLite up front because the connection is not safe to
    # share across the async phase -- the same reason, and the same shape, as
    # `evaluate._adjudicate_all`.
    id_pairs: list[tuple[str, str, str, str, str]] = []  # (id_a, id_b, term_a, term_b, kind)
    for a, b, kind in surviving_candidates:
        for id_a in label_to_ids.get(a, [a]):
            for id_b in label_to_ids.get(b, [b]):
                id_pairs.append((id_a, id_b, a, b, kind))

    wanted = {i for p in id_pairs for i in (p[0], p[1])}
    context.load_section_priors(conn)
    blocks: dict[str, str] = {i: context.fetch(conn, i).render("ENTITY A") for i in wanted}
    logger.info(
        f"Phase B.5: {len(surviving_candidates)} label pairs -> {len(id_pairs)} entity-id pairs; "
        f"context for {len(blocks)} entities"
    )

    # Phase C: LLM Adjudication
    final_proposals = []
    approved_edges = []
    rejected_edges = []
    error_types: collections.Counter = collections.Counter()

    logger.info(f"Invoking LLM on {len(id_pairs)} pairs (concurrency {concurrency})...")

    # Streamed to disk as verdicts arrive, not accumulated and written once.
    # A 136k-pair run is an overnight job against a 24h Slurm cap: a single
    # `json.dump` at the end means a crash at pair 120,000 loses everything.
    # Same reasoning as the evaluation runner flushing every record.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path = out_path.with_suffix(".jsonl")

    async def process_candidates(sink):
        # Cap in-flight requests. 100 suits a dedicated vLLM; a shared, rate-limited
        # API needs far less, hence the parameter.
        sem = asyncio.Semaphore(concurrency)

        async def fetch(id_a, id_b, a, b, kind):
            async with sem:
                ctx_a = blocks.get(id_a, "")
                ctx_b = blocks.get(id_b, "")
                # render() stamps the name at fetch time; re-label B.
                if ctx_b.startswith("ENTITY A:"):
                    ctx_b = "ENTITY B:" + ctx_b[len("ENTITY A:"):]
                res = await adjudicate.adjudicate_pair(
                    a, b, kind=kind, context_a=ctx_a, context_b=ctx_b
                )
                return id_a, id_b, a, b, kind, res

        done = []
        tasks = [fetch(*p) for p in id_pairs]
        for n, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            done.append(result)
            sink(result)
            if n % 2000 == 0:
                logger.info(f"  adjudicated {n:,} / {len(tasks):,}")
        return done

    with stream_path.open("w", encoding="utf-8") as stream:
        def sink(result) -> None:
            id_a, id_b, a, b, kind, res = result
            stream.write(json.dumps({
                "id_a": id_a, "id_b": id_b, "term_a": a, "term_b": b, "kind": kind,
                "status": res["status"], "is_match": res["is_match"],
                "confidence": res["confidence"], "explanation": res["explanation"],
                "error_type": res["error_type"],
            }, ensure_ascii=False) + "\n")
            stream.flush()

        results = asyncio.run(process_candidates(sink))

    logger.info(f"Streamed {len(results):,} verdicts to {stream_path}")

    for id_a, id_b, a, b, kind, res in results:
        entry = {
            # Ids first: they are what a writeback needs, and what was missing.
            "id_a": id_a,
            "id_b": id_b,
            "term_a": a,
            "term_b": b,
            "kind": kind,
            "ambiguous_label": len(label_to_ids.get(a, [])) > 1
                               or len(label_to_ids.get(b, [])) > 1,
            "status": res["status"],
            "is_match": res["is_match"],
            "confidence": res["confidence"],
            "explanation": res["explanation"],
            "error_type": res["error_type"],
        }

        # Route on status: a failed call is NOT a negative verdict, and must not
        # reach approved_edges or rejected_edges.
        if res["status"] != "ok":
            error_types[res["error_type"]] += 1
        elif res["is_match"]:
            approved_edges.append((id_a, id_b))
        else:
            rejected_edges.append((id_a, id_b))

        final_proposals.append(entry)

    # Phase D: Transitivity Consistency Checker
    #
    # A rejected pair that other approvals bridge anyway is a contradiction: at
    # least one of the three verdicts is wrong. That makes paradoxes a **free
    # precision signal** -- no labels, no human, no second model -- so they are
    # counted and attached to the records rather than only written to a log
    # nobody aggregates. A rising paradox rate between two runs means the
    # adjudicator got less self-consistent, which is measurable the same night.
    clusters = connected_components(approved_edges)
    paradoxes: list[tuple[str, str]] = []
    membership: dict[str, int] = {}
    for i, cluster_set in enumerate(clusters):
        for node in cluster_set:
            membership[node] = i
    for a, b in rejected_edges:
        if a in membership and membership.get(a) == membership.get(b):
            paradoxes.append((a, b))
            logger.warning(
                f"TRANSITIVITY PARADOX: rejected '{a}' == '{b}', but other approvals "
                f"bridge them into one cluster"
            )
    # TWO different flags, on two different sides, and conflating them made
    # "paradoxes first" a no-op in the review sample on 2026-08-02:
    #
    #   transitivity_paradox   on a REJECTED pair: approvals bridge it anyway,
    #                          so the model contradicted itself about this pair.
    #   in_tainted_cluster     on an APPROVED pair: it sits inside a component
    #                          that contains such a contradiction, so the merge
    #                          may be a link in a bad chain.
    #
    # The review file samples matches, so only the second one can ever apply
    # there. Flagging just the first meant the sample prioritised nothing.
    paradox_pairs = set(paradoxes)
    for entry in final_proposals:
        pair = (entry["id_a"], entry["id_b"])
        entry["transitivity_paradox"] = pair in paradox_pairs
        entry["in_tainted_cluster"] = (
            bool(entry.get("is_match")) and membership.get(entry["id_a"]) in
            {membership[a] for a, b in paradox_pairs if a in membership}
        )
    if paradoxes:
        rate = len(paradoxes) / max(len(rejected_edges), 1)
        logger.warning(
            f"{len(paradoxes)} transitivity paradoxes among {len(rejected_edges)} "
            f"rejections ({rate:.1%}) -- each implies a wrong verdict somewhere"
        )

    # Bring the streamed file up to date now that the flags exist.
    #
    # It is written during Phase C so a crash keeps completed work, which means
    # it necessarily predates Phase D. Leaving it stale is not harmless: every
    # downstream tool reads this file, and on 2026-08-02 that silently disabled
    # paradox-first ordering in the review sample -- the file looked complete and
    # the flag was simply absent from every record.
    with stream_path.open("w", encoding="utf-8") as stream:
        for entry in final_proposals:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Sort and output JSON
    final_proposals.sort(key=lambda x: x.get("confidence", 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_proposals, f, indent=2)

    logger.info(f"Wrote {len(final_proposals)} adjudicated proposals to {out_path}")

    n_errors = sum(error_types.values())
    if n_errors:
        rate = n_errors / len(final_proposals) if final_proposals else 0.0
        breakdown = ", ".join(f"{k}={v}" for k, v in error_types.most_common())
        message = f"{n_errors} of {len(final_proposals)} calls failed ({rate:.1%}): {breakdown}"
        if rate > ERROR_RATE_WARN:
            # Loud: the verdicts are incomplete, so any downstream count over the
            # negatives (e.g. a contradiction analysis) is measuring partly noise.
            logger.error(f"HIGH ERROR RATE -- {message}. Treat these results as incomplete.")
        else:
            logger.warning(message)

    if stability_sample:
        stats_stability = _stability_check(
            final_proposals, blocks, concurrency, stability_sample,
            stability_repeats, out_path,
        )
    else:
        stats_stability = {}

    stats = {
        "candidates": len(all_candidates),
        "after_guards": len(surviving_candidates),
        "id_pairs": len(id_pairs),
        "matched": len(approved_edges),
        "rejected": len(rejected_edges),
        "errors": n_errors,
        "written": len(final_proposals),
        "paradoxes": len(paradoxes),
        "clusters": len(clusters),
    }
    stats.update({f"stability_{k}": v for k, v in stats_stability.items()})
    stats.update({f"error_{k}": v for k, v in error_types.items()})
    return stats


def stratified_matches(records: list[dict], per_kind: int = 8, seed: int = 0) -> list[dict]:
    """Pick a fair sample of MATCHES: a few from every kind, paradoxes first.

    Stratified because precision is not uniform across kinds --
    systematic_uncertainty supplied 1,336 of 4,238 matches in the stale run, so a
    uniform sample would be a third systematics and would tell you almost nothing
    about generators or observables.

    Shared by the human review file and the stability re-check **on purpose**: the
    pairs a person labels are then the pairs whose run-to-run stability is known,
    so a disagreement between human and model reads as *"the model is wrong"*
    rather than *"the model is unstable"*. Those are different problems with
    different fixes, and one sample answers both.
    """
    import random

    matches = [r for r in records if r.get("is_match") and r.get("status", "ok") == "ok"]
    by_kind: dict[str, list] = {}
    for r in matches:
        by_kind.setdefault(r.get("kind", "?"), []).append(r)

    rng = random.Random(seed)
    chosen: list = []
    for _kind, rows in sorted(by_kind.items()):
        # `in_tainted_cluster` is the flag that applies to matches -- see the
        # note in propose_deep_semantics on why the other one never fires here.
        suspect = lambda r: r.get("in_tainted_cluster") or r.get("transitivity_paradox")  # noqa: E731
        paradox = [r for r in rows if suspect(r)]
        rest = [r for r in rows if not suspect(r)]
        take = paradox[:per_kind]
        if len(take) < per_kind:
            take += rng.sample(rest, min(per_kind - len(take), len(rest)))
        chosen += take
    return chosen


def _stability_check(records: list[dict], blocks: dict[str, str], concurrency: int,
                     per_kind: int, repeats: int, out_path: Path) -> dict:
    """Re-adjudicate a sample and report how often the verdict holds.

    Temperature is 0.0, so identical results look guaranteed -- and they are not.
    vLLM is not bit-deterministic: batching changes the order of floating-point
    reductions, so the same prompt can land on a different verdict depending on
    what else was in flight. Assuming determinism because temperature is zero is
    the mistake this exists to prevent.

    Why it matters concretely: if the adjudicator flips 8% of verdicts between
    two identical runs, a 5% precision gain from adding graph context is noise.
    The harness learned the same lesson for the query layer (S-52), where the
    spread turned out to be zero and every ablation difference became readable.

    Cheap on purpose -- a few hundred pairs against a 136k-pair job.
    """
    from hepcoveragekg.aliases import adjudicate

    sample = stratified_matches(records, per_kind=per_kind, seed=0)
    if not sample or repeats < 2:
        return {}

    logger.info(f"Phase E: re-checking {len(sample)} pairs x {repeats - 1} more run(s)")

    async def one_pass():
        sem = asyncio.Semaphore(concurrency)

        async def fetch(r):
            async with sem:
                ctx_a = blocks.get(r["id_a"], "")
                ctx_b = blocks.get(r["id_b"], "")
                if ctx_b.startswith("ENTITY A:"):
                    ctx_b = "ENTITY B:" + ctx_b[len("ENTITY A:"):]
                res = await adjudicate.adjudicate_pair(
                    r["term_a"], r["term_b"], kind=r.get("kind", "concept"),
                    context_a=ctx_a, context_b=ctx_b,
                )
                return res["is_match"] if res["status"] == "ok" else None

        return await asyncio.gather(*[fetch(r) for r in sample])

    runs = [[r["is_match"] for r in sample]]
    for _ in range(repeats - 1):
        runs.append(asyncio.run(one_pass()))

    per_pair = []
    stable = 0
    for i, r in enumerate(sample):
        verdicts = [run[i] for run in runs]
        agrees = len({v for v in verdicts if v is not None}) <= 1
        stable += bool(agrees)
        per_pair.append({
            "id_a": r["id_a"], "id_b": r["id_b"], "kind": r.get("kind", ""),
            "term_a": r.get("term_a"), "term_b": r.get("term_b"),
            "verdicts": verdicts, "stable": agrees,
        })

    rate = stable / len(sample)
    report = {"sampled": len(sample), "repeats": repeats, "stable": stable,
              "agreement": round(rate, 4), "pairs": per_pair}
    out_path.with_suffix(".stability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Loud when unstable, because every downstream comparison inherits it.
    message = (f"Phase E: {stable}/{len(sample)} verdicts stable across {repeats} runs "
               f"({rate:.1%})")
    if rate < 0.95:
        logger.error(f"{message} -- a precision difference smaller than "
                     f"{1 - rate:.0%} is NOT readable from this adjudicator.")
    else:
        logger.info(message)
    return {k: v for k, v in report.items() if k != "pairs"}


def sample_for_review(proposals_path: Path | str, out_path: Path | str,
                      per_kind: int = 8, seed: int = 0) -> dict[str, int]:
    """A stratified random sample of MATCHES, ready to be labelled by hand.

    D-044 records that precision was judged by reading fourteen rows and calling
    it "roughly half". That is not a measurement, and without a tool the next
    run gets judged the same way. This writes a review file with an empty
    `verdict` column: fill it with y/n and the precision is a division.

    Stratified **by kind**, because precision is not uniform across them --
    systematic_uncertainty supplied 1,336 of 4,238 matches in the stale run and
    would dominate a uniform sample, hiding whatever the smaller kinds do.
    Paradoxical pairs are always included: they are the cases where the
    adjudicator already contradicted itself, so they are the most informative
    rows a human can spend time on.
    """
    records = load_proposals(proposals_path)
    matches = [r for r in records if r.get("is_match") and r.get("status", "ok") == "ok"]
    by_kind = {r.get("kind", "?") for r in matches}
    chosen = stratified_matches(records, per_kind=per_kind, seed=seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Fill `verdict` with y (same thing) or n (different). Leave blank to skip.\n")
        fh.write("# SUSPECT = this merge sits in a chain the model contradicted "
                 "somewhere; check it first.\n")
        fh.write("# verdict\tkind\tconfidence\tsuspect\tterm_a\tterm_b\texplanation\n")
        for r in chosen:
            fh.write("\t".join([
                "", r.get("kind", ""), str(r.get("confidence", "")),
                "SUSPECT" if (r.get("in_tainted_cluster")
                              or r.get("transitivity_paradox")) else "",
                str(r.get("term_a", "")), str(r.get("term_b", "")),
                str(r.get("explanation", ""))[:160].replace("\t", " "),
            ]) + "\n")

    return {"matches": len(matches), "kinds": len(by_kind), "sampled": len(chosen)}


def load_proposals(path: Path | str) -> list[dict]:
    """Read a proposals file, whichever form it is in.

    Runs before 2026-08-02 wrote one JSON array at the end; runs after stream
    JSONL as verdicts arrive (so a killed overnight job keeps what it finished).
    Both are still around, and a reader that only understood one of them would
    silently return nothing for the other.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def writeback_plan(proposals_path: Path | str) -> dict:
    """Classify approved pairs for writeback. **Pairs only -- never chains.**

    Deduplication answers a question about IDENTITY: are these two strings the
    same thing? It emits pairs. Deciding that `Pythia 8.230` and `Pythia 8.212`
    belong in one FAMILY is a different question, and it belongs to the grouping
    layer (S-28: clusters collapse, groups expand).

    Conflating the two is what produced the 2026-08-02 failure. Taking the
    transitive closure of 2,262 approved pairs gave 330 clusters, the largest
    holding **75 statistical methods** -- profile likelihood, the CLs procedure,
    and log-normal nuisance constraints welded into one node because each
    NEIGHBOURING pair looked similar. Nobody ever judged the ends of that chain,
    and they are obviously different. This is the classic entity-resolution
    failure: transitive closure over pairwise decisions.

    So the plan chains nothing. Each approved pair is classified:

      clean    no explicit rejection anywhere in the connected component it
               would join. Safe to propose.
      paradox  the component contains a pair the model itself rejected, so the
               component is provably inconsistent. The PAIR may still be right --
               it is the chain that is wrong -- so these are held for review
               rather than dropped.

    Non-destructive either way: `entity_canonical` is a lookup, the original
    entities and their per-paper occurrences are untouched, and a wrong merge
    costs wrong counts until it is deleted rather than lost data.
    """
    records = load_proposals(proposals_path)
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    approved = [r for r in ok if r.get("is_match")]
    rejected_pairs = {
        tuple(sorted((r.get("id_a", r.get("term_a")), r.get("id_b", r.get("term_b")))))
        for r in ok if not r.get("is_match")
    }

    edges = [(r.get("id_a", r.get("term_a")), r.get("id_b", r.get("term_b")))
             for r in approved]
    clusters = connected_components(edges)
    membership: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            membership[node] = i

    # Components a rejection falls inside are inconsistent by construction.
    tainted = {membership[a] for a, b in rejected_pairs
               if a in membership and membership.get(a) == membership.get(b)}

    clean, paradox = [], []
    for r in approved:
        a, b = r.get("id_a", r.get("term_a")), r.get("id_b", r.get("term_b"))
        (paradox if membership.get(a) in tainted else clean).append(r)

    return {
        "clean": clean,
        "paradox": paradox,
        "clusters": clusters,
        "tainted_clusters": sorted(tainted),
        "counts": {"approved": len(approved), "clean": len(clean),
                   "paradox": len(paradox), "clusters": len(clusters),
                   "tainted_clusters": len(tainted)},
    }


def cluster_report(conn, proposals_path: Path | str, out_path: Path | str,
                   min_size: int = 10) -> dict[str, int]:
    """List the big would-be clusters for a human to look at.

    A cluster can be wrong even when every pair inside it looks right, so
    sampling pairs alone cannot catch it -- this is the other half of the review,
    and it is a very short list: 17 clusters at size >= 15 on the 2026-08-02 run.
    Nothing here is applied; it is a reading list.
    """
    plan = writeback_plan(proposals_path)
    big = [c for c in plan["clusters"] if len(c) >= min_size]
    big.sort(key=len, reverse=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Would-be clusters if approved pairs were chained transitively.\n")
        fh.write("# NOT applied -- deduplication writes pairs only (see writeback_plan).\n")
        fh.write("# For each: are these really ONE thing, or a family that grouping\n")
        fh.write("# should keep distinct? Mark the file up freely.\n\n")
        for n, cluster in enumerate(big, 1):
            members = sorted(cluster)
            kinds: set = set()
            labels = []
            for eid in members:
                row = conn.execute(
                    "SELECT label, kind FROM entity WHERE entity_id = ?", (eid,)
                ).fetchone()
                if row:
                    kinds.add(row["kind"] or "?")
                    labels.append(f"    {row['label']}")
            fh.write(f"## cluster {n}: {len(members)} members, kinds={sorted(kinds)}\n")
            fh.write("\n".join(labels[:40]) + "\n")
            if len(labels) > 40:
                fh.write(f"    ... and {len(labels) - 40} more\n")
            fh.write("\n")

    return {"clusters_listed": len(big),
            "entities_covered": sum(len(c) for c in big),
            "min_size": min_size}
