# =============================================================================
# HEPCoverageKG: command-line importer
#
#   python -m hepcoveragekg.cli import <path>      # a bundles dir or one file
#   python -m hepcoveragekg.cli verify-counts      # check the milestone-1 target
#
# One bad bundle is reported and the run continues; the command exits non-zero
# if anything failed. verify-counts checks the pilot target and sets its exit
# code accordingly. See vault/ideas/bundle-importer-design.md (build step 8).
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from hepcoveragekg.ingest import importer
from hepcoveragekg.ingest.errors import BundleError
from hepcoveragekg.kg import queries, store

# The milestone-1 acceptance target: the 60 pilot bundles by status.
PILOT_TARGET = {"machine_verified": 11309, "quarantined": 2555, "rejected": 324}


def _find_bundles(path: Path) -> list[Path]:
    """Bundles ship as .json.gz, so a directory is scanned for those only --
    directories also hold metadata (manifest.json, README.md) that must not be
    imported. A single-file path is taken as-is (.json or .json.gz), so an
    individual fixture can still be imported directly.
    """
    if path.is_dir():
        return sorted(path.glob("*.json.gz"))
    return [path]


def _cmd_import(args) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)

    bundles = _find_bundles(Path(args.path))
    if not bundles:
        print(f"no bundles found at {args.path}", file=sys.stderr)
        return 1

    print(f"importing {len(bundles)} bundle(s) into {args.db}")
    imported = skipped = failed = 0
    for path in bundles:
        try:
            result = importer.import_bundle_file(conn, path)
        except BundleError as exc:
            failed += 1
            print(f"  ✗ {path.name} — {exc.reason}")
            continue

        if result.skipped:
            skipped += 1
            print(f"  ~ {result.paper_id} — already imported (skipped)")
        else:
            imported += 1
            print(
                f"  ✓ {result.paper_id} — "
                f"{result.inserted.get('assertion', 0)} assertions, "
                f"{result.inserted.get('entity', 0)} entities"
            )
        for warning in result.warnings:
            print(f"      ⚠ {warning}")

    print(f"\ndone: {imported} imported, {skipped} skipped, {failed} failed")
    _print_status(conn)
    return 1 if failed else 0


def _cmd_verify_counts(args) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    counts = _print_status(conn)

    missing = {k: (counts.get(k, 0), v) for k, v in PILOT_TARGET.items() if counts.get(k, 0) != v}
    total_ok = sum(counts.values()) == sum(PILOT_TARGET.values())
    if not missing and total_ok:
        print("✓ milestone-1 target met (14,188 / 11,309 / 2,555 / 324)")
        return 0
    print("✗ milestone-1 target NOT met:")
    for status, (got, want) in missing.items():
        print(f"    {status}: got {got}, want {want}")
    if not total_ok:
        print(f"    total: got {sum(counts.values())}, want {sum(PILOT_TARGET.values())}")
    return 1


def _print_status(conn) -> dict[str, int]:
    counts = queries.status_counts(conn)
    total = sum(counts.values())
    parts = ", ".join(f"{status} {n}" for status, n in sorted(counts.items()))
    print(f"status: {parts or '(empty)'}  (total {total})")
    return counts


def _cmd_aliases(args) -> int:
    from hepcoveragekg.aliases import run as aliases_run, store as aliases_store, report as aliases_report

    conn = aliases_store.connect(args.db)
    if args.action == "build":  # Tiers 1 + 1.5 only: deterministic, offline, no model
        added = aliases_run.build(conn)
        total = conn.execute("SELECT COUNT(*) FROM same_as").fetchone()[0]
        print("aliases build — proposals written (nothing resolves until confirmed):")
        for method, n in added.items():
            print(f"  {method}: +{n}")
        print(f"  same_as rows now: {total}")
        return 0
    if args.action == "deep":  # Tiers 2/2.5/3: needs an embedding model + LLM endpoint
        out = Path(args.deep_out or aliases_run.DEFAULT_DEEP_OUT)
        if args.dry_run:
            print("aliases deep --dry-run — Phases A+B only, no LLM calls, nothing written")
        else:
            print(f"aliases deep — Tiers 2/2.5/3 (embeddings + guards + LLM) -> {out}")
        stats = aliases_run.propose_deep_semantics(
            conn, out, concurrency=args.concurrency, dry_run=args.dry_run,
            stability_sample=args.stability_sample,
            stability_repeats=args.stability_repeats,
        )
        for key, n in stats.items():
            print(f"  {key}: {n}")
        if stats.get("errors"):
            rate = stats["errors"] / max(stats["written"], 1)
            print(f"  WARNING: {rate:.1%} of calls failed — these pairs have NO verdict.")
            print("  Do not read the negatives as rejections until they are re-run.")
        if args.dry_run:
            print(f"  would send {stats['after_guards']:,} pairs to the LLM; nothing written")
        else:
            print("  review the JSON by hand; nothing is written to same_as yet")
        return 0
    if args.action == "clusters":
        # Deliberately its own action, not a phase of `deep`: deduplication
        # answers identity and emits pairs, grouping decides families. Also
        # means a prompt change costs minutes, not another full pairwise pass.
        from hepcoveragekg.aliases import clusters as aliases_clusters
        from hepcoveragekg.query import templates as _T

        src = Path(args.deep_out or aliases_run.DEFAULT_DEEP_OUT)
        stats = aliases_clusters.check_clusters(
            _T.read_only(args.db), src, Path(args.out) / "cluster_verdicts.jsonl",
            concurrency=args.concurrency or 16,
        )
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.action == "confidence":
        # Temperature > 0, sampled. Measures the MODEL's uncertainty, which
        # greedy decoding hides. Separate from the temperature-0 stability check,
        # which measures whether the serving stack is deterministic.
        from hepcoveragekg.aliases import confidence as aliases_confidence
        from hepcoveragekg.query import templates as _T2

        src = Path(args.deep_out or aliases_run.DEFAULT_DEEP_OUT)
        stats = aliases_confidence.measure(
            _T2.read_only(args.db), src, Path(args.out) / "pair_agreement.jsonl",
            samples=args.samples, temperature=args.temperature,
            concurrency=args.concurrency or 16,
        )
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.action == "report":
        info = aliases_report.write_report(conn, args.out)
        print(f"draft alias list: {info['clusters']} clusters / {info['ids']} ids")
        print(f"  {info['markdown']}")
        print(f"  {info['csv']}")
        return 0
    if args.action == "confirm":  # promote proposed -> auto (run only after review)
        n = aliases_store.confirm(conn, method=args.method)
        mapped = aliases_store.materialize_canonical(conn)
        print(f"confirmed {n} proposals; {mapped} entities now resolve to a canonical id")
        return 0
    return 1


def _cmd_facets(args) -> int:
    from hepcoveragekg.facets import derive as facets_derive

    conn = facets_derive.connect(args.db)

    if args.action == "derive":
        info = facets_derive.derive_facets(conn)
        print(f"facets derive ({info['vocabulary']}) — "
              f"{info['rows']} rows over {info['entities_tagged']} entities")
        print(f"  {'kind':24} {'entities':>8} {'matched':>8} {'missed':>7} {'hit':>6}")
        for kind, c in info["coverage"].items():
            print(f"  {kind:24} {c['entities']:8} {c['matched']:8} "
                  f"{c['missed']:7} {c['hit_rate']:5.1%}")
        # A miss is silent: the entity keeps its label and facts, it is just
        # invisible to facet navigation. Printed so it cannot go unnoticed.
        total = sum(c["entities"] for c in info["coverage"].values())
        missed = sum(c["missed"] for c in info["coverage"].values())
        print(f"  {'TOTAL':24} {total:8} {total - missed:8} {missed:7} "
              f"{(total - missed) / total:5.1%}" if total else "")
        return 0

    if args.action == "signatures":
        info = facets_derive.derive_signatures(conn)
        print(f"signatures derive ({info['vocabulary']})")
        print(f"  candidate assertions      {info['candidates']}")
        print(f"  native signatures         {info['native_signatures']}")
        print(f"  derived                   {info['derived']}")
        print(f"  of which OR-group members {info['or_group_members']}")
        print(f"  carrying a subchannel flag{info['with_subchannel_flag']:>4}"
              "   <- the OR rule's only signal")
        return 0

    if args.action == "gaps":
        labels = facets_derive.unmatched_labels(conn, args.kind, limit=args.limit)
        print(f"{len(labels)} unmatched '{args.kind}' labels (vocabulary blind spots):")
        for label in labels:
            print(f"  {label}")
        return 0

    if args.action == "card":
        card = facets_derive.card(conn, args.paper)
        coverage = facets_derive.card_coverage(conn, args.paper)
        print(json.dumps(card, indent=2, sort_keys=True))
        print("\ncoverage (what the card leaves out):")
        for kind, c in sorted(coverage.items()):
            flag = "  <- partial" if c["missed"] else ""
            print(f"  {kind:24} {c['matched']:4}/{c['total']:<4} matched{flag}")
        return 0

    return 1


def _cmd_reader(args) -> int:
    """The reference reader: answer the supervisor's questions from the PAPERS.

    Independent of the graph on purpose. His gold is graph-agreement gold, so
    scoring against it measures whether two pipelines built from one extraction
    agree with each other — where the extraction dropped something, both drop it
    and both score 100%.
    """
    import asyncio

    from hepcoveragekg.eval import reader as R, supervisor as S
    from hepcoveragekg.query.templates import read_only

    conn = read_only(args.db)

    records = S.build_records()
    if args.qid:
        wanted = set(args.qid.split(","))
        records = [r for r in records if r["qid"] in wanted]
    if args.scope == "sweep":
        records = [r for r in records if r["provenance"]["paper_scope"] is None]
    elif args.scope == "single":
        records = [r for r in records if r["provenance"]["paper_scope"] is not None]
    records = [r for r in records if not r["provenance"].get("blocked")]

    if not records:
        print("no questions selected")
        return 1

    all_papers = [r[0] for r in conn.execute("SELECT arxiv_id FROM paper ORDER BY arxiv_id")]

    def papers_for(record: dict) -> list[str]:
        # Scope comes from the question TEXT (an arXiv id in it), never from the
        # gold answer -- scoping to the papers the answer names could only ever
        # confirm the answer.
        return record["provenance"]["paper_scope"] or all_papers

    print(f"reader — {len(records)} question(s), "
          f"{sum(len(papers_for(r)) for r in records)} paper-reads, "
          f"repeats={args.repeats}, cascade={not args.no_cascade}")
    for r in records:
        scope = r["provenance"]["paper_scope"]
        print(f"  {r['qid']}  {'all 60' if scope is None else ','.join(scope):22} "
              f"{r['text'][:62]}")

    if args.rescore:
        # Every consensus rule is derived from the stored verdicts, so a scoring
        # change costs a file read rather than 22,000 LLM calls.
        info = R.rescore(args.rescore)
        print(f"rescored {info['reads']} reads, {info['changed']} changed")
        for k in ("yes", "no", "split"):
            print(f"  {k}: {info.get(k, 0)}")
        print(f"-> {info['out']}")
        return 0

    if args.recheck:
        # Stage 2: only the papers stage 1 rejected, with the stronger model.
        info = asyncio.run(R.recheck(
            conn, args.recheck, records, concurrency=args.concurrency))
        print(f"recheck with {info['model']}")
        print(f"  rechecked  {info['rechecked']} (stage-1 no/split)")
        print(f"  untouched  {info['untouched']} (already yes)")
        print(f"  RECOVERED  {info['recovered']}")
        print(f"-> {info['out']}")
        return 0

    if args.check_support:
        # The second pass: a verified quote proves the sentence is in the paper,
        # not that it answers the question. Measured on gf-08, half the cited
        # sentences answered something else.
        # The judge is given the same per-paper phrasing the reader was, or it
        # repeats the reader's own failure: refusing a good quote because one
        # sentence cannot name which of 60 papers do something.
        judged = [{"qid": r["qid"], "text": r.get("per_paper") or r["text"]}
                  for r in records]
        client = model = None
        if args.judge_url:
            # A judge on a DIFFERENT model from the reader: a model marking its
            # own homework is the self-preference problem, and both servers are
            # already running.
            from openai import AsyncOpenAI
            import os as _os
            client = AsyncOpenAI(base_url=args.judge_url,
                                 api_key=_os.environ.get("LLM_API_KEY", "dummy"),
                                 timeout=900)
            model = args.judge_model
        info = asyncio.run(R.verify_supports(
            args.check_support, judged, concurrency=args.concurrency,
            client=client, model=model, max_tokens=args.judge_max_tokens))
        print(f"checked {info['checked']} yes-answers")
        print(f"  upheld     {info.get('upheld', 0)}")
        print(f"  downgraded {info.get('downgraded', 0)}")
        if info["precision"] is not None:
            print(f"  precision  {info['precision']:.0%}")
        print(f"-> {info['out']}")
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing called")
        return 0

    # Existence for the corpus sweep ("which analyses do X?"); extraction for
    # the questions that name a paper, which ask WHAT or WHY. Answering "what is
    # the observed 95% CL limit on the stop mass" with yes/no throws away the
    # number that makes it an answer.
    mode = R.EXTRACTION if args.scope == "single" else R.EXISTENCE
    print(f"  mode: {mode}")

    # A sweep question is asked one paper at a time, in its per-paper form.
    asked = [{**r, "text": r.get("per_paper") or r["text"]} for r in records]

    meta = asyncio.run(R.run(
        conn, asked, papers_for, args.out,
        repeats=args.repeats, temperature=args.temperature,
        concurrency=args.concurrency, cascade=not args.no_cascade,
        limit_papers=args.limit_papers, mode=mode,
    ))
    print(f"\n-> {args.out}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_graph(args) -> int:
    from hepcoveragekg.kg import export
    import subprocess
    
    if args.action == "export":
        conn = store.connect(args.db)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        export.export_csvs(conn, out_dir)
        print(f"Graph CSVs exported to {out_dir}")
        return 0
    elif args.action == "import":
        script = Path(__file__).resolve().parent / "kg" / "import_neo4j.sh"
        print(f"Running Neo4j import script: {script}")
        result = subprocess.run([str(script), args.out])
        return result.returncode
    return 1


def _cmd_eval(args) -> int:
    """The evaluation harness (vault/ideas/eval-harness-design.md).

    `describe` and `report` need nothing; `run --system stub` needs nothing
    either, which is the point of S-22 -- the harness produces a number before
    a model or a question set exists.
    """
    # The endpoint, key and model name live in .env, and the run record's
    # `model` field is part of a result's identity -- a run that does not know
    # which model produced it is not comparable to anything.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from hepcoveragekg.eval import questions as Q
    from hepcoveragekg.eval import report as R
    from hepcoveragekg.eval import runner, systems

    if args.action == "compare":
        if not (args.path and args.other):
            print("compare needs two run files", file=sys.stderr)
            return 2
        print(R.compare(args.path, args.other))
        return 0

    if args.action == "generate":
        from hepcoveragekg.eval import generate as Gen
        from hepcoveragekg.query import templates as Tm

        conn = Tm.read_only(args.db)
        qs = Gen.generate(conn, per_predicate=args.per_predicate,
                          paraphrases=args.paraphrases)
        out = Path(args.path or "eval/questions/dev-generated.jsonl")
        Q.save(qs, out)
        import json as _json
        print(f"wrote {len(qs)} questions to {out}\n")
        print(_json.dumps(Q.load(out).summary(), indent=2))
        return 0

    if args.action == "score":
        if not (args.path and args.other):
            print("score needs a run file and its question file", file=sys.stderr)
            return 2
        qset = Q.load(args.other, allow_test=args.unlock_test, reason=args.reason)
        n = runner.rescore(args.path, qset)
        print(f"rescored {n} records in {args.path}\n")
        print(R.render_path(args.path, tag=args.tag))
        return 0

    if args.action == "report":
        if not args.path:
            runs = list(runner.iter_runs())
            if not runs:
                print("no runs found in eval/runs/", file=sys.stderr)
                return 1
            args.path = runs[-1]  # the most recent, which is almost always what is wanted
        print(R.render_path(args.path, tag=args.tag))
        return 0

    if not args.path:
        print("need a question file", file=sys.stderr)
        return 2

    try:
        qset = Q.load(args.path, allow_test=args.unlock_test, reason=args.reason)
    except Q.QuestionError as exc:
        print(f"question set rejected: {exc}", file=sys.stderr)
        return 1

    if args.action == "describe":
        import json as _json
        print(_json.dumps(qset.summary(), indent=2))
        blocked = [q.qid for q in qset if q.blocked_on_m3]
        if blocked:
            print(f"\nblocked on M3 (need final-state signatures, S-57): {', '.join(blocked)}")
        return 0

    # --- run ---------------------------------------------------------------
    if args.system == "stub":
        system = systems.StubSystem()
    elif args.system == "planner":
        from hepcoveragekg.query import retrieve, templates
        conn = templates.read_only(args.db)
        index = retrieve.build(conn, cache="data/processed/retrieval_index.npz")
        system = systems.PlannerSystem(
            conn, index,
            max_rounds=args.max_rounds, max_places=args.max_places,
            minimal_prompt=args.minimal_prompt,
        )
    else:
        print(f"unknown system {args.system!r}", file=sys.stderr)
        return 2

    done = {"n": 0}

    def tick(record) -> None:
        done["n"] += 1
        mark = "!" if record.answer.get("error") else "."
        print(mark, end="", flush=True)

    path = runner.run(qset, system, repeats=args.repeats, on_record=tick)
    print(f"\n\nwrote {path}\n")
    print(R.render_path(path, tag=args.tag))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hepcoveragekg", description="HEP coverage KG importer")
    parser.add_argument(
        "--db", default=str(store.DEFAULT_DB_PATH), help="SQLite database path"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import a bundle directory or a single file")
    p_import.add_argument("path", help="a directory of bundles, or one .json/.json.gz file")
    p_import.set_defaults(func=_cmd_import)

    p_verify = sub.add_parser("verify-counts", help="check the milestone-1 count target")
    p_verify.set_defaults(func=_cmd_verify_counts)

    p_reader = sub.add_parser(
        "reader", help="answer the supervisor's questions from the PAPERS (graph-independent)")
    p_reader.add_argument("--out", default="eval/reader/reader.jsonl",
                          help="streamed JSONL output path")
    p_reader.add_argument("--qid", default=None,
                          help="comma-separated question ids, e.g. gf-02,gf-08")
    p_reader.add_argument("--scope", choices=["all", "sweep", "single"], default="all",
                          help="sweep = the 7 corpus-wide questions; single = the 8 that name a paper")
    p_reader.add_argument("--repeats", type=int, default=3,
                          help="samples per window; agreement is the confidence signal")
    p_reader.add_argument("--temperature", type=float, default=0.3,
                          help="must be > 0 or the repeats measure nothing")
    p_reader.add_argument("--concurrency", type=int, default=None,
                          help="in-flight requests (env LLM_CONCURRENCY)")
    p_reader.add_argument("--no-cascade", action="store_true",
                          help="read whole papers instead of routing to sections first")
    p_reader.add_argument("--limit-papers", type=int, default=None,
                          help="first N papers per question — for a smoke test")
    p_reader.add_argument("--rescore", default=None, metavar="RUN.jsonl",
                          help="recompute answers from a run's stored verdicts (no LLM calls)")
    p_reader.add_argument("--check-support", default=None, metavar="RESCORED.jsonl",
                          help="re-check every YES: does the cited quote actually answer "
                               "the question? A verified quote proves the sentence is in "
                               "the paper, not that it was read correctly.")
    p_reader.add_argument("--recheck", default=None, metavar="STAGE1.jsonl",
                          help="stage 2: re-read only the papers stage 1 rejected, "
                               "with whatever model LLM_MODEL_NAME points at")
    p_reader.add_argument("--judge-url", default=None,
                          help="judge on a different endpoint from the reader")
    p_reader.add_argument("--judge-model", default=None, help="judge model id")
    p_reader.add_argument("--judge-max-tokens", type=int, default=None,
                          help="completion budget for the judge (raise for a reasoning model)")
    p_reader.add_argument("--dry-run", action="store_true",
                          help="print the plan and the read count, call nothing")
    p_reader.set_defaults(func=_cmd_reader)

    p_facets = sub.add_parser(
        "facets", help="derive the closed-vocabulary facet and signature layers")
    p_facets.add_argument(
        "action",
        choices=["derive", "signatures", "gaps", "card"],
        help="derive: entity facet tags. signatures: rebuilt cut trees. "
             "gaps: labels the vocabulary misses. card: one paper's analysis card.",
    )
    p_facets.add_argument("--kind", default="detector_object",
                          help="gaps: which entity kind to list (default detector_object)")
    p_facets.add_argument("--paper", default=None, help="card: arXiv id")
    p_facets.add_argument("--limit", type=int, default=50, help="gaps: how many labels")
    p_facets.set_defaults(func=_cmd_facets)

    p_aliases = sub.add_parser("aliases", help="build / deep / report / confirm the aliases layer")
    p_aliases.add_argument(
        "action",
        choices=["build", "deep", "clusters", "confidence", "report", "confirm"],
        help="build: Tiers 1+1.5, offline. deep: Tiers 2/2.5/3, needs a model + LLM endpoint.",
    )
    p_aliases.add_argument("--out", default="data/processed", help="report output directory")
    p_aliases.add_argument(
        "--deep-out",
        default=None,
        help="deep: JSON path for adjudicated proposals (default: data/processed/aliases_proposed.json)",
    )
    p_aliases.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="deep: in-flight LLM requests (env LLM_CONCURRENCY, default 8; raise for a dedicated server)",
    )
    p_aliases.add_argument(
        "--dry-run",
        action="store_true",
        help="deep: stop after the guards and report how many pairs WOULD be sent (no LLM calls)",
    )
    p_aliases.add_argument(
        "--stability-sample",
        type=int,
        default=0,
        help="deep: re-adjudicate N matches per kind at the end and report how often "
             "the verdict holds (0 = off). Temperature is 0.0 but vLLM is not "
             "bit-deterministic, and an unmeasured noise floor makes any later "
             "precision comparison unreadable.",
    )
    p_aliases.add_argument("--stability-repeats", type=int, default=3,
                           help="deep: total runs per sampled pair (default 3)")
    p_aliases.add_argument("--samples", type=int, default=5,
                           help="confidence: samples per pair (default 5)")
    p_aliases.add_argument("--temperature", type=float, default=0.7,
                           help="confidence: sampling temperature (default 0.7). "
                                "Too low and nothing separates; too high and even "
                                "certain pairs wander.")
    p_aliases.add_argument("--method", default=None, help="confirm: restrict to one tier/method")
    p_aliases.set_defaults(func=_cmd_aliases)

    p_graph = sub.add_parser("graph", help="Neo4j graph projection operations")
    p_graph.add_argument("action", choices=["export", "import"])
    p_graph.add_argument("out", help="output directory for CSV files (export) or input directory (import)")
    p_graph.set_defaults(func=_cmd_graph)

    p_eval = sub.add_parser("eval", help="evaluation harness: run / report / compare / describe")
    p_eval.add_argument("action",
                        choices=["generate", "run", "score", "report", "compare", "describe"])
    p_eval.add_argument("path", nargs="?", help="question file (run/describe) or run file (report/compare)")
    p_eval.add_argument("other", nargs="?",
                        help="compare: the second run file. score: the question file")
    p_eval.add_argument("--system", default="stub",
                        help="stub (default, needs nothing) or planner (needs a live model)")
    p_eval.add_argument("--repeats", type=int, default=1,
                        help="run each question N times; 3+ before comparing anything (S-52)")
    p_eval.add_argument("--max-rounds", type=int, default=6)
    p_eval.add_argument("--max-places", type=int, default=8)
    p_eval.add_argument("--minimal-prompt", action="store_true",
                        help="planner: the reduced PURPOSE variant -- an ablation axis")
    p_eval.add_argument("--unlock-test", action="store_true",
                        help="allow test-split questions; logged to eval/TEST_OPENED.log (S-10)")
    p_eval.add_argument("--reason", default="", help="why the test set was opened")
    p_eval.add_argument("--per-predicate", type=int, default=10,
                        help="generate: how many entity clusters per predicate")
    p_eval.add_argument("--paraphrases", type=int, default=3,
                        help="generate: wordings per question (metamorphic groups)")
    p_eval.add_argument("--tag", default="shape", help="report: break down by shape | difficulty | needs")
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The library reports progress through `logging`, and without a handler that
    # output is silently discarded -- `aliases deep --dry-run` printed nothing at
    # all, because its entire report is logger.info() calls.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
