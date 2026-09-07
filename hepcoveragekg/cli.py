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

    if args.action == "region-roles":
        info = facets_derive.derive_region_roles(conn)
        print(f"region roles derive ({info['vocabulary']}) — "
              f"{info['matched']}/{info['regions']} regions ({info['hit_rate']:.0%})")
        for role, n in sorted(info["by_role"].items(), key=lambda kv: -kv[1]):
            print(f"  {role:14} {n:5}")
        # Which rung supplied the role. An attribute is what the extraction
        # asserted; a label is read off the region's name, and if that rung ever
        # proves unreliable this is the number that says how much rests on it.
        src = info["by_source"]
        print(f"  {'from attribute':14} {src.get('attribute', 0):5}")
        print(f"  {'from label':14} {src.get('label', 0):5}")
        print(f"  {'untagged':14} {info['missed']:5}   "
              "(not a role: superbins, excluded regions, is_signal_region=False)")
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

    if args.conditions:
        # A multi-condition question, asked one condition at a time with the AND
        # computed over the whole paper rather than per window.
        q = next((r for r in records if r["qid"] == args.conditions), None)
        if q is None or not q.get("conditions"):
            print(f"{args.conditions} has no `conditions` defined"); return 1
        papers = q["provenance"]["paper_scope"] or all_papers
        if args.limit_papers:
            papers = papers[:args.limit_papers]
        if args.only_papers:
            wanted = set(args.only_papers.split(","))
            papers = [p for p in papers if p in wanted]
        print(f"{q['qid']}: {len(q['conditions'])} conditions x {len(papers)} papers")
        for i, cond in enumerate(q["conditions"], 1):
            print(f"   {i}. {cond}")
        if args.dry_run:
            print("\n--dry-run: nothing called"); return 0
        meta = asyncio.run(R.run_conditions(
            conn, q, papers, args.out, repeats=args.repeats,
            temperature=args.temperature, concurrency=args.concurrency))
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"-> {args.out}")
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
        judged = R.judge_records(records)
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
    from hepcoveragekg.eval import free_sql

    def _fewshot_block(a):
        if not getattr(a, "fewshot", None):
            return ""
        from hepcoveragekg.eval import questions as _q
        from hepcoveragekg.query import fewshot as _fs
        # `path` is the question file for `eval run`. Named wrong on the
        # first attempt, which crashed every few-shot arm AFTER the index
        # had been built -- three arms produced a header and no numbers.
        scored = {x.qid for x in _q.load(a.path)}
        # TRY EVERY METRIC. A run predating Gabriel's gold carries no judged_f1
        # at all, so defaulting to it found zero candidates and produced an empty
        # block -- and an empty block is a baseline run wearing an arm's name.
        # Two few-shot arms came back identical to baseline in every metric
        # before anyone noticed.
        pool = []
        for metric in ("judged_f1", "set_f1", "count_correct"):
            pool = _fs.candidates(a.fewshot, metric=metric)
            if pool:
                break
        chosen = _fs.select(pool, scored)
        block = _fs.render(chosen, with_plan=bool(getattr(a, "fewshot_plan", False)))
        if not block:
            # LOUD. Asking for few-shot and silently getting none is how three
            # arms in a row measured nothing while looking like clean nulls.
            raise SystemExit(
                f"--fewshot {a.fewshot} yielded no usable examples: no record "
                f"scored above {_fs.MIN_SCORE} on judged_f1, set_f1 or "
                f"count_correct outside the evaluation set. Refusing to run an "
                f"arm that would be identical to baseline.")
        return block


    if args.system == "stub":
        make_system = systems.StubSystem
        system = make_system()
    elif args.system == "free-sql":
        # The control. Same runner, same scorers, same model -- one tool that
        # takes SQL instead of nine typed ones. See eval/free_sql.py.
        from hepcoveragekg.query import retrieve, templates
        conn = templates.read_only(args.db)
        # The SAME index the planner uses. Withholding it would make this a
        # comparison of search technology rather than of typed structure.
        index = retrieve.build(
            conn,
            cache=retrieve.cache_path(args.index_values, args.index_quotes),
            include_values=args.index_values,
            include_quotes=args.index_quotes)
        # A FACTORY, NOT AN OBJECT. With --workers each thread needs its own
        # system and its own sqlite connection: FreeSQLSystem keeps search sets
        # as temp tables named `search_N` on the connection, and two questions
        # sharing one would silently overwrite each other's sets. The INDEX is
        # shared deliberately -- it is immutable after build() and ~44MB.
        def make_system():
            return free_sql.FreeSQLSystem(
                templates.read_only(args.db), index, max_rounds=args.max_rounds,
                persist=not args.no_persist,
                reviewer=args.reviewer,
                state_objective=args.state_objective,
                index_values=args.index_values,
                index_quotes=args.index_quotes,
                search_sets=args.search_sets,
                concept_prompt=args.concept_prompt,
                subgoals=args.subgoals,
                subgoal_status=args.subgoal_status)
        system = make_system()
    elif args.system == "planner":
        from hepcoveragekg.query import retrieve, templates
        conn = templates.read_only(args.db)
        index = retrieve.build(
            conn,
            cache=retrieve.cache_path(args.index_values, args.index_quotes),
            include_values=args.index_values,
            include_quotes=args.index_quotes)
        def make_system():
            return systems.PlannerSystem(
                templates.read_only(args.db), index, index_values=args.index_values,
                index_quotes=args.index_quotes,
                max_rounds=args.max_rounds, max_places=args.max_places,
                minimal_prompt=args.minimal_prompt,
                use_critic=args.critic,
                critic_seed=args.critic_seed,
                answer_contract=args.answer_contract,
                contract=args.contract,
                force_critic_set=args.force_critic_set,
                persist=args.persist,
                push_further=args.push_further,
                simple_answer=args.simple_answer,
                fewshot=_fewshot_block(args),
                tool_examples=args.tool_examples,
                reviewer=args.reviewer,
                state_objective=args.state_objective,
                subgoals=args.subgoals,
                subgoal_status=args.subgoal_status,
                path_tool=args.path_tool,
                answer_gate=args.answer_gate,
                answer_critic=args.answer_critic,
                rerank=args.rerank,
                name_ids=args.name_ids,
                kind_fallback=args.kind_fallback,
            )
        system = make_system()
    else:
        make_system = None
        print(f"unknown system {args.system!r}", file=sys.stderr)
        return 2

    done = {"n": 0}

    def tick(record) -> None:
        done["n"] += 1
        mark = "!" if record.answer.get("error") else "."
        print(mark, end="", flush=True)

    # A TIMED-OUT RECORD SCORES ZERO, so the default penalises slowness rather
    # than wrongness. Measured 2026-09-01: typed+reviewer lost 5 of 24 records
    # and qwen3.8-27b lost 7, both at exactly 600s -- arms that add a per-round
    # LLM call need a bigger budget or they are marked down for the cost of the
    # mechanism being tested.
    path = runner.run(qset, system, repeats=args.repeats, on_record=tick,
                      max_workers=args.workers, make_system=make_system,
                      timeout=(args.timeout or runner.DEFAULT_TIMEOUT))
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
    p_reader.add_argument("--conditions", default=None, metavar="QID",
                          help="ask a multi-condition question one condition at a "
                               "time, combining them over the whole paper")
    p_reader.add_argument("--only-papers", default=None,
                          help="comma-separated arXiv ids, for a targeted diagnostic")
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
        choices=["derive", "region-roles", "signatures", "gaps", "card"],
        help="derive: entity facet tags. region-roles: signal/control/validation"
             " /fiducial/preselection per region. signatures: rebuilt cut trees."
             " gaps: labels the vocabulary misses. card: one paper's analysis card.",
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
                        help="stub (default, needs nothing), planner, or free-sql "
                             "-- the control: one SQL tool instead of the typed ones")
    p_eval.add_argument("--workers", type=int, default=1,
                        help="answer this many questions at once. The bottleneck "
                             "is a ~90s network call per question, so this is close "
                             "to a linear speed-up; 12 turns a 22-hour arm into two. "
                             "Each worker gets its OWN system and sqlite connection, "
                             "because free-SQL keeps search sets in temp tables named "
                             "on the connection and sharing one corrupts answers silently.")
    p_eval.add_argument("--repeats", type=int, default=1,
                        help="run each question N times; 3+ before comparing anything (S-52)")
    p_eval.add_argument("--max-rounds", type=int, default=6)
    p_eval.add_argument("--max-places", type=int, default=8)
    p_eval.add_argument("--minimal-prompt", action="store_true",
                        help="planner: the reduced PURPOSE variant -- an ablation axis")
    # The critic is an ablation ARM, so it belongs in the config hash: two runs
    # that differ only by this must not be mistaken for repeats of one another.
    # It is off by default, which makes the control arm the ordinary code path
    # rather than a second implementation (D-060).
    p_eval.add_argument("--critic", action="store_true",
                        help="planner: judge search candidates and read facet "
                             "labels before use. Flags, never filters.")
    # Ranked order is the default and is now IN DOUBT: on 2026-08-15 the shuffled
    # arms discriminated by true retrieval rank BETTER than ranked did, which is
    # the opposite of what the default was chosen for. This makes that a third
    # arm rather than an argument.
    # The v2 answer contract: cite a set instead of retyping ids, `refine` to
    # mark rows out, and a challenged abstention. An ARM, not a default -- it
    # adds a tool and changes `answer`'s schema, so a run with it differs from
    # one without by more than the thing under test.
    p_eval.add_argument("--answer-contract", action="store_true",
                        help="planner: cite sets in the answer, allow `refine`, "
                             "and challenge an abstention held against evidence")
    p_eval.add_argument("--contract", default="", choices=["", "v1", "v2", "v3"],
                        help="planner: answer contract. v3 keeps only citing the "
                             "paper set and the challenged abstention")
    # The critic judges every candidate and the planner then uses its verdict
    # for only 58% of counts. This makes "is the critic better than the
    # planner's discretion?" measurable.
    p_eval.add_argument("--force-critic-set", action="store_true",
                        help="planner: substitute the critic's kept set wherever "
                             "a raw search set is passed to a tool")
    p_eval.add_argument("--timeout", type=float, default=None,
                        help="seconds per record before it is abandoned and "
                             "scored zero (default 600). Raise it for arms that "
                             "add an LLM call per round")
    p_eval.add_argument("--path-tool", action="store_true",
                        help="planner: a `path` tool that applies SEVERAL "
                             "predicate->object constraints at once. gf-01 asks "
                             "for b-tagged jets AND missing transverse momentum "
                             "and scores 0.48; the same concept with one "
                             "condition scores 0.89")
    p_eval.add_argument("--subgoals", action="store_true",
                        help="decompose the question into at most 3 ordered "
                             "sub-objectives once, before the first round; the "
                             "list is fixed and never updated")
    p_eval.add_argument("--subgoal-status", action="store_true",
                        help="--subgoals plus a status per objective, rewritten "
                             "every round and carried in the prompt (PoG's "
                             "highest-value mechanism; targets gf-01's dropped "
                             "conditions). Implies --subgoals")
    p_eval.add_argument("--concept-prompt", action="store_true",
                        help="free-sql: tell it a search result is a SAMPLE OF "
                             "THE VOCABULARY, not the answer set -- match the "
                             "family with a pattern instead of pasting the ids "
                             "it happened to be shown (implied by --search-sets)")
    p_eval.add_argument("--search-sets", action="store_true",
                        help="free-sql: materialise the FULL match set of each "
                             "search as a temp table the model filters in SQL, "
                             "instead of retyping a few ids from the 20 shown")
    p_eval.add_argument("--index-quotes", action="store_true",
                        help="index the verbatim evidence quotes (normalised) as "
                             "searchable surface forms: 97%% of assertions carry "
                             "one and none of it was reachable -- 2102.01444 has "
                             "17 quotes naming a ttZ control region and no entity "
                             "labelled ttZ")
    p_eval.add_argument("--index-values", action="store_true",
                        help="index assertion object_value as searchable surface "
                             "forms: 31%% of assertions hold their object as free "
                             "text (selections, quantities, region definitions) "
                             "that `search` could not reach at all")
    p_eval.add_argument("--reviewer", action="store_true",
                        help="planner: a second strong model checks each round's "
                             "objective and proposed calls BEFORE they run; a "
                             "rejection is re-planned without spending a round")
    p_eval.add_argument("--state-objective", action="store_true",
                        help="planner: each turn must open with GOT (did the last "
                             "result meet the objective) and AIM (what this round "
                             "is for) before its tool calls")
    p_eval.add_argument("--tool-examples", action="store_true",
                        help="planner: attach one real worked call per tool, "
                             "mined from runs that answered correctly")
    p_eval.add_argument("--kind-fallback", action="store_true",
                        help="planner: when a search names a `kind`, also "
                             "search without it and append what other kinds "
                             "match. Offline on Gabriel's questions: 61 -> 74 "
                             "of 79 gold papers reached; gf-05 2 -> 13 (D-119)")
    p_eval.add_argument("--name-ids", action="store_true",
                        help="planner: ask the answer to WRITE the arXiv ids "
                             "into `text`. v3 currently says the opposite -- "
                             "cite a set instead of writing ids -- and `text` "
                             "is the only field the set scorers read (D-117)")
    p_eval.add_argument("--rerank", action="store_true",
                        help="planner: order candidates BEST FIRST instead of "
                             "dropping them. Alone it costs nothing -- it uses "
                             "the exact/broader rung the critic already "
                             "computes. With --answer-critic it also grades and "
                             "reorders each `papers_of` result. Never removes a "
                             "paper: the answerer's own truncation stays where "
                             "it is and the ranking decides what survives it "
                             "(D-113)")
    p_eval.add_argument("--answer-gate", action="store_true",
                        help="planner: reject an answer that names no arXiv "
                             "ids -- a placeholder, a set reference, a list of "
                             "titles, or an instruction to run another tool -- "
                             "and ask for it once. No model call (D-107)")
    p_eval.add_argument("--answer-critic", action="store_true",
                        help="planner: a small judge decides, per cited paper, "
                             "whether it satisfies the question, and drops the "
                             "ones that do not. Can only narrow the answer "
                             "(D-106)")
    p_eval.add_argument("--simple-answer", action="store_true",
                        help="planner: answer with a literal list of arXiv ids "
                             "instead of naming a set in `papers_from`")
    p_eval.add_argument("--fewshot", default=None,
                        help="planner: a run file to harvest worked examples "
                             "from. Refuses any drawn from the questions being "
                             "scored")
    p_eval.add_argument("--fewshot-plan", action="store_true",
                        help="planner: exemplars show the PLAN as well as the "
                             "answer (default: answer only)")
    p_eval.add_argument("--push-further", action="store_true",
                        help="planner: when it ANSWERS after one search with "
                             "rounds to spare, offer one untried route first. "
                             "Separate from --persist: the risk runs the other "
                             "way, so it is measured on its own")
    p_eval.add_argument("--no-persist", action="store_true",
                        help="free-sql: turn OFF its widening ladder (on by "
                             "default, to match the planner's persist arm)")
    p_eval.add_argument("--persist", action="store_true",
                        help="planner: before abstaining while holding rows, "
                             "offer one concrete untried route. Finite ladder "
                             "(4 rungs, each once); see query/widen.py")
    p_eval.add_argument("--critic-seed", type=int, default=None,
                        help="planner: shuffle the candidates the critic sees, "
                             "with this seed. Omit for retrieval order.")
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
