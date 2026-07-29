"""Run the aliases trial set through the configured adjudicator and score it."""
import argparse, logging, os, sys
from pathlib import Path
from hepcoveragekg.aliases import store, trialset, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/processed/hepkg.db")
    ap.add_argument("--pairs", default="data/processed/trialset.jsonl")
    ap.add_argument("--out", default="data/processed/trial_results.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--no-context", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    conn = store.connect(a.db)
    if not Path(a.pairs).exists():
        trialset.write(trialset.build(conn), a.pairs)
    pairs = list(trialset.load(a.pairs))
    if a.limit:
        pairs = pairs[: a.limit]

    print(f"model     : {os.environ.get('LLM_MODEL_NAME')}")
    print(f"endpoint  : {os.environ.get('LLM_BASE_URL')}")
    print(f"pairs     : {len(pairs)}   context={'off' if a.no_context else 'on'}   "
          f"concurrency={a.concurrency}", flush=True)

    sc = evaluate.run(conn, pairs, a.out, concurrency=a.concurrency,
                      use_context=not a.no_context)
    print()
    print(evaluate.report(sc))
    print(f"\nraw results -> {a.out}")

if __name__ == "__main__":
    sys.exit(main())
