"""What an OpenRouter arm will cost, before any of it is spent."""
import json, sys, glob, statistics
from hepcoveragekg.query import budget

qfile = sys.argv[1] if len(sys.argv) > 1 else "eval/questions/gabriel-gold-2026-08-25.jsonl"
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
n = sum(1 for l in open(qfile, encoding="utf-8") if l.strip()) * repeats

# measured on the real planner, this corpus, contract v3 + critic
PT, CT = 18899, 419
print(f"{qfile.split('/')[-1]}  x{repeats} = {n} records")
print(f"at {PT:,} prompt + {CT} completion tokens each "
      f"-> {n*PT:,} in, {n*CT:,} out\n")
print(f"  {'model':36s} {'cost':>8}  {'budget to set':>14}")
for m, (i, o) in sorted(budget.PRICES.items(), key=lambda x: x[1][0]):
    c = n*PT/1e6*i + n*CT/1e6*o
    print(f"  {m:36s} {'$'+format(c,'.2f'):>8}  {'$'+format(c*1.3,'.2f'):>14}")
print("\n  budget suggestion is cost x1.3 -- retries and a longer tail on a")
print("  model that reasons more than Qwen does.")
