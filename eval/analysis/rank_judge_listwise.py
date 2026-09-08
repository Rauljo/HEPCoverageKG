"""LISTWISE rerank: each chunk returns an ORDER, grades derived by position.
Same output shape as rank_judge.py so the same scorer applies (D-129)."""
import json, os, pickle, sys, re, collections
from dotenv import load_dotenv
from openai import OpenAI
from hepcoveragekg.query import answer_critic as AC
from hepcoveragekg.query import budget as B
load_dotenv(os.path.join(os.getcwd(), ".env"))
MODEL, OUT, PAIRS = sys.argv[1], sys.argv[2], sys.argv[3]
CAP = float(os.environ.get("JUDGE_CAP_USD", "0.20"))
PROMPT = """\
You ORDER papers by how well each satisfies a question. For each paper you are \
given the material retrieved from it: entity labels, and verbatim sentences.

Judge the CONDITION, not the topic. A paper about b-tagging is not automatically \
a paper whose event selection uses b-tagged jets. A paper that VETOES b-jets does \
use them -- a veto is a selection on the object. A paper that merely mentions a \
technique in passing does not use it.

Return ALL the paper ids you were given, best first, worst last. Every id \
exactly once. Reply with JSON and nothing else:
{"order": ["<best arxiv id>", "<next>", ...]}"""
pairs = pickle.load(open(PAIRS, "rb"))
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120, max_retries=3)
pin, pout = B.PRICES[MODEL]; spent = {"in": 0, "out": 0}; SPAN = re.compile(r"\{.*\}", re.DOTALL)
def chat(messages):
    usd = spent["in"]/1e6*pin + spent["out"]/1e6*pout
    if usd > CAP: raise B.BudgetExceeded(f"${usd:.4f} > cap ${CAP}")
    kw = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=600)
    if MODEL in B.REASONING_MODELS or "qwen3" in MODEL:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(**kw); u = getattr(r, "usage", None)
    if u: spent["in"] += u.prompt_tokens or 0; spent["out"] += u.completion_tokens or 0
    return r
by_q = collections.defaultdict(list)
for p in pairs: by_q[(p.qid, p.question)].append(p)
rows = []; unordered = 0
for (qid, question), group in sorted(by_q.items()):
    grade = {}
    for i in range(0, len(group), 8):
        batch = group[i:i+8]; ids = [p.paper for p in batch]
        blocks = "\n\n".join(AC._render(p.paper, p.labels, p.quotes) for p in batch)
        try:
            raw = chat([{"role": "system", "content": PROMPT},
                        {"role": "user", "content": f"QUESTION: {question}\n\n{blocks}"}]).choices[0].message.content or ""
        except B.BudgetExceeded: raise
        except Exception as e: print("call failed:", e, file=sys.stderr); raw = ""
        m = SPAN.search(raw); order = []
        if m:
            try: order = [str(x) for x in (json.loads(m.group(0)).get("order") or []) if str(x) in ids]
            except json.JSONDecodeError: pass
        order += [x for x in ids if x not in order]      # anything omitted goes last, in input order
        n = len(order)
        for pos, pid in enumerate(order):                 # position -> grade: quartiles of the chunk
            grade[pid] = 3 if pos < n/4 else 2 if pos < n/2 else 1 if pos < 3*n/4 else 0
    for p in group:
        if p.paper not in grade: unordered += 1
        rows.append({"qid": qid, "paper": p.paper, "gold": p.gold, "grade": grade.get(p.paper, -1.0)})
usd = spent["in"]/1e6*pin + spent["out"]/1e6*pout
json.dump({"model": MODEL, "usd": round(usd, 4), "ungraded": unordered, "rows": rows}, open(OUT, "w"), indent=1)
print(f"{MODEL}  ${usd:.4f}  unordered={unordered}/{len(rows)}")
