"""Graded rerank instead of keep/drop, scored as precision@k against Gabriel."""
import json, os, pickle, sys, re, statistics, collections, logging
from dotenv import load_dotenv
from openai import OpenAI
from hepcoveragekg.query import answer_critic as AC
from hepcoveragekg.query import budget as B

logging.basicConfig(level=logging.ERROR)
load_dotenv(os.path.join(os.getcwd(), ".env"))
MODEL, OUT, PAIRS = sys.argv[1], sys.argv[2], sys.argv[3]
CAP = float(os.environ.get("JUDGE_CAP_USD", "0.20"))

# A GRADE, NOT A VERDICT. The binary critic must pick one global threshold and
# D-112 showed it cannot: it dropped 92% of a set that was 69% correct. A grade
# defers the threshold to whoever consumes the list -- and the consumer here is
# an answerer that truncates at ~16 papers anyway, so the only question that
# matters is whether the right ones are at the top.
PROMPT = """\
You rank PAPERS by how well each one satisfies a question. For each paper you \
are given the material retrieved from it: entity labels, and verbatim sentences.

Grade each paper 0-3:
  3  the retrieved text SHOWS the paper satisfies the condition asked about
  2  the paper very likely satisfies it, but the text is suggestive not explicit
  1  related to the topic, but does not satisfy the condition asked
  0  does not bear on the question at all

Judge the CONDITION, not the topic. A paper about b-tagging is not automatically \
a paper whose event selection uses b-tagged jets. A paper that VETOES b-jets does \
use them -- a veto is a selection on the object. A paper that merely mentions a \
technique in passing does not use it.

Use the whole scale. If everything looks the same grade, you are not \
discriminating and the ranking is useless.

Reply with JSON and nothing else:
{"grades": [{"paper": "<arxiv id>", "grade": 0|1|2|3, "why": "<8 words>"}]}"""

pairs = pickle.load(open(PAIRS, "rb"))
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"], timeout=120, max_retries=3)
pin, pout = B.PRICES[MODEL]
spent = {"in": 0, "out": 0}
SPAN = re.compile(r"\{.*\}", re.DOTALL)

def chat(messages):
    usd = spent["in"]/1e6*pin + spent["out"]/1e6*pout
    if usd > CAP:
        raise B.BudgetExceeded(f"${usd:.4f} > cap ${CAP}")
    kw = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=1500)
    if MODEL in B.REASONING_MODELS or "qwen3" in MODEL:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(**kw)
    u = getattr(r, "usage", None)
    if u:
        spent["in"] += u.prompt_tokens or 0
        spent["out"] += u.completion_tokens or 0
    return r

by_q = collections.defaultdict(list)
for p in pairs:
    by_q[(p.qid, p.question)].append(p)

rows, ungraded = [], 0
for (qid, question), group in sorted(by_q.items()):
    grades = {}
    for i in range(0, len(group), 8):
        batch = group[i:i+8]
        blocks = "\n\n".join(AC._render(p.paper, p.labels, p.quotes) for p in batch)
        try:
            r = chat([{"role": "system", "content": PROMPT},
                      {"role": "user", "content": f"QUESTION: {question}\n\n{blocks}"}])
            raw = r.choices[0].message.content or ""
        except B.BudgetExceeded:
            raise
        except Exception as exc:
            print("call failed:", exc, file=sys.stderr); raw = ""
        m = SPAN.search(raw)
        if not m:
            continue
        try:
            got = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for item in (got.get("grades") or []):
            pid = str(item.get("paper") or "").strip()
            g = item.get("grade")
            if pid and isinstance(g, (int, float)):
                grades[pid] = float(g)
    for p in group:
        if p.paper not in grades:
            ungraded += 1
        rows.append({"qid": qid, "paper": p.paper, "gold": p.gold,
                     "grade": grades.get(p.paper, -1.0)})

usd = spent["in"]/1e6*pin + spent["out"]/1e6*pout
json.dump({"model": MODEL, "usd": round(usd, 4), "ungraded": ungraded, "rows": rows},
          open(OUT, "w"), indent=1)
print(f"{MODEL}  ${usd:.4f}  ungraded={ungraded}/{len(rows)}")
