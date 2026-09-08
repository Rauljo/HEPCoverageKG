"""Judge Gabriel's 253 pairs with one model, under a hard spend cap."""
import json, os, pickle, sys, logging
from dotenv import load_dotenv
from openai import OpenAI
from hepcoveragekg.eval import judge_gold as JG
from hepcoveragekg.query import budget as B

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
load_dotenv(os.path.join(os.getcwd(), ".env"))
MODEL = sys.argv[1]
OUT = sys.argv[2]
CAP = float(os.environ.get("JUDGE_CAP_USD", "0.25"))

pairs = pickle.load(open(sys.argv[3], "rb"))
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout=120, max_retries=3)

spent = {"in": 0, "out": 0}
pin, pout = B.PRICES[MODEL]

def chat(messages):
    usd = spent["in"] / 1e6 * pin + spent["out"] / 1e6 * pout
    if usd > CAP:
        raise B.BudgetExceeded(f"${usd:.4f} > cap ${CAP}")
    kw = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=1200)
    if MODEL in B.REASONING_MODELS or "qwen3" in MODEL:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(**kw)
    u = getattr(r, "usage", None)
    if u:
        spent["in"] += u.prompt_tokens or 0
        spent["out"] += u.completion_tokens or 0
    return r

out, rows = JG.run(pairs, chat)
usd = spent["in"] / 1e6 * pin + spent["out"] / 1e6 * pout
res = {"model": MODEL, "billed_prompt": spent["in"], "billed_completion": spent["out"],
       "usd": round(usd, 4), **out.to_dict()}
json.dump({"summary": res, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
