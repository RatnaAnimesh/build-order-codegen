"""Exploratory re-scoring of Stage A's saved attempts. Nothing here was fixed before the run.

1. Plain under other budget rules, beside the paper's instruct and GRPO rows: does a rule explain the failed check?
2. The three arms with GPU-seconds per attempt as the cost instead of generated tokens.

    python3 -I rescore.py ~/slm-runs/qwen7b/full_run/s3_state_<ts>/stage_a/state.json
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import PAPER, attempts_at, boot, pass_at_k

s = json.load(open(sys.argv[1]))
att = {arm: {q: [a for a in v if a.get("passed") is not None] for q, v in s["attempts"][arm].items()}
       for arm in ("plain", "decomp", "idea")}
ids = [q for q in att["plain"] if all(att[arm].get(q) for arm in att)]
rng = random.Random(0)


def per(arm, b, cost, qs=ids, solved=lambda q, a: a["passed"]):
    """{problem: (unbiased pass@k, has k graded attempts)} for the k that budget b buys at this cost per attempt."""
    k = attempts_at(b, cost)
    return {q: (pass_at_k(len(att[arm][q]), sum(bool(solved(q, a)) for a in att[arm][q]), k), len(att[arm][q]) >= k)
            for q in qs}


def cell(p):
    v, cov = (sum(x[i] for x in p.values()) / len(p) for i in (0, 1))
    return f"{v:.3f}{'' if cov >= 0.95 else '*'}"


# 1. plain under other rules; '*' = fewer than 95% of problems have the k attempts (pass@n, biased low)
B = PAPER["table1_lcb_qwen7b"]["budgets"][:4]
n_plain = sum(len(att["plain"][q]) for q in ids)
gen = sum(a["tokens"] for q in ids for a in att["plain"][q]) / n_plain
prompt = {q: sum(a["prompt_tokens"] for a in s["attempts"]["plain"][q]) for q in ids}  # ~one full prompt: cache hits
with_prompt = sum(a["tokens"] + prompt[q] for q in ids for a in att["plain"][q]) / n_plain
leet, atcoder = [q for q in ids if q.isdigit()], [q for q in ids if not q.isdigit()]
rules = {
    f"ours: {gen:.0f} generated tokens per attempt": [per("plain", b, gen) for b in B],
    "paper's Table 3 cost, 387 per attempt": [per("plain", b, 387) for b in B],
    f"prompt counted every attempt, {with_prompt:.0f} per attempt": [per("plain", b, with_prompt) for b in B],
    "LeetCode (function) problems all failed": [per("plain", b, gen, solved=lambda q, a: a["passed"] and not q.isdigit())
                                                for b in B],
    f"LeetCode problems only ({len(leet)})": [per("plain", b, gen, leet) for b in B],
    f"AtCoder problems only ({len(atcoder)})": [per("plain", b, gen, atcoder) for b in B],
}
print("| plain, rule | " + " | ".join(f"{b // 1000}k" for b in B) + " |\n|---|" + "---|" * len(B))
for name, ps in rules.items():
    print(f"| {name} | " + " | ".join(cell(p) for p in ps) + " |")
for name in ("instruct", "GRPO"):
    print(f"| paper {name} | " + " | ".join(f"{v:.2f}" for v in PAPER["table1_lcb_qwen7b"][name][:4]) + " |")

# 2. GPU-seconds per attempt (generation wall time per arm, prefill included) as the cost
sec = {arm: s["timing"][arm][0] / sum(len(v) for v in s["attempts"][arm].values()) for arm in att}
S = [5, 10, 15, 20, 40, 60, 80]
P = {arm: [per(arm, b, sec[arm]) for b in S] for arm in att}
print("\n| GPU-s per problem | " + " | ".join(f"{arm} ({sec[arm]:.1f} s/attempt)" for arm in att) + " |\n|---|---|---|---|")
for i, b in enumerate(S):
    print(f"| {b} | " + " | ".join(f"{cell(P[arm][i])} (k={attempts_at(b, sec[arm])})" for arm in att) + " |")


def largest(arms):
    ok = [i for i in range(len(S)) if all(sum(c for _, c in P[a][i].values()) / len(ids) >= 0.95 for a in arms)]
    return ok[-1] if ok else None


for x, y in (("idea", "decomp"), ("idea", "plain"), ("decomp", "plain")):
    i = largest((x, y))
    if i is not None:
        d = boot([P[x][i][q][0] - P[y][i][q][0] for q in ids], rng)
        print(f"{x} - {y} at {S[i]} GPU-s: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
