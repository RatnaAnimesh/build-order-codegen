"""Stage A report: pass@tokens per arm, paired differences, and the paper's numbers beside ours.

pass@tokens(B), the paper's rule (Appendix K, primary): a budget B buys k = max(1, floor(B / c)) attempts, c being the
arm's mean generated tokens per attempt over all graded attempts; a problem's value is the unbiased pass@k from its n
graded attempts (Chen et al. 2021). A budget is covered by an arm when COVER of problems have n >= k. The paper
doesn't state its rounding; floor with at least one whole attempt fits Table 1, whose DecompRL row (4,000 tokens per
attempt) is flat at 0.18 from 1k to 5k.
Exact-token check: walk the attempts in a random order, add each one's tokens, and count the problem solved if an
attempt that passed ends at or before B; averaged over PERMS orders. Covered when COVER of problems used >= B tokens.
Only graded attempts count, and only problems graded in all three arms.

Fixed before the full run (README "Decisions"):
  * harness check: plain within 0.05 of the paper's instruct row (Table 1) at 5k and 10k tokens;
  * primary: idea - decomp at the largest budget both decomposition arms cover, paired bootstrap over problems,
    95% CI; the idea's inference beats DecompRL's if the CI is above 0;
  * secondary: idea - plain and decomp - plain at the largest budget all three arms cover (plain is sampled up to
    10k only); decomp - plain at 4k against Table 4's -3 points.
    python report.py --state out/state.json --lcb lcb/ --out out/
"""
import argparse, json, math, random
from collections import Counter
from pathlib import Path

PERMS, BOOT, COVER = 400, 4000, 0.95
BUDGETS = [1000, 2000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 100000, 200000, 500000]
PAPER = {
    "table1_lcb_qwen7b": {"budgets": [1000, 5000, 10000, 50000, 100000, 500000],
                          "instruct": [0.06, 0.16, 0.21, 0.30, 0.33, 0.39],
                          "GRPO": [0.18, 0.29, 0.32, 0.38, 0.40, 0.44],
                          "pass@8": [0.15, 0.27, 0.31, 0.39, 0.41, 0.46],
                          "DecompRL": [0.18, 0.18, 0.25, 0.40, 0.44, 0.48]},
    "table4_lcb_qwen7b_at_4k": {"standard": 0.140, "hierarchical_no_rl": 0.110, "DecompRL": 0.180},
    "table3_tokens_per_attempt_qwen7b": {"no_training": 387, "GRPO": 545, "DecompRL": 4000},
}
ARMS = ("plain", "decomp", "idea")


def attempts_at(budget, cost):
    return max(1, int(budget // max(cost, 1e-9)))


def pass_at_k(n, c, k):
    """The chance that k of n attempts drawn without replacement include one of the c that passed; pass@n if k > n."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def curve(attempts, budgets, rng, perms=PERMS):
    att = [(a["tokens"], bool(a["passed"])) for a in attempts if a.get("passed") is not None]
    if not any(ok for _, ok in att):
        return [0.0] * len(budgets)
    vals, idx = [], list(range(len(att)))
    for _ in range(perms):
        rng.shuffle(idx)
        c = 0
        for i in idx:
            c += att[i][0]
            if att[i][1]:
                vals.append(c)
                break
    return [sum(v <= b for v in vals) / perms for b in budgets]


def boot(diffs, rng, n=BOOT):
    m = len(diffs)
    if m == 0:
        return None
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"mean": sum(diffs) / m, "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n) - 1]}


def build(state, problems, seed=0, budgets=BUDGETS):
    rng = random.Random(seed)
    att = state["attempts"]
    arms = tuple(a for a in ARMS if att.get(a))          # a Stage B dev run samples one arm
    has = lambda *xs: all(x in arms for x in xs)
    graded = lambda arm, q: [a for a in att[arm].get(q, []) if a.get("passed") is not None]
    ids = [p["id"] for p in problems if all(graded(arm, p["id"]) for arm in arms)]
    m = max(len(ids), 1)
    diff = {p["id"]: p["difficulty"] for p in problems}
    g = {arm: {q: graded(arm, q) for q in ids} for arm in arms}
    n = {arm: {q: len(g[arm][q]) for q in ids} for arm in arms}
    cost = {arm: sum(a["tokens"] for q in ids for a in g[arm][q]) / max(sum(n[arm].values()), 1) for arm in arms}
    k = {arm: [attempts_at(b, cost[arm]) for b in budgets] for arm in arms}
    per = {arm: {q: [pass_at_k(n[arm][q], sum(bool(a["passed"]) for a in g[arm][q]), kk) for kk in k[arm]]
                 for q in ids} for arm in arms}
    cover = {arm: [sum(n[arm][q] >= kk for q in ids) / m for kk in k[arm]] for arm in arms}
    tot = {arm: {q: sum(a["tokens"] for a in g[arm][q]) for q in ids} for arm in arms}
    per_x = {arm: {q: curve(g[arm][q], budgets, rng) for q in ids} for arm in arms}
    cover_x = {arm: [sum(tot[arm][q] >= b for q in ids) / m for b in budgets] for arm in arms}
    mean = lambda pr: {arm: [sum(pr[arm][q][i] for q in ids) / m for i in range(len(budgets))] for arm in arms}

    def largest(cov, arms):
        ok = [b for i, b in enumerate(budgets) if all(cov[a][i] >= COVER for a in arms)]
        return ok[-1] if ok else None

    def paired(x, y, b, pr=per):
        i = budgets.index(b)
        return boot([pr[x][q][i] - pr[y][q][i] for q in ids], rng)

    def by_difficulty(b, arms):
        out = {}
        for lvl in ("easy", "medium", "hard"):
            qs = [q for q in ids if diff[q] == lvl]
            if qs and b:
                i = budgets.index(b)
                out[lvl] = {"n": len(qs), **{arm: sum(per[arm][q][i] for q in qs) / len(qs) for arm in arms}}
        return {"budget": b, **out}

    top, top3 = largest(cover, ("decomp", "idea")) if has("decomp", "idea") else None, largest(cover, arms)
    r = {"problems_paired": len(ids), "problems_total": len(problems), "budgets": budgets, "rule": "appendix_k",
         "tokens_per_attempt": cost, "attempts_at": k, "pass_at_tokens": mean(per), "coverage": cover,
         "largest_covered_budget": top, "largest_covered_all_arms": top3, "paper": PAPER}
    if top:
        r["primary_idea_minus_decomp"] = {"budget": top, **paired("idea", "decomp", top)}
    if top3:
        r["secondary"] = {f"{x}_minus_{y}": {"budget": top3, **paired(x, y, top3)}
                          for x, y in (("idea", "plain"), ("decomp", "plain")) if has(x, y)}
    for b in (4000, 5000, 10000):
        if b in budgets and top3 and b <= top3:
            r.setdefault("at", {})[str(b)] = {f"{x}_minus_{y}": paired(x, y, b)
                                              for x, y in (("idea", "decomp"), ("idea", "plain"), ("decomp", "plain"))
                                              if has(x, y)}
    topx = largest(cover_x, ("decomp", "idea")) if has("decomp", "idea") else None
    r["check_exact_tokens"] = {"pass_at_tokens": mean(per_x), "coverage": cover_x, "largest_covered_budget": topx}
    if topx:
        r["check_exact_tokens"]["idea_minus_decomp"] = {"budget": topx, **paired("idea", "decomp", topx, per_x)}
    r["harness_check"] = {str(b): {"plain": r["pass_at_tokens"]["plain"][budgets.index(b)], "paper_instruct": v,
                                   "covered": cover["plain"][budgets.index(b)] >= COVER,
                                   "within_0.05": abs(r["pass_at_tokens"]["plain"][budgets.index(b)] - v) <= 0.05}
                          for b, v in ((1000, 0.06), (5000, 0.16), (10000, 0.21)) if b in budgets and has("plain")}
    r["by_difficulty"] = by_difficulty(top, ("decomp", "idea"))
    r["by_difficulty_all_arms"] = by_difficulty(top3, arms)
    r["arms"] = {arm: arm_stats(att[arm], ids) for arm in arms}
    r["llm"], r["grader"], r["sessions"] = state.get("llm"), state.get("grader"), len(state.get("sessions", [])) or 1
    return r


def arm_stats(att_arm, ids):
    a = [x for q in ids for x in att_arm.get(q, [])]
    g = [x for x in a if x.get("passed") is not None]
    s = {"attempts": len(a), "graded": len(g), "attempts_per_problem": len(a) / max(len(ids), 1),
         "tokens_per_attempt": sum(x["tokens"] for x in a) / max(len(a), 1),
         "prompt_tokens_per_attempt": sum(x.get("prompt_tokens", 0) for x in a) / max(len(a), 1),
         "prompt_tokens_per_problem": sum(x.get("prompt_tokens", 0) for x in a) / max(len(ids), 1),
         "gen_tokens_per_problem": sum(x["tokens"] for x in a) / max(len(ids), 1),
         "programs_per_attempt": sum(x.get("n_programs", 0) for x in a) / max(len(a), 1),
         "attempt_pass_rate": sum(bool(x["passed"]) for x in g) / max(len(g), 1),
         "statuses": dict(sum((Counter(x.get("statuses", {})) for x in g), Counter()))}
    if a and "units" in a[0]:
        s["units_mean"] = sum(x["units"] for x in a) / len(a)
        s["reasons"] = dict(Counter(x["reason"] for x in a if x["reason"]))
        s["parsed_impls_per_attempt"] = sum(x.get("parsed_impls", 0) for x in a) / len(a)
        s["uses_line_share"] = sum(x.get("uses_lines", 0) for x in a) / max(sum(x["units"] for x in a), 1)
        s["levels_mean"] = sum(len(x.get("levels", [])) for x in a) / len(a)
        for k in ("missing_explicit", "missing_checker", "helpers_built", "resumed", "coherent"):
            if k in a[0]:
                s[k] = sum(x.get(k, 0) for x in a)
        passed = [x for x in g if x["passed"]]
        s["first_pass_index_median"] = sorted(x["first_pass"] for x in passed)[len(passed) // 2] if passed else None
        if "coherent" in a[0]:
            s["passed_by_coherent_program"] = sum(x["first_pass"] < x.get("coherent", 0) for x in passed)
            s["passed_attempts"] = len(passed)
    return s


def markdown(r):
    t1, arms = PAPER["table1_lcb_qwen7b"], list(r["arms"])
    star = lambda cov, i, a: "" if cov[a][i] >= COVER else "*"
    L = ["# Stage A: Qwen2.5-7B-Instruct on LiveCodeBench 2024-08-01..2025-02-01",
         f"\nPaired problems: {r['problems_paired']} of {r['problems_total']}, {r['sessions']} session(s). Tokens per "
         "attempt: " + ", ".join(f"{a} {r['tokens_per_attempt'][a]:.0f}" for a in arms) + ". Largest budget covered "
         f"({int(COVER * 100)}% of problems) by decomp and idea: {r['largest_covered_budget']}; by all three: "
         f"{r['largest_covered_all_arms']}.\n",
         "Budget rule: the paper's (Appendix K), k = max(1, floor(B / tokens per attempt)) attempts, unbiased pass@k.\n",
         "| tokens | " + " | ".join(arms) + " | k " + "/".join(arms) + " | paper instruct | paper DecompRL (RL) |",
         "|---" * (len(arms) + 4) + "|"]
    for i, b in enumerate(r["budgets"]):
        if all(r["coverage"][a][i] < 0.5 for a in arms):
            continue
        cells = [f"{r['pass_at_tokens'][a][i]:.3f}" + star(r["coverage"], i, a) for a in arms]
        j = t1["budgets"].index(b) if b in t1["budgets"] else None
        L.append(f"| {b} | " + " | ".join(cells) + " | " + "/".join(str(r["attempts_at"][a][i]) for a in arms) +
                 f" | {t1['instruct'][j] if j is not None else ''} | {t1['DecompRL'][j] if j is not None else ''} |")
    L.append("\n`*` fewer than 95% of problems have k graded attempts (pass@n with n < k, biased low).\n")
    if "primary_idea_minus_decomp" in r:
        x = r["primary_idea_minus_decomp"]
        L.append(f"**Primary, idea - decomp at {x['budget']} tokens:** {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
                 f" -> {'passes (CI above 0)' if x['lo'] > 0 else 'does not pass'}.\n")
    for k, v in r.get("secondary", {}).items():
        L.append(f"- {k} at {v['budget']}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    for b, v in r.get("at", {}).items():
        L.append(f"- at {b}: " + ", ".join(f"{k} {w['mean']:+.3f} [{w['lo']:+.3f}, {w['hi']:+.3f}]" for k, w in v.items()))
    L.append("\nHarness check against Table 1 instruct: " + ", ".join(
        f"{b}: {v['plain']:.3f} vs {v['paper_instruct']} ({'ok' if v['within_0.05'] else 'off'}"
        f"{'' if v['covered'] else ', not covered'})" for b, v in r["harness_check"].items()))
    x = r["check_exact_tokens"]
    L += ["\nCheck, exact-token rule (an attempt counts once all its tokens fit):\n",
          "| tokens | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for i, b in enumerate(r["budgets"]):
        if all(x["coverage"][a][i] < 0.5 for a in arms):
            continue
        L.append(f"| {b} | " + " | ".join(f"{x['pass_at_tokens'][a][i]:.3f}" + star(x["coverage"], i, a)
                                          for a in arms) + " |")
    if "idea_minus_decomp" in x:
        y = x["idea_minus_decomp"]
        L.append(f"\nidea - decomp at {y['budget']} (exact-token rule): {y['mean']:+.3f} [{y['lo']:+.3f}, {y['hi']:+.3f}]")
    L.append("\n```\n" + json.dumps({"arms": r["arms"], "by_difficulty": r["by_difficulty"],
                                     "by_difficulty_all_arms": r["by_difficulty_all_arms"], "llm": r["llm"],
                                     "grader": r["grader"]}, indent=1) + "\n```")
    return "\n".join(L)


def write(state, problems, out, budgets=BUDGETS):
    r = build(state, problems, budgets=budgets)
    Path(out, "report.json").write_text(json.dumps(r, indent=1))
    Path(out, "report.md").write_text(markdown(r))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--lcb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from lcb_data import load
    print(markdown(write(json.load(open(a.state)), load(a.lcb), a.out)))
