"""Stage B dev comparison (README "Stage B, round 1"): the untrained model's three arms beside each trained model's
own arm, on generated tokens and on GPU-seconds. For steering only; the 279 stay untouched.

    python3 -I dev_report.py --base dev_base/state.json --trained dev_plain/state.json dev_decomp/state.json \
        dev_idea/state.json --out dev_report.json

Series: plain, decomp, idea (untrained) and plain+, decomp+, idea+ (trained; each trained run samples one arm).
Problems: those graded in every series. Cost per attempt: generated tokens per graded attempt (report.py's rule), or
generation seconds per attempt (state timing / attempts, as rescore.py). A budget buys k = attempts_at(budget, cost)
attempts, scored by unbiased pass@k, and is covered when COVER of problems have k graded attempts.
Comparisons, each at the largest budget both series cover, paired bootstrap 95% CI over problems: trained minus
untrained per arm; idea+ - decomp+; decomp+ - plain+ and idea+ - plain+.
"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import ARMS, BUDGETS, COVER, attempts_at, boot, pass_at_k

SECONDS = [5, 10, 15, 20, 40, 60, 80, 120, 160]
PAIRS = [(f"{a}+", a) for a in ARMS] + [("idea+", "decomp+"), ("decomp+", "plain+"), ("idea+", "plain+")]


def series(base, trained):
    """-> {name: (graded attempts by problem, generation seconds per attempt)}"""
    out = {}
    runs = [(base, "", ARMS)] + [(t, "+", [a for a in ARMS if t["attempts"].get(a)]) for t in trained]
    for s, suffix, arms in runs:
        assert suffix == "" or len(arms) == 1, f"a trained run samples one arm, this one has {arms}"
        for arm in arms:
            assert arm + suffix not in out, f"two runs of {arm + suffix}"
            att = s["attempts"][arm]
            out[arm + suffix] = ({q: [a for a in v if a.get("passed") is not None] for q, v in att.items()},
                                 s["timing"][arm][0] / sum(len(v) for v in att.values()))
    return out


def build(base, trained, seed=0, budgets=BUDGETS, seconds=SECONDS):
    S, rng = series(base, trained), random.Random(seed)
    ids = sorted(q for q in set.intersection(*(set(a) for a, _ in S.values())) if all(S[x][0][q] for x in S))
    assert ids, "no problem is graded in every series"
    tok = {x: sum(a["tokens"] for q in ids for a in att[q]) / sum(len(att[q]) for q in ids)
           for x, (att, _) in S.items()}
    r = {"problems": len(ids), "tokens_per_attempt": tok, "seconds_per_attempt": {x: s for x, (_, s) in S.items()}}
    for unit, budgets, cost in (("tokens", budgets, tok), ("gpu_seconds", seconds, r["seconds_per_attempt"])):
        C = {}
        for x, (att, _) in S.items():
            C[x] = []
            for b in budgets:
                k = attempts_at(b, cost[x])
                per = {q: pass_at_k(len(att[q]), sum(bool(a["passed"]) for a in att[q]), k) for q in ids}
                C[x].append({"k": k, "pass": sum(per.values()) / len(ids),
                             "coverage": sum(len(att[q]) >= k for q in ids) / len(ids), "per": per})
        cmp = {}
        for x, y in PAIRS:
            ok = [i for i in range(len(budgets))
                  if x in C and y in C and min(C[x][i]["coverage"], C[y][i]["coverage"]) >= COVER]
            if ok:
                i = ok[-1]
                cmp[f"{x} - {y}"] = {"budget": budgets[i],
                                     **boot([C[x][i]["per"][q] - C[y][i]["per"][q] for q in ids], rng)}
        r[unit] = {"budgets": budgets, "series": {x: [{k: v for k, v in c.items() if k != "per"} for c in C[x]]
                                                  for x in C}, "compare": cmp}
    return r


def markdown(r):
    L = [f"# Stage B dev: {r['problems']} problems graded in every series (steering only)\n",
         "Per attempt: " + ", ".join(f"{x} {r['tokens_per_attempt'][x]:.0f} tokens / {r['seconds_per_attempt'][x]:.1f} s"
                                     for x in r["tokens_per_attempt"]) + "\n"]
    for unit in ("tokens", "gpu_seconds"):
        u, names = r[unit], list(r[unit]["series"])
        L += [f"\n| {unit} | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for i, b in enumerate(u["budgets"]):
            cells = [u["series"][x][i] for x in names]
            if all(c["coverage"] < 0.5 for c in cells):
                continue
            L.append(f"| {b} | " + " | ".join(f"{c['pass']:.3f}" + ("" if c["coverage"] >= COVER else "*")
                                              for c in cells) + " |")
        L.append("")
        L += [f"- {k} at {v['budget']} {unit}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
              for k, v in u["compare"].items()]
    L.append(f"\n`*` fewer than {COVER:.0%} of problems have k graded attempts (pass@n with n < k, biased low).")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", nargs="+", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    r = build(json.load(open(a.base)), [json.load(open(t)) for t in a.trained])
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print(markdown(r))
