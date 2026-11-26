"""Stage A scheduler: the three arms topped up to the attempts a token budget buys under the paper's rule.

An attempt is DecompRL's unit (Appendix K): plain = one generation; decomp / idea = one decomposition, its k
implementations per unit, and up to `evals` assembled programs, solved if any passes. Every generated token is
counted against the attempt, including the idea's MISSING replies, helpers and resumed callers. A budget B buys
floor(B / c) attempts, c being the arm's mean tokens per attempt (report.py), so top_up gives each problem
max(1, floor(margin * B / c)) attempts per arm, with c measured on everything generated so far (PRIOR before that).
A problem can get several attempts of one arm in the same batch; they are independent samples.

decomp  DecompRL's hierarchical inference: all units implemented independently from the stubs, k each; programs
        are uniform random combinations (their evaluation samples combinations the same way, up to 4096).
idea    The user's build order: the decomposition names what each function uses; units are built level by level,
        basics first. Each of a unit's k samples is shown one random already-built implementation of each function
        it uses (the dependency in view), and it can stop with MISSING: the helper is built, then the unit is asked
        again. A checker also catches calls to functions nobody defined and builds those helpers. Programs: first
        the k "coherent" ones (each top unit with exactly the implementations it was shown), then random
        combinations, up to the same `evals`.
plain   LCB's prompt; the same rule, up to the budget the run caps it at (the paper's 1k-10k points).
Units beyond nmax (DecompRL's n_max = 6) fail the attempt, for both decomposition arms alike.
"""
import gzip, json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import decomp as D
import prompts as P

ARMS = ("plain", "decomp", "idea")
PRIOR = {"plain": 207.0, "decomp": 14000.0, "idea": 5800.0}     # tokens per attempt in the 2026-11-10 pilot


class StageA:
    def __init__(self, llm, grader, out, k=8, nmax=6, evals=128, max_plain=2048, max_decomp=2048, max_impl=1024,
                 max_missing=2, session=0, t0=None, log=print, train=False, arms=ARMS):
        self.llm, self.grader, self.out, self.log = llm, grader, Path(out), log
        self.k, self.nmax, self.evals, self.max_missing = k, nmax, evals, max_missing
        self.max_plain, self.max_decomp, self.max_impl = max_plain, max_decomp, max_impl
        self.session, self.prev, self.t0 = session, [], t0 or time.time()
        self.arms = arms                                          # what need() tops up
        self.rng = random.Random(session)
        self.att = {a: defaultdict(list) for a in ARMS}
        self.timing = {a: [0.0, 0] for a in ARMS}                 # seconds, generated tokens
        self.out.mkdir(parents=True, exist_ok=True)
        self.raw = {a: gzip.open(self.out / f"raw_{a}.jsonl.gz", "at") for a in ARMS}
        # Stage B: a passing attempt's training examples go to train_<arm>.jsonl.gz once the grader says which
        # program passed first; gid -> (arm, problem id, first-pass index -> [(messages, reply)])
        self.trainf = {a: gzip.open(self.out / f"train_{a}.jsonl.gz", "at") for a in ARMS} if train else None
        self.pending = {}

    # ------------------------------------------------------------ helpers
    def gen(self, reqs, owners=None):
        """owners: the attempt each request belongs to, charged its prompt tokens (prefill once per request)."""
        if not reqs:
            return []
        res, lens = self.llm.generate(reqs)
        for a, n in zip(owners or [], lens):
            a["prompt_tokens"] = a.get("prompt_tokens", 0) + n
        self.grader.poll()
        return res

    def tokens(self, arm, qid):
        return sum(a["tokens"] for a in self.att[arm][qid])

    def cost(self, arm):
        n = sum(len(v) for v in self.att[arm].values())
        return sum(self.tokens(arm, q) for q in self.att[arm]) / n if n else PRIOR[arm]

    def need(self, probs, budget, caps, margin):
        """New attempts per arm and problem for every problem to have what `budget` buys (capped per arm)."""
        out = {}
        for arm in self.arms:
            b = min(budget, caps.get(arm, budget))
            want = max(1, int(margin * b // self.cost(arm)))
            out[arm] = {p["id"]: want - len(self.att[arm][p["id"]]) for p in probs
                        if len(self.att[arm][p["id"]]) < want}
        return out

    def seconds_for(self, need):
        """Predicted time for `need`, from this run's seconds per generated token; None before an arm has run."""
        t = 0.0
        for arm, d in need.items():
            if d and not self.timing[arm][1]:
                return None
            t += sum(d.values()) * self.cost(arm) * self.timing[arm][0] / max(self.timing[arm][1], 1)
        return t

    def top_up(self, probs, need):
        for arm, f in (("decomp", self.decomp_round), ("idea", self.idea_round), ("plain", self.plain_round)):
            todo = [p for p in probs for _ in range(need.get(arm, {}).get(p["id"], 0))]
            if todo:
                t, g = time.time(), self.llm.gen_tokens
                f(todo)
                self.timing[arm][0] += time.time() - t
                self.timing[arm][1] += self.llm.gen_tokens - g

    def _add(self, arm, p, a, progs, raw, train=None):
        a["n_programs"] = len(progs)
        a["gid"] = self.grader.submit(p["id"], progs) if progs else None
        a["session"] = self.session
        if self.trainf is not None and a["gid"] is not None:
            self.pending[a["gid"]] = (arm, p["id"], train)
        self.att[arm][p["id"]].append(a)
        self.raw[arm].write(json.dumps({"id": p["id"], "attempt": len(self.att[arm][p["id"]]) - 1, **raw,
                                        "programs_head": progs[:2]}) + "\n")

    def _decompose(self, probs, uses):
        atts = [{"tokens": 0, "units": 0, "reason": None} for _ in probs]
        outs = self.gen([(P.decomp_messages(p, uses=uses), 1, self.max_decomp) for p in probs], atts)
        res = []
        for p, o, a in zip(probs, outs, atts):
            text, nt = o[0]
            a["tokens"] += nt
            d = D.parse_decomposition(text)
            if d is None:
                a["reason"] = "no_decomposition"
            else:
                a["units"] = len(d.units)
                a["levels"] = D.dependency_levels(d)
                a["uses_lines"] = sum(u.uses_raw is not None for u in d.units)
                if len(d.units) > self.nmax:
                    a["reason"] = "too_many_units"
            res.append((p, a, d if a["reason"] is None else None, text))
        return res

    def _programs(self, p, d, impls, first=()):
        """-> (programs, the combination each was assembled from)."""
        cs = D.combos(d, impls, self.evals, self.rng, first)
        return [D.assemble(d, {n: (impls[n][i] if i is not None else None) for n, i in c.items()},
                           p["testtype"], p["fn_name"]) for c in cs], cs

    @staticmethod
    def _examples(p, uses, text, impls, cs):
        """The training examples of program f: the decomposition and each implementation in it, with its prompt."""
        return lambda f: [(P.decomp_messages(p, uses=uses), text)] + [
            (impls[n][i].msgs, impls[n][i].text) for n, i in cs[f].items() if i is not None]

    # ------------------------------------------------------------ DecompRL
    def decomp_round(self, probs):
        dec = self._decompose(probs, uses=False)
        jobs = []
        for i, (p, a, d, _) in enumerate(dec):                    # keyed by position: a problem can come twice
            if d is None:
                continue
            shown = d.render()
            for u in d.units:
                others = [v.header for v in d.units if v.name != u.name]
                jobs.append((i, a, d, u, P.impl_messages(p, shown, u.fname, others, u.stub)))
        outs = self.gen([(m, self.k, self.max_impl) for *_, m in jobs], [j[1] for j in jobs])
        impls, texts = defaultdict(dict), defaultdict(dict)
        for (i, a, d, u, m), o in zip(jobs, outs):
            fn = {v.fname for v in d.units}
            lst = []
            for text, nt in o:
                a["tokens"] += nt
                im = D.parse_impl(text, u, fn)
                if im:
                    im.tokens, im.msgs, im.text = nt, m, text
                lst.append(im)
            impls[i][u.name] = lst
            texts[i][u.name] = [t for t, _ in o]
        for i, (p, a, d, text) in enumerate(dec):
            progs, train = [], None
            if d is not None:
                ims = impls[i]
                a["parsed_impls"] = sum(im is not None for lst in ims.values() for im in lst)
                progs, cs = self._programs(p, d, ims)
                train = self._examples(p, False, text, ims, cs)
            self._add("decomp", p, a, progs, {"decomposition": text, "impls": texts.get(i, {})}, train)

    # ------------------------------------------------------------ the idea
    def idea_round(self, probs):
        dec = self._decompose(probs, uses=True)
        st = []
        for p, a, d, text in dec:
            a.update(missing_explicit=0, missing_checker=0, helpers_built=0, resumed=0)
            s = {"p": p, "a": a, "d": d, "text": text, "texts": defaultdict(list)}
            if d is not None:
                s["fn"] = {u.fname for u in d.units}
                s["impls"] = {u.name: [None] * self.k for u in d.units}
            st.append(s)
        live = [s for s in st if s["d"] is not None]
        depth = max((len(s["a"]["levels"]) for s in live), default=0)
        for level in range(depth):
            reqs, meta = [], []
            for s in live:
                d, p = s["d"], s["p"]
                for name in (s["a"]["levels"][level] if level < len(s["a"]["levels"]) else []):
                    u = d.unit(name)
                    built = [w for w in d.deps[name] if any(s["impls"][w])]    # lower levels are done by now
                    for j in range(self.k):
                        seen = {w: self.rng.choice([i for i, im in enumerate(s["impls"][w]) if im]) for w in built}
                        shown = d.render({w: s["impls"][w][i].def_src for w, i in seen.items()})
                        others = [v.header for v in d.units if v.name != name]
                        reqs.append((P.impl_messages(p, shown, u.fname, others, u.stub, idea=True), 1, self.max_impl))
                        meta.append((s, u, j, seen, shown, others))
            outs = self.gen(reqs, [m[0]["a"] for m in meta])
            helper_reqs, hmeta, stopped = [], [], []
            for (s, u, j, seen, shown, others), (msgs, _, _), o in zip(meta, reqs, outs):
                text, nt = o[0]
                s["a"]["tokens"] += nt
                s["texts"][u.name].append(text)
                miss = D.parse_missing(text)
                if miss:                                   # it stopped: build what it named, then ask again
                    s["a"]["missing_explicit"] += 1
                    pend = {"s": s, "u": u, "j": j, "seen": seen, "shown": shown, "others": others,
                            "imports": [], "helpers": [], "built": []}
                    stopped.append(pend)
                    for name, sig in miss[:self.max_missing]:
                        helper_reqs.append((P.helper_messages(s["p"], shown, name, signature=sig, caller=u.fname),
                                            1, self.max_impl))
                        hmeta.append((s, pend, name))
                    continue
                im = D.parse_impl(text, u, s["fn"])
                if im is None:
                    continue
                im.tokens, im.seen, im.msgs, im.text = nt, seen, msgs, text
                s["impls"][u.name][j] = im
                und = D.undefined_calls(im, s["d"])
                if und:                                    # it called something nobody wrote: build it
                    s["a"]["missing_checker"] += 1
                    for name in und[:self.max_missing]:
                        helper_reqs.append((P.helper_messages(s["p"], shown, name, caller=u.fname,
                                                              caller_code=im.def_src), 1, self.max_impl))
                        hmeta.append((s, im, name))
            for (s, target, name), o in zip(hmeta, self.gen(helper_reqs, [h[0]["a"] for h in hmeta])):
                text, nt = o[0]
                s["a"]["tokens"] += nt
                h = D.parse_helper(text, name)
                if not h:
                    continue
                s["a"]["helpers_built"] += 1
                if isinstance(target, dict):               # for a unit that stopped: shown when it is asked again
                    target["imports"] += h[0]
                    target["helpers"] += h[1]
                    target["built"].append(name)
                else:                                      # a checker helper: attached to the implementation
                    target.imports += h[0]
                    target.helpers += h[1]
                    target.built.append(name)
            resume = [(pd, P.impl_messages(pd["s"]["p"], "\n\n".join([pd["shown"]] + pd["imports"] + pd["helpers"]),
                                           pd["u"].fname, pd["others"] + [f"{n} (shown above)" for n in pd["built"]],
                                           pd["u"].stub, idea=True)) for pd in stopped]
            for (pd, msgs), o in zip(resume, self.gen([(m, 1, self.max_impl) for _, m in resume],
                                                      [pd["s"]["a"] for pd, _ in resume])):
                text, nt = o[0]
                s, u = pd["s"], pd["u"]
                s["a"]["tokens"] += nt
                s["a"]["resumed"] += 1
                s["texts"][u.name].append(text)
                im = D.parse_impl(text, u, s["fn"])       # a second MISSING has no code, so it stays empty
                if im:
                    im.tokens, im.seen, im.built, im.msgs, im.text = nt, pd["seen"], list(pd["built"]), msgs, text
                    im.imports, im.helpers = pd["imports"] + im.imports, pd["helpers"] + im.helpers
                    s["impls"][u.name][pd["j"]] = im
        for s in st:
            p, a, d = s["p"], s["a"], s["d"]
            progs, coherent, train = [], [], None
            if d is not None:
                ims = s["impls"]
                a["parsed_impls"] = sum(im is not None for lst in ims.values() for im in lst)
                top = D.roots(d)[0]
                for j, im in enumerate(ims[top]):
                    if im is None:
                        continue
                    c, todo = {top: j}, [top]
                    while todo:
                        v = todo.pop()
                        for w, i in ims[v][c[v]].seen.items():
                            if w not in c:
                                c[w] = i
                                todo.append(w)
                    for u in d.units:                      # units the top one never reaches: any built one
                        if u.name not in c:
                            ok = [i for i, x in enumerate(ims[u.name]) if x]
                            c[u.name] = self.rng.choice(ok) if ok else None
                    coherent.append(c)
                a["coherent"] = len(coherent)
                progs, cs = self._programs(p, d, ims, first=coherent)
                train = self._examples(p, True, s["text"], ims, cs)
            self._add("idea", p, a, progs, {"decomposition": s["text"], "impls": dict(s["texts"])}, train)

    # ------------------------------------------------------------ plain
    def plain_round(self, probs, per_request=64):
        reqs, who = [], []
        for qid, n in Counter(p["id"] for p in probs).items():
            p = next(x for x in probs if x["id"] == qid)
            for j in range(0, n, per_request):
                reqs.append((P.plain_messages(p), min(per_request, n - j), self.max_plain))
                who.append(p)
        holders = [{} for _ in reqs]
        for p, o, h, (msgs, _, _) in zip(who, self.gen(reqs, holders), holders, reqs):
            for text, nt in o:                             # n samples share one prompt: each pays 1/n of it
                code = D.lcb_extract(text)
                self._add("plain", p, {"tokens": nt, "prompt_tokens": h["prompt_tokens"] / len(o)},
                          [code] if code.strip() else [], {"text": text}, lambda f, m=msgs, t=text: [(m, t)])

    # ------------------------------------------------------------ results
    def collect(self):
        """Fill this session's attempts' results from the grader. None = not graded yet."""
        self.grader.poll()
        for gid in [g for g in self.pending if g in self.grader.done]:
            arm, qid, build = self.pending.pop(gid)
            r = self.grader.done[gid]
            if r["passed"]:
                self.trainf[arm].write(json.dumps({"id": qid, "examples": [
                    {"messages": m, "reply": t} for m, t in build(r["first_pass"])]}) + "\n")
        for arm in ARMS:
            for lst in self.att[arm].values():
                for a in lst:
                    if a["session"] != self.session:
                        continue
                    if a.get("gid") is None:
                        a["passed"] = False
                    elif a["gid"] in self.grader.done:
                        r = self.grader.done[a["gid"]]
                        a.update(passed=r["passed"], first_pass=r["first_pass"], n_graded=r["n_graded"],
                                 statuses=dict(r["statuses"]))
                    else:
                        a["passed"] = None

    def load(self, state):
        """Continue an earlier session. Its graded attempts are kept; ungraded ones are dropped, because their
        programs went with that session's grader, and the next top-up samples them again. -> attempts dropped."""
        dropped = 0
        for arm in ARMS:
            for q, lst in state["attempts"][arm].items():
                keep = [a for a in lst if a.get("passed") is not None]
                dropped += len(lst) - len(keep)
                self.att[arm][q] = keep
        self.prev = state.get("sessions", [])
        self.timing = state.get("timing", self.timing)
        return dropped

    def save(self, extra=None):
        self.collect()
        for f in list(self.raw.values()) + list((self.trainf or {}).values()):
            f.flush()
        cur = {"session": self.session, "hours": (time.time() - self.t0) / 3600, "llm": {
            "gen_tokens": self.llm.gen_tokens, "prompt_tokens": self.llm.prompt_tokens, "seconds": self.llm.seconds,
            "too_long": self.llm.too_long}, "grader": {"seconds": self.grader.seconds, "counts": dict(self.grader.counts),
            "pending": self.grader.pending(), "errors": self.grader.errors[:50]}}
        sessions = self.prev + [cur]
        state = {"attempts": {a: dict(v) for a, v in self.att.items()}, "sessions": sessions, "timing": self.timing,
                 "llm": {k: sum(s["llm"][k] for s in sessions) for k in cur["llm"]}, "grader": cur["grader"],
                 **(extra or {})}
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(self.out / "state.json")

    def close(self):
        for f in list(self.raw.values()) + list((self.trainf or {}).values()):
            f.close()
