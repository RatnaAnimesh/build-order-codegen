"""Stage A run: plain vs DecompRL inference vs the idea on Qwen2.5-7B-Instruct, LCB 279, no training.

    python run_stage_a.py --model /tmp/qwen --lcb lcb/ --out out/ --gen_hours 10 --total_hours 10.6 [--resume]

Problems are shuffled once (seed 0) and cut into chunks. Rounds raise a token budget per problem (--targets); in
each round every chunk is topped up, arm by arm, to the attempts that budget buys under the paper's rule, plain only
up to --plain_cap. A chunk-round starts only if its predicted time (seconds per generated token so far) fits before
--gen_hours, so a session that runs out leaves every problem at the same round, give or take one chunk. Grading
runs in the background meanwhile; after generation it gets until --total_hours, then the report is written from
what is graded. Hours are counted from --t0 (the job's start), so setup time counts against the budget.
After the last round, coverage passes top up, without the margin, any problem still short of what the last budget
buys at the final tokens per attempt (cheap and costly problems move that mean as attempts come in).
--resume continues the state.json in --out from an earlier session: rounds and chunks already topped up need
nothing and are skipped, and the session number seeds both the sampler and the combination order.
"""
import argparse, json, random, sys, time
from pathlib import Path

from arms import StageA
from lcb_data import load
from lcb_exec import Grader
import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--lcb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--gen_hours", type=float, default=10.0)
    ap.add_argument("--total_hours", type=float, default=10.6)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--targets", default="5000,10000,20000,30000,40000,50000")
    ap.add_argument("--plain_cap", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=1.0, help="> 1 samples ahead of drift in tokens per attempt; "
                                                             "the coverage passes catch it anyway")
    ap.add_argument("--max_chunk_rounds", type=int, default=0, help="stop after this many in this session (tests)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--evals", type=int, default=128)
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--budgets", default=",".join(map(str, report.BUDGETS)))
    ap.add_argument("--arms", default="plain,decomp,idea", help="Stage B dev runs: one arm per trained model")
    ap.add_argument("--scripted", default=None, help="module:function giving a stand-in reply (dry run, trusted code)")
    a = ap.parse_args()
    t0 = a.t0 or time.time()
    gen_end, total_end = t0 + a.gen_hours * 3600, t0 + a.total_hours * 3600
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(*x):
        line = f"[{(time.time() - t0) / 60:6.1f}m] " + " ".join(str(y) for y in x)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prev = json.load(open(out / "state.json")) if a.resume else None
    session = len(prev.get("sessions", [])) if prev else 0
    probs = load(a.lcb)
    random.Random(0).shuffle(probs)
    if a.limit:
        probs = probs[:a.limit]
    targets = [int(x) for x in a.targets.split(",")]
    log(f"session {session}: {len(probs)} problems, arms {a.arms}; k={a.k} evals={a.evals} nmax={a.nmax} "
        f"chunk={a.chunk} targets={targets} plain_cap={a.plain_cap} margin={a.margin}")
    grader = Grader(a.lcb, workers=a.workers, trusted=bool(a.scripted))
    if a.scripted:
        import importlib
        from llm import ScriptedChat
        mod, fn = a.scripted.split(":")
        llm = ScriptedChat(getattr(importlib.import_module(mod), fn))
    else:
        from llm import VLLMChat
        llm = VLLMChat(a.model, tp=a.tp, max_model_len=a.max_model_len, seed=session, log=log)
    log("model ready")
    run = StageA(llm, grader, out, k=a.k, nmax=a.nmax, evals=a.evals, session=session, t0=t0, log=log,
                 arms=tuple(a.arms.split(",")))
    done = []
    if prev:
        dropped = run.load(prev)
        done = prev.get("chunks_done", [])
        log(f"resumed {sum(len(v) for arm in run.att.values() for v in arm.values())} graded attempts from "
            f"{session} earlier session(s); dropped {dropped} ungraded ones (sampled again)")
    caps = {"plain": a.plain_cap}
    chunks = [probs[i:i + a.chunk] for i in range(0, len(probs), a.chunk)]
    ran, stop = 0, None
    for b, margin in [(b, a.margin) for b in targets] + [(targets[-1], 1.0)] * 3:     # then coverage passes
        for ci, ch in enumerate(chunks):
            need = run.need(ch, b, caps, margin)
            if not any(need.values()):
                continue
            plan = {arm: sum(d.values()) for arm, d in need.items()}
            pred = run.seconds_for(need)
            if a.max_chunk_rounds and ran >= a.max_chunk_rounds:
                stop = f"stop before {b} chunk {ci}: --max_chunk_rounds {a.max_chunk_rounds}"
            elif pred is not None and time.time() + 1.15 * pred > gen_end:
                stop = f"stop before {b} chunk {ci}: predicted {pred / 60:.1f} min for {plan} doesn't fit"
            if stop:
                break
            ts = time.time()
            run.top_up(ch, need)
            ran += 1
            took = time.time() - ts
            done.append([session, b, margin, ci, len(ch), plan, round(took)])
            run.save({"chunks_done": done})
            log(f"budget {b} x{margin} chunk {ci}: {plan} in {took / 60:.1f} min (predicted "
                f"{'-' if pred is None else f'{pred / 60:.1f}'}); tokens per attempt "
                + ", ".join(f"{arm} {run.cost(arm):.0f}" for arm in run.arms)
                + f"; grader pending {grader.pending()}, graded {sum(grader.counts.values())} programs "
                f"{dict(grader.counts)}")
        if stop:
            log(stop)
            break
    else:
        log("all rounds done")
    log("generation over; grading the rest")
    finished = grader.wait(deadline=total_end)
    run.save({"chunks_done": done, "grading_finished": finished, "all_rounds_done": stop is None})
    run.close()
    state = json.load(open(out / "state.json"))
    r = report.write(state, probs, out, budgets=[int(x) for x in a.budgets.split(",")])
    log("report:\n" + report.markdown(r))
    grader.close()
    log("DONE" if finished else "DONE (grading cut at the deadline; ungraded attempts left out)")


if __name__ == "__main__":
    sys.exit(main())
