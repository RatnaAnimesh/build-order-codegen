"""Stage B collection: untrained Qwen samples every arm on the CodeContests pool with equal generation GPU time.

    python collect.py --model /tmp/qwen --lcb cc_train/ --out out/ --arm_hours 1.5 --gen_hours 5.2 --total_hours 5.8

Arms take turns one chunk at a time, the arm with the least generation time so far going next, until each has
--arm_hours (Stage A's `timing`: wall time spent generating that arm's requests). A turn gives the arm's next
--chunk problems of the shuffled pool one attempt each (plain --plain_per), cycling through the pool. Passing
attempts' training examples land in train_<arm>.jsonl.gz (arms.py). A turn starts only if its predicted time fits
before --gen_hours; grading then gets until --total_hours. Hours count from --t0 (the job's start).
"""
import argparse, json, random, sys, time
from pathlib import Path

from arms import ARMS, StageA
from lcb_data import load
from lcb_exec import Grader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--lcb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--arm_hours", type=float, default=1.5)
    ap.add_argument("--gen_hours", type=float, default=5.2)
    ap.add_argument("--total_hours", type=float, default=5.8)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--plain_per", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--evals", type=int, default=128)
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--scripted", default=None, help="module:function giving a stand-in reply (dry run, trusted code)")
    a = ap.parse_args()
    t0 = a.t0 or time.time()
    gen_end, total_end, limit = t0 + a.gen_hours * 3600, t0 + a.total_hours * 3600, a.arm_hours * 3600
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(*x):
        line = f"[{(time.time() - t0) / 60:6.1f}m] " + " ".join(str(y) for y in x)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    probs = load(a.lcb)
    random.Random(0).shuffle(probs)
    probs = probs[:a.limit or None]
    log(f"collect: {len(probs)} problems, {a.arm_hours} h per arm, chunk {a.chunk}, plain {a.plain_per} per pass")
    grader = Grader(a.lcb, workers=a.workers, trusted=bool(a.scripted))
    if a.scripted:
        import importlib
        from llm import ScriptedChat
        mod, fn = a.scripted.split(":")
        llm = ScriptedChat(getattr(importlib.import_module(mod), fn))
    else:
        from llm import VLLMChat
        llm = VLLMChat(a.model, tp=a.tp, max_model_len=a.max_model_len, log=log)
    run = StageA(llm, grader, out, k=a.k, nmax=a.nmax, evals=a.evals, t0=t0, log=log, train=True)
    pos, stop = dict.fromkeys(ARMS, 0), None
    while True:
        arm = min(ARMS, key=lambda x: run.timing[x][0])
        if run.timing[arm][0] >= limit:
            break
        ch = [probs[(pos[arm] + i) % len(probs)] for i in range(a.chunk)]
        need = {x: ({p["id"]: a.plain_per if arm == "plain" else 1 for p in ch} if x == arm else {}) for x in ARMS}
        pred = run.seconds_for(need)
        if pred is not None and time.time() + 1.15 * pred > gen_end:
            stop = f"stop before {arm}'s turn at {pos[arm]}: predicted {pred / 60:.1f} min doesn't fit"
            break
        ts = time.time()
        run.top_up(ch, need)
        pos[arm] += a.chunk
        run.save({"pos": pos})
        log(f"{arm} turn to {pos[arm]} in {(time.time() - ts) / 60:.1f} min; generation h "
            + ", ".join(f"{x} {run.timing[x][0] / 3600:.2f}" for x in ARMS) + f"; grader pending {grader.pending()}")
    log(stop or "every arm has its hours")
    finished = grader.wait(deadline=total_end)
    run.save({"pos": pos, "grading_finished": finished, "all_rounds_done": stop is None})
    run.close()
    grader.close()
    st = json.load(open(out / "state.json"))
    for x in ARMS:
        att = [t for v in st["attempts"][x].values() for t in v]
        log(f"{x}: {len(att)} attempts, {sum(bool(t.get('passed')) for t in att)} passed, "
            f"{run.timing[x][0] / 3600:.2f} GPU-h, {run.timing[x][1]} generated tokens")
    log("DONE" if finished else "DONE (grading cut at the deadline)")


if __name__ == "__main__":
    sys.exit(main())
