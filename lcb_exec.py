"""LiveCodeBench grader with DecompRL's time rule. Generated code runs only inside a Kaggle session.

Grading copies lcb_runner/evaluation/testing_util.py:
  * LCB's import prelude (`from collections import *`, ..., recursion limit 50000) goes above every program;
  * functional (LeetCode): exec once, `Solution()` once if "class Solution" is in the code, then method(*json args)
    per test; tuples count as lists; plain ==; SystemExit fails;
  * stdin (AtCoder): the program runs once per test with that test's input on fd 0 and sys.stdin; SystemExit is
    fine; output lines are stripped and compared exactly, else element-wise as Decimals.
Tests run in LCB's order (public, then private) and stop at the first failure.

Time, as in DecompRL (Appendix H): 2T+1 s for all T tests. It is CPU time here (RLIMIT_CPU), because the GPU job
shares 4 vCPUs with vLLM and wall time would punish correct programs for our contention. A hang that uses no CPU is
caught by wall limits per test: PUBLIC_WALL for the small public tests, PRIVATE_WALL after. These only bound cost;
any program a 2T+1 CPU rule passes and that makes progress at least every PRIVATE_WALL s passes here too.

Isolation: each program runs in a forked child of a spawned worker (no CUDA, no threads, so forking is safe), in
its own session (killed as a group), in a scratch directory, with memory, file-size and CPU limits, stderr to
/dev/null. A child that dies without its final byte (segfault, os._exit, CPU limit) fails.
"""
import builtins, io, json, multiprocessing as mp, os, resource, select, signal, sys, tempfile, threading, time
from collections import Counter, OrderedDict
from decimal import Decimal
from hashlib import sha1
from pathlib import Path

from lcb_data import safe_id

PRELUDE = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\n"
           "from bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\n"
           "from itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\n"
           "from json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
           "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
           "import itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\n"
           "sys.setrecursionlimit(50000)\n")
PUBLIC_WALL, PRIVATE_WALL = 2.5, 12.0
MEM = 3 << 30                      # address space allowed above what the worker already maps
FSIZE = 256 << 20
CACHE = 3                          # problems' tests a worker keeps in memory (up to ~200 MB of JSON each)
STATUS = {ord("P"): "pass", ord("F"): "wrong", ord("E"): "error", ord("C"): "compile"}


def cpu_limit(n_tests):
    return 2 * n_tests + 1


def stripped_lines(s):
    return [x.strip() for x in s.strip().split("\n")]


def _decimals(line):
    try:
        return [Decimal(e) for e in line.split()]
    except Exception:
        return None


def stdio_match(pred, gt):
    a, b = stripped_lines(pred), stripped_lines(gt)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        dx, dy = _decimals(x), _decimals(y)
        if dx is None or dy is None or dx != dy:
            return False
    return True


# ---------------------------------------------------------------- child (one program)

def _send(w, b):
    try:
        os.write(w, b)
    except OSError:
        pass


def _vmsize():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    except OSError:                                       # not Linux (local tests): no address-space limit
        pass
    return 0


def _join_threads():
    for th in threading.enumerate():
        if th is not threading.main_thread() and not th.daemon:
            th.join()


def _functional(w, code, t):
    try:
        g = {"__name__": "tmp_sol", "__builtins__": builtins}
        exec(compile(PRELUDE + "\n\n" + code, "<solution>", "exec"), g)
        sol = g["Solution"]() if "class Solution" in code else None
        method = getattr(sol, t["fn_name"]) if sol is not None else g[t["fn_name"]]
    except BaseException:
        return _send(w, b"C")
    for args, exp in zip(t["args"], t["expected"]):
        try:
            pred = method(*args)
            if isinstance(pred, tuple):
                pred = list(pred)
            ok = bool(pred == exp)
        except BaseException:
            return _send(w, b"E")
        if not ok:
            return _send(w, b"F")
        _send(w, b".")
    _send(w, b"P")


def _stdio(w, code, t):
    try:
        prog = compile(PRELUDE + "\n\n" + code, "<solution>", "exec")
    except BaseException:
        return _send(w, b"C")
    for inp, out in zip(t["inputs"], t["outputs"]):
        with open("in.txt", "w", encoding="utf-8") as f:
            f.write(inp)
        fi = os.open("in.txt", os.O_RDONLY)
        if fi != 0:                                       # fd 0 is free if the last test's program closed it
            os.dup2(fi, 0)
            os.close(fi)
        fo = os.open("out.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        if fo != 1:
            os.dup2(fo, 1)
            os.close(fo)
        sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, "r", closefd=False)), encoding="utf-8")
        sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.FileIO(1, "w", closefd=False)), encoding="utf-8")
        try:
            exec(prog, {"__name__": "__main__", "__builtins__": builtins})
            _join_threads()
        except SystemExit:
            pass
        except BaseException:
            return _send(w, b"E")
        try:
            sys.stdout.flush()
        except BaseException:
            return _send(w, b"E")
        with open("out.txt", encoding="utf-8", errors="replace") as f:
            pred = f.read()
        if not stdio_match(pred, out):
            return _send(w, b"F")
        _send(w, b".")
    _send(w, b"P")


def _child(w, code, t, workdir):
    try:
        os.setsid()
        os.chdir(workdir)
        lim = cpu_limit(len(t["outputs"]))
        resource.setrlimit(resource.RLIMIT_CPU, (lim, lim + 1))
        vm = _vmsize()
        if vm:
            resource.setrlimit(resource.RLIMIT_AS, (vm + MEM, vm + MEM))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE, FSIZE))
        null = os.open(os.devnull, os.O_RDWR)
        os.dup2(null, 2)
        sys.stderr = open(os.devnull, "w")
        if t["testtype"] == "functional":
            os.dup2(null, 0)
            os.dup2(null, 1)
            sys.stdin, sys.stdout = open(os.devnull), open(os.devnull, "w")
            _functional(w, code, t)
        else:
            _stdio(w, code, t)
    except BaseException:
        _send(w, b"E")
    finally:
        os._exit(0)


def run_one(code, t, workdir, walls=(PUBLIC_WALL, PRIVATE_WALL)):
    """-> (status, tests passed). status: pass | wrong | error | compile | timeout | crash."""
    n, npub = len(t["outputs"]), t["n_public"]
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        _child(w, code, t, workdir)
    os.close(w)
    start, done, status = time.monotonic(), 0, None
    total_wall = 2 * cpu_limit(n) + 5
    while status is None:
        left = total_wall - (time.monotonic() - start)
        tmo = min(walls[0] if done < npub else walls[1], left)
        if tmo <= 0 or not select.select([r], [], [], tmo)[0]:
            status = "timeout"
            break
        chunk = os.read(r, 4096)
        if not chunk:
            status = "crash"
            break
        for c in chunk:
            if c == 46:                                   # "."
                done += 1
            else:
                status = STATUS.get(c, "error")
                break
    for kill in (os.killpg, os.kill):                     # the group, and the child itself if setsid never ran
        try:
            kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _, ws = os.waitpid(pid, 0)
    os.close(r)
    if status == "crash" and os.WIFSIGNALED(ws) and os.WTERMSIG(ws) in (signal.SIGXCPU, signal.SIGKILL):
        status = "timeout"                                # RLIMIT_CPU: soft limit sends SIGXCPU, hard SIGKILL
    return status, done


# ---------------------------------------------------------------- worker (spawned process)

def load_tests(lcb_dir, qid):
    t = json.loads((Path(lcb_dir) / "tests" / f"{safe_id(qid)}.json").read_text())
    if t["testtype"] == "functional":                     # parsed once per problem, inherited by every child
        t["args"] = [[json.loads(l) for l in x.split("\n")] for x in t["inputs"]]
        t["expected"] = [json.loads(x) for x in t["outputs"]]
    return t


def _worker(jobs, results, lcb_dir, walls):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.nice(10)                                           # vLLM's scheduler shares the 4 vCPUs; limits are CPU time
    cache, workdir = OrderedDict(), tempfile.mkdtemp(prefix="lcbx_")
    while True:
        job = jobs.get()
        if job is None:
            break
        gid, qid, programs, stop_first = job
        t0 = time.monotonic()
        try:
            if qid not in cache:
                cache[qid] = load_tests(lcb_dir, qid)
                while len(cache) > CACHE:
                    cache.popitem(last=False)
            cache.move_to_end(qid)
            t, out = cache[qid], []
            for code in programs:
                out.append(run_one(code, t, workdir, walls))
                if stop_first and out[-1][0] == "pass":
                    break
            results.put((gid, out, time.monotonic() - t0, None))
        except Exception as e:                            # a grader bug must not look like a failed program
            results.put((gid, [], time.monotonic() - t0, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- the pool, used by the main process

def _h(qid, code):
    return sha1((qid + "\0" + code).encode("utf-8", "replace")).hexdigest()


class Grader:
    """submit(qid, programs) -> gid; a group stops at its first passing program when stop_first.
    Identical programs for the same problem are graded once (the main process keeps every known status)."""

    def __init__(self, lcb_dir, workers=3, trusted=False, walls=(PUBLIC_WALL, PRIVATE_WALL)):
        if not trusted and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise SystemExit("refusing to execute generated code outside a Kaggle session")
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(k, "1")
        ctx = mp.get_context("spawn")
        self.jobs, self.res = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_worker, args=(self.jobs, self.res, str(lcb_dir), walls), daemon=True)
                      for _ in range(workers)]
        for p in self.procs:
            p.start()
        self.groups, self.done, self.known = {}, {}, {}
        self.next_gid, self.seconds, self.counts, self.errors = 0, 0.0, Counter(), []

    def submit(self, qid, programs, stop_first=True):
        gid, self.next_gid = self.next_gid, self.next_gid + 1
        order, seen, todo = [], set(), []
        for code in programs:
            h = _h(qid, code)
            order.append(h)
            if h not in seen and h not in self.known:
                seen.add(h)
                todo.append(code)
        self.groups[gid] = {"qid": qid, "order": order, "todo": [_h(qid, c) for c in todo], "stop_first": stop_first}
        if stop_first and any(self.known.get(h) == "pass" for h in order):
            todo = []                                     # already solved by a program graded before
        if todo:
            self.jobs.put((gid, qid, todo, stop_first))
        else:
            self._finish(gid)
        return gid

    def _finish(self, gid):
        g = self.groups.pop(gid)
        sts = [self.known.get(h) for h in g["order"]]
        first = next((i for i, s in enumerate(sts) if s == "pass"), -1)
        self.done[gid] = {"passed": first >= 0, "first_pass": first, "n_programs": len(sts),
                          "n_graded": sum(s is not None for s in sts), "statuses": Counter(s for s in sts if s)}

    def poll(self, block=False, timeout=None):
        got = 0
        while True:
            try:
                gid, out, secs, err = self.res.get(block=block and got == 0, timeout=timeout)
            except Exception:                             # queue.Empty
                return got
            got += 1
            self.seconds += secs
            g = self.groups[gid]
            if err:
                self.errors.append((g["qid"], err))
            for h, (st, _) in zip(g["todo"], out):
                self.known[h] = st
                self.counts[st] += 1
            self._finish(gid)

    def pending(self):
        return len(self.groups)

    def wait(self, deadline=None):
        while self.groups:
            if deadline is not None and time.time() > deadline:
                return False
            self.poll(block=True, timeout=5)
        return True

    def close(self):
        for _ in self.procs:
            self.jobs.put(None)
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


def grade_now(grader, qid, programs, stop_first=False):
    """Blocking helper for tests: -> (result dict, per-program statuses in order)."""
    gid = grader.submit(qid, programs, stop_first)
    grader.wait()
    return grader.done[gid], [grader.known.get(_h(qid, c)) for c in programs]
