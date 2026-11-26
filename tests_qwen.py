"""Tests for the Stage A code. Every program run here is hand-written in this file (trusted), never model output.

    python tests_qwen.py            # all
    python -c "import tests_qwen as t; t.test_grader()"
"""
import base64, json, os, pickle, re, subprocess, sys, tempfile, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decomp as D  # noqa: E402
import lcb_data  # noqa: E402
from lcb_exec import Grader, grade_now  # noqa: E402

PROBLEMS = [
    {"id": "sumto", "title": "Sum to N", "platform": "atcoder", "date": "2024-09-01T00:00:00", "difficulty": "easy",
     "content": "Given N, print 1+2+...+N.", "starter_code": "", "fn_name": None, "testtype": "stdin",
     "tests": [("3\n", "6\n"), ("1\n", "1\n"), ("10\n", "55\n"), ("100000\n", "5000050000\n")], "n_public": 1},
    {"id": "addall", "title": "Add all", "platform": "leetcode", "date": "2024-10-01T00:00:00", "difficulty": "easy",
     "content": "Return the sum of nums.", "starter_code": "class Solution:\n    def addAll(self, nums: List[int]) -> int:\n        ",
     "fn_name": "addAll", "testtype": "functional", "tests": [("[1, 2, 3]", "6"), ("[]", "0"), ("[5]", "5")], "n_public": 1},
    {"id": "half", "title": "Half", "platform": "atcoder", "date": "2024-11-01T00:00:00", "difficulty": "medium",
     "content": "Print N/2.", "starter_code": "", "fn_name": None, "testtype": "stdin",
     "tests": [("3\n", "1.5\n"), ("4\n", "2\n")], "n_public": 1},
    {"id": "pairup", "title": "Pair up", "platform": "leetcode", "date": "2024-12-01T00:00:00", "difficulty": "hard",
     "content": "Return nums unchanged.", "starter_code": "class Solution:\n    def pairUp(self, nums: List[int]) -> List[int]:\n        ",
     "fn_name": "pairUp", "testtype": "functional", "tests": [("[1, 2]", "[1, 2]"), ("[3]", "[3]")], "n_public": 1},
]


def make_lcb(d):
    d = Path(d)
    (d / "tests").mkdir(parents=True, exist_ok=True)
    with open(d / "problems.jsonl", "w") as f:
        for p in PROBLEMS:
            row = {k: v for k, v in p.items() if k != "tests"}
            row["n_tests"] = len(p["tests"])
            f.write(json.dumps(row) + "\n")
            (d / "tests" / f"{p['id']}.json").write_text(json.dumps({
                "fn_name": p["fn_name"], "testtype": p["testtype"], "inputs": [i for i, _ in p["tests"]],
                "outputs": [o for _, o in p["tests"]], "n_public": p["n_public"]}))
    return d


def test_lcb_data():
    tests = [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
    enc = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert lcb_data.decode_private(enc) == tests
    assert lcb_data.decode_private(json.dumps(tests)) == tests

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    bad = base64.b64encode(zlib.compress(pickle.dumps(Evil()))).decode()
    try:
        lcb_data.decode_private(bad)
        raise AssertionError("evil pickle accepted")
    except pickle.UnpicklingError:
        pass
    row = {"question_id": "abc1_a", "question_title": "t", "platform": "atcoder", "contest_date": "2024-08-01T00:00:00",
           "difficulty": "easy", "question_content": "c", "starter_code": "", "metadata": "{}",
           "public_test_cases": json.dumps([{"input": "0\n", "output": "1\n", "testtype": "stdin"}]),
           "private_test_cases": enc}
    with tempfile.TemporaryDirectory() as t:
        raw = Path(t) / "raw.jsonl"
        rows = [row, {**row, "question_id": "late", "contest_date": "2025-02-02T00:00:00"},
                {**row, "question_id": "early", "contest_date": "2024-07-31T23:00:00"}, row]
        raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        probs, stats = lcb_data.build([raw], Path(t) / "lcb")
        assert [p["id"] for p in probs] == ["abc1_a"] and stats["problems"] == 1
        t2 = json.loads((Path(t) / "lcb/tests/abc1_a.json").read_text())
        assert t2["inputs"] == ["0\n", "1\n"] and t2["n_public"] == 1
    print("test_lcb_data ok")


STDIN_PROGS = {
    "n=int(input())\nprint(n*(n+1)//2)": "pass",
    "import sys\nn=int(sys.stdin.buffer.read())\nprint(n*(n+1)//2)": "pass",
    "n=int(open(0).read())\nprint(n*(n+1)//2)": "pass",
    "import sys\nn=int(input())\nprint(n*(n+1)//2)\nsys.exit(0)\nprint('extra')": "pass",
    "import threading\ndef main():\n    n=int(input())\n    print(n*(n+1)//2)\nthreading.Thread(target=main).start()": "pass",
    "n = int(input())\nprint(sum(accumulate([2]*n)))": "wrong",              # prelude names, wrong answer
    "n = int(input())\nprint(n*(n+1)//2 if n < 50 else 0)": "wrong",           # passes public, fails private
    "print(1/0)": "error",
    "def f(:\n  pass": "compile",
    "while True:\n    pass": "timeout",
    "import os\nos._exit(0)": "crash",
}


def test_grader():
    with tempfile.TemporaryDirectory() as t:
        lcb = make_lcb(Path(t) / "lcb")
        g = Grader(lcb, workers=2, trusted=True, walls=(1.0, 2.0))
        try:
            progs = list(STDIN_PROGS)
            r, sts = grade_now(g, "sumto", progs)
            assert sts == list(STDIN_PROGS.values()), list(zip(sts, STDIN_PROGS.values()))
            assert r["passed"] and r["first_pass"] == 0 and r["n_graded"] == len(progs)
            r, sts = grade_now(g, "half", ["n=int(input())\nprint(f'{n/2:.2f}')", "print(int(input())//2)"])
            assert sts == ["pass", "wrong"], sts                      # 1.50 == 1.5 as Decimals; 2.00 == 2
            fun = {"class Solution:\n    def addAll(self, nums):\n        return sum(nums)": "pass",
                   "class Solution:\n    def addAll(self, nums):\n        return 0": "wrong",
                   "class Solution:\n    def addAll(self, nums):\n        raise SystemExit": "error",
                   "class Solution:\n    def nope(self, nums):\n        return 0": "compile"}
            r, sts = grade_now(g, "addall", list(fun))
            assert sts == list(fun.values()), sts
            r, sts = grade_now(g, "pairup", ["class Solution:\n    def pairUp(self, nums):\n        return tuple(nums)"])
            assert sts == ["pass"], sts                               # tuples count as lists
            n0 = sum(g.counts.values())
            gid = g.submit("sumto", ["print(1/0)", progs[0], progs[1]], stop_first=True)
            g.wait()
            assert g.done[gid]["passed"] and sum(g.counts.values()) == n0  # all known: nothing re-run
        finally:
            g.close()
    print("test_grader ok")


DECOMP_STDIN = '''Let me think.

```python
def read_input() -> int:
    """
    Reads N from stdin.
    Uses: -
    """
    pass

def total(n: int) -> int:
    """
    Returns 1 + 2 + ... + n.
    Uses: -
    """
    pass

def main() -> None:
    """
    Reads N and prints the total.
    Uses: read_input, total
    """
    pass

if __name__ == "__main__":
    main()
```'''

DECOMP_FUN = '''```python
def add_values(nums: list[int]) -> int:
    """Adds the numbers.
    Uses: -"""
    pass

def addAll(nums: list[int]) -> int:
    """The LeetCode entry.
    Uses: add_values"""
    pass
```'''

DECOMP_CLASS = '''```python
import math

def helper(x: int) -> int:
    """Doubles x."""
    pass

class Solution:
    def addAll(self, nums: list[int]) -> int:
        """Sum of doubled values.
        Uses: helper"""
        pass
```'''


def _code(src):
    return f"Here it is.\n```python\n{src}\n```\n"


_calls = {}


def scripted_reply(messages, j):
    """j is the sample index within one request; the idea arm sends k requests of one sample each, so it also
    gets a per-function call count."""
    text = messages[-1]["content"]
    m = re.search(r"Please help me implement the function (\w+)", text)
    if m and "MISSING:" in text and "(shown above)" not in text:
        j = _calls[m.group(1)] = _calls.get(m.group(1), -1) + 1
    if "### Question:" in text:
        if "addAll" in text:
            return _code("class Solution:\n    def addAll(self, nums):\n        return " + ("sum(nums)" if j % 2 == 0 else "0"))
        if "pairUp" in text:
            return _code("class Solution:\n    def pairUp(self, nums):\n        return list(nums)")
        if "N/2" in text:
            return _code("print(int(input()) / 2)")
        return _code("n = int(input())\nprint(" + ("n*(n+1)//2" if j % 2 == 0 else "n") + ")")
    if "break them down into components" in text:
        if "addAll" in text:
            return DECOMP_FUN
        if "pairUp" in text:
            return "I can't decompose this."                        # no code block: a failed attempt
        if "N/2" in text:
            return "```python\n" + "\n\n".join(f"def f{i}():\n    pass" for i in range(8)) + "\n```"   # > nmax
        return DECOMP_STDIN
    m = re.search(r"Please help me implement the function (\w+)", text)
    if m:
        name, idea = m.group(1), "MISSING:" in text
        if name == "read_input":
            return _code("def read_input() -> int:\n    return int(input())")
        if name == "total":
            return _code("def total(n):\n    return " + ("n * (n + 1) // 2" if j % 2 == 0 else "n * n"))
        if name == "main":
            if "fmt (shown above)" in text:
                return _code("def main():\n    print(fmt(total(read_input())))")
            if idea and j == 1:
                return "MISSING: def fmt(x: int) -> str: formats the answer"
            if idea and j == 2:
                return _code("def main():\n    show(total(read_input()))")
            return _code("def main():\n    print(total(read_input()))")
        if name == "add_values":
            return _code("def add_values(nums):\n    return " + ("sum(nums)" if j % 2 == 0 else "-1"))
        if name == "addAll":
            return _code("def addAll(nums):\n    return add_values(nums)")
    m = re.search(r"Please write the function `(\w+)`", text)
    if m:
        return _code({"fmt": "def fmt(x):\n    return str(x)", "show": "def show(x):\n    print(x)"}.get(m.group(1), "x = 1"))
    return "no idea"


def test_decomp():
    d = D.parse_decomposition(DECOMP_STDIN)
    assert [u.name for u in d.units] == ["read_input", "total", "main"]
    assert d.unit("main").uses_raw == ["read_input", "total"] and d.unit("total").uses_raw == []
    assert D.dependency_levels(d) == [["read_input", "total"], ["main"]] and D.roots(d) == ["main"]
    shown = d.render()
    assert "Uses: read_input, total" in shown and "__main__" not in shown and shown.count("pass") == 3
    fn = {u.fname for u in d.units}
    ims = {u.name: D.parse_impl(scripted_reply([{"content": f"Please help me implement the function {u.fname}"}], 0), u, fn)
           for u in d.units}
    prog = D.assemble(d, ims, "stdin", None)
    assert prog.count("main()") == 2 and "return n * (n + 1) // 2" in prog    # its own entry kept, no extra call
    noentry = D.parse_decomposition(DECOMP_STDIN.replace('if __name__ == "__main__":\n    main()\n', ""))
    assert D.assemble(noentry, ims, "stdin", None).rstrip().endswith("main()")
    im = D.parse_impl(_code("from math import gcd\nLIMIT = 10\ndef extra(a):\n    return a\n"
                            "def main():\n    show(gcd(4, 6), defaultdict(int), extra(1), LIMIT)"), d.unit("main"), fn)
    assert im.imports == ["from math import gcd"] and len(im.helpers) == 2
    assert D.undefined_calls(im, d) == ["show"]
    assert D.parse_missing("MISSING: def fmt(x: int) -> str: formats") == [("fmt", "def fmt(x: int) -> str: formats")]
    assert D.parse_missing(_code("x = 1") + "MISSING: def g(): no") == []
    assert D.parse_helper(_code("import math\ndef fmt(x):\n    return str(x)"), "fmt") == (["import math"],
                                                                                         ["def fmt(x):\n    return str(x)"])
    f = D.parse_decomposition(DECOMP_FUN)
    p = D.assemble(f, {u.name: D.parse_impl(_code(f"def {u.fname}(nums):\n    return sum(nums)"), u, {"add_values", "addAll"})
                       for u in f.units}, "functional", "addAll")
    assert "class Solution:" in p and "return addAll(*args, **kwargs)" in p
    c = D.parse_decomposition(DECOMP_CLASS)
    assert [u.name for u in c.units] == ["helper", "Solution.addAll"] and "class Solution:" in c.render()
    assert D.dependency_levels(c) == [["helper"], ["Solution.addAll"]] and D.roots(c) == ["Solution.addAll"]
    im = D.parse_impl(_code("class Solution:\n    def addAll(self, nums):\n        return sum(helper(x) for x in nums)"),
                      c.unit("Solution.addAll"), {"helper", "addAll"})
    prog = D.assemble(c, {"Solution.addAll": im, "helper": None}, "functional", "addAll")
    assert "    def addAll(self, nums):\n        return sum(" in prog and prog.startswith("import math")
    compile(prog, "p", "exec")
    import random
    impls = {"read_input": [1, 1], "total": [1, None, 1], "main": [None, None]}
    cs = D.combos(d, impls, 10, random.Random(0))
    assert len(cs) == 4 and all(x["main"] is None and x["total"] in (0, 2) for x in cs)
    print("test_decomp ok")


def test_report():
    import itertools
    import report
    for n, c, k in ((5, 2, 3), (4, 1, 2), (6, 0, 2), (3, 3, 1), (7, 2, 4)):
        subs = list(itertools.combinations(range(n), k))          # attempts 0..c-1 passed
        assert abs(report.pass_at_k(n, c, k) - sum(min(s) < c for s in subs) / len(subs)) < 1e-12
    assert report.pass_at_k(2, 1, 4) == 1.0 and report.pass_at_k(2, 0, 4) == 0.0
    assert [report.attempts_at(b, 4000) for b in (1000, 5000, 10000, 50000)] == [1, 1, 2, 12]
    att = lambda tok, res: [{"tokens": tok, "passed": x} for x in res]
    state = {"attempts": {"plain": {"a": att(100, [1] + [0] * 9), "b": att(100, [0] * 10)},
                          "decomp": {"a": att(1000, [0, 1]), "b": att(1000, [0])},
                          "idea": {"a": att(500, [0] * 4), "b": att(500, [0, 0, 1, 0])}}}
    r = report.build(state, [{"id": "a", "difficulty": "easy"}, {"id": "b", "difficulty": "hard"}],
                     budgets=[500, 1000, 2000])
    assert r["attempts_at"] == {"plain": [5, 10, 20], "decomp": [1, 1, 2], "idea": [1, 2, 4]}
    assert r["pass_at_tokens"]["plain"] == [0.25, 0.5, 0.5] and r["coverage"]["plain"] == [1.0, 1.0, 0.0]
    assert r["pass_at_tokens"]["decomp"] == [0.25, 0.25, 0.5] and r["coverage"]["decomp"] == [1.0, 1.0, 0.5]
    assert r["pass_at_tokens"]["idea"] == [0.125, 0.25, 0.5] and r["coverage"]["idea"] == [1.0, 1.0, 1.0]
    assert r["largest_covered_budget"] == r["largest_covered_all_arms"] == 1000
    x = r["primary_idea_minus_decomp"]
    assert x["budget"] == 1000 and x["mean"] == 0.0 and x["lo"] <= 0 <= x["hi"]
    assert r["check_exact_tokens"]["coverage"]["decomp"] == [1.0, 1.0, 0.5]
    report.markdown(r)
    print("test_report ok")


def _stage_a(lcb, out, *extra):
    r = subprocess.run([sys.executable, str(HERE / "run_stage_a.py"), "--lcb", str(lcb), "--out", str(out),
                        "--scripted", "tests_qwen:scripted_reply", "--targets", "200,400,800", "--plain_cap", "100",
                        "--chunk", "3", "--workers", "2", "--evals", "16", "--budgets", "50,100,200,400,800", *extra],
                       cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, json.loads((out / "state.json").read_text())


def test_end_to_end():
    """Two sessions on the scripted model: the first stops after three chunk-rounds, the second resumes."""
    with tempfile.TemporaryDirectory() as t:
        lcb = make_lcb(Path(t) / "lcb")
        out = Path(t) / "out"
        _, s1 = _stage_a(lcb, out, "--max_chunk_rounds", "3")
        assert len(s1["chunks_done"]) == 3 and len(s1["sessions"]) == 1 and not s1["all_rounds_done"]
        log, st = _stage_a(lcb, out, "--resume")
        assert "resumed" in log and "all rounds done" in log and st["all_rounds_done"], log[-3000:]
        assert [s["session"] for s in st["sessions"]] == [0, 1] and all(c[0] == 1 for c in st["chunks_done"][3:])
        count = lambda s, arm, sess=None: sum(sess is None or a["session"] == sess
                                              for v in s["attempts"][arm].values() for a in v)
        for arm in ("plain", "decomp", "idea"):            # session 0's attempts were all graded, so all kept
            assert count(st, arm, 0) == count(s1, arm) and count(st, arm, 1) > 0, arm
        rep = json.loads((out / "report.json").read_text())
        assert rep["problems_paired"] == 4 and rep["sessions"] == 2
        for arm, top in (("decomp", 800), ("idea", 800), ("plain", 100)):   # what each arm's last budget buys
            i = rep["budgets"].index(top)
            assert rep["coverage"][arm][i] == 1.0, (arm, rep["coverage"][arm], rep["tokens_per_attempt"])
        assert rep["largest_covered_budget"] == 800 and rep["largest_covered_all_arms"] == 100
        dec = [a for v in st["attempts"]["decomp"].values() for a in v]
        reasons = {a["reason"] for a in dec}
        assert reasons == {None, "no_decomposition", "too_many_units"}, reasons
        s = rep["arms"]["idea"]
        assert s["missing_explicit"] > 0 and s["missing_checker"] > 0 and s["helpers_built"] > 0 and s["resumed"] > 0
        sumto = st["attempts"]["idea"]["sumto"]
        assert len(sumto) > 1 and all(a["passed"] for a in sumto) and sumto[0]["coherent"] > 0
        assert all(a["passed"] for a in st["attempts"]["decomp"]["sumto"])
        assert not any(a["passed"] for a in st["attempts"]["decomp"]["pairup"])
        assert st["attempts"]["plain"]["pairup"] and all(a["passed"] for a in st["attempts"]["plain"]["pairup"])
        assert "DONE" in log and (out / "report.md").exists()
        log, one = _stage_a(lcb, Path(t) / "idea", "--arms", "idea")      # Stage B dev: a trained arm runs alone
        assert "all rounds done" in log and not any(one["attempts"][a] for a in ("plain", "decomp"))
        rep = json.loads((Path(t) / "idea" / "report.json").read_text())
        assert list(rep["arms"]) == ["idea"] and rep["problems_paired"] == 4 and not rep["harness_check"]
        assert "primary_idea_minus_decomp" not in rep and rep["coverage"]["idea"][rep["budgets"].index(800)] == 1.0
        import dev_report
        r = dev_report.build(st, [one], budgets=[50, 100, 200, 400, 800], seconds=[1e-9])
        assert r["problems"] == 4 and list(r["tokens"]["series"]) == ["plain", "decomp", "idea", "idea+"]
        assert "idea+ - idea" in r["tokens"]["compare"] and "idea+ - decomp+" not in r["tokens"]["compare"]
        dev_report.markdown(r)
    print("test_end_to_end ok")


def test_dev_report():
    import dev_report
    att = lambda tok, res: [{"tokens": tok, "passed": x} for x in res]
    base = {"attempts": {"plain": {"a": att(100, [1, 0]), "b": att(100, [0, 0])},
                         "decomp": {"a": att(1000, [0]), "b": att(1000, [1])},
                         "idea": {"a": att(600, [1]), "b": att(600, [0])}},
            "timing": {"plain": [4.0, 400], "decomp": [20.0, 2000], "idea": [10.0, 1200]}}
    idea = {"attempts": {"plain": {}, "decomp": {}, "idea": {"a": att(600, [1, 1]), "b": att(600, [1, 0])}},
            "timing": {"plain": [0.0, 0], "decomp": [0.0, 0], "idea": [20.0, 2400]}}
    r = dev_report.build(base, [idea])
    assert r["seconds_per_attempt"] == {"plain": 1.0, "decomp": 10.0, "idea": 5.0, "idea+": 5.0}
    for unit, b in (("tokens", 1000), ("gpu_seconds", 5)):           # both buy one attempt: 0.75 vs 0.5
        c = r[unit]["compare"]["idea+ - idea"]
        assert c["budget"] == b and c["mean"] == 0.25 and c["lo"] <= 0.25 <= c["hi"], (unit, c)
    assert r["tokens"]["series"]["idea+"][1]["coverage"] == 0.0 and "idea+ - plain+" not in r["tokens"]["compare"]
    dev_report.markdown(r)
    try:
        dev_report.build(base, [base])
        raise AssertionError("a three-arm run passed as a trained run")
    except AssertionError as e:
        assert "samples one arm" in str(e), e
    print("test_dev_report ok")


def test_cc_data():
    """CodeContests filters, the test cap and the reference-solution check, on a tiny hand-made parquet file."""
    import pyarrow as pa, pyarrow.parquet as pq
    import cc_data
    ok = "n = int(input())\nprint(n * (n + 1) // 2)"
    t = lambda xs: {"input": [f"{x}\n" for x in xs], "output": [f"{x * (x + 1) // 2}\n" for x in xs]}
    base = {"name": "Sum", "description": "Given N, print 1+2+...+N.", "public_tests": t([3]), "private_tests": t([1, 10]),
            "generated_tests": t(range(40)), "source": 2, "solutions": {"language": [1, 3], "solution": ["print 1", ok]},
            "cf_contest_id": 1, "cf_index": "A", "cf_rating": 900, "is_description_translated": False,
            "input_file": "", "output_file": ""}
    rows = [base, {**base, "cf_index": "B", "solutions": {"language": [3], "solution": ["print(0)"]}},    # wrong
            {**base, "cf_index": "C", "cf_rating": 2000}, {**base, "cf_index": "D", "is_description_translated": True},
            {**base, "cf_index": "E", "solutions": {"language": [1], "solution": ["print 1"]}},             # Python 2
            {**base, "cf_index": "F", "source": 1}, {**base, "cf_index": "G", "input_file": "in.txt"}]
    with tempfile.TemporaryDirectory() as d:
        pq.write_table(pa.Table.from_pylist(rows), f"{d}/a.parquet")
        s = cc_data.build([f"{d}/a.parquet"], f"{d}/cc", n=10, workers=2, trusted=True)
        assert (s["candidates"], s["reference_passed"], s["kept"]) == (2, 1, 1), s
        probs = lcb_data.load(f"{d}/cc")
        assert [p["id"] for p in probs] == ["cf1A"] and probs[0]["n_public"] == 1 and probs[0]["n_tests"] == 30
        assert sorted(os.listdir(f"{d}/cc/tests")) == ["cf1A.json"]
        tests = json.loads(Path(f"{d}/cc/tests/cf1A.json").read_text())
        assert tests["inputs"][:3] == ["3\n", "1\n", "10\n"] and len(tests["outputs"]) == 30
    print("test_cc_data ok")


def test_collect():
    """Equal-time collection on the scripted model; every decomp training record rebuilds into its passing program."""
    import gzip
    with tempfile.TemporaryDirectory() as t:
        lcb, out = make_lcb(Path(t) / "lcb"), Path(t) / "out"
        r = subprocess.run([sys.executable, str(HERE / "collect.py"), "--lcb", str(lcb), "--out", str(out),
                            "--scripted", "tests_qwen:scripted_reply", "--arm_hours", "1e-7", "--chunk", "6",
                            "--plain_per", "2", "--workers", "2", "--evals", "16"],
                           cwd=HERE, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "every arm has its hours" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
        st = json.loads((out / "state.json").read_text())
        for arm, per in (("plain", 2), ("decomp", 1), ("idea", 1)):       # chunk 6 over 4 problems wraps the pool
            n = sorted(len(v) for v in st["attempts"][arm].values())
            assert n == [per] * 2 + [2 * per] * 2, (arm, n)
        recs = {arm: [json.loads(l) for l in gzip.open(out / f"train_{arm}.jsonl.gz", "rt")] for arm in st["attempts"]}
        passed = {arm: sum(bool(a["passed"]) for v in st["attempts"][arm].values() for a in v) for arm in recs}
        assert all(len(recs[arm]) == passed[arm] > 0 for arm in recs), (passed, {a: len(r) for a, r in recs.items()})
        g = Grader(lcb, workers=2, trusted=True)
        for rec in recs["decomp"] + recs["idea"]:
            ex = rec["examples"]
            assert "break them down into components" in ex[0]["messages"][-1]["content"]
            d = D.parse_decomposition(ex[0]["reply"])
            fn = {u.fname for u in d.units}
            ims = {u.name: None for u in d.units}
            for e in ex[1:]:
                name = re.search(r"implement the function (\w+)", e["messages"][-1]["content"]).group(1)
                u = next(u for u in d.units if u.fname == name)
                ims[u.name] = D.parse_impl(e["reply"], u, fn)
            p = next(p for p in PROBLEMS if p["id"] == rec["id"])
            res, _ = grade_now(g, rec["id"], [D.assemble(d, ims, p["testtype"], p["fn_name"])])
            built = any("(shown above)" in e["messages"][-1]["content"] for e in ex)   # helper code isn't a reply
            assert res["passed"] or built, rec
        for rec in recs["plain"]:
            p = next(p for p in PROBLEMS if p["id"] == rec["id"])
            assert grade_now(g, rec["id"], [D.lcb_extract(rec["examples"][0]["reply"])])[0]["passed"], rec
        g.close()
    print("test_collect ok")


def test_train_data():
    """train_lora's example loading (4 attempts per problem, prompt as a prefix, too long skipped) and its budget."""
    import gzip
    import train_lora as T

    class Tok:                                            # one token per character, a Qwen-like template
        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
            s = "".join(f"<{m['role']}>{m['content']}<end>\n" for m in msgs) + ("<assistant>" if add_generation_prompt else "")
            return [ord(c) for c in s]

        def decode(self, ids):
            return "".join(map(chr, ids))
    with tempfile.TemporaryDirectory() as d:
        ex = lambda q, r: {"messages": [{"role": "user", "content": q}], "reply": r}
        recs = [{"id": "a", "examples": [ex("q", f"r{i}")]} for i in range(6)] + \
               [{"id": "b", "examples": [ex("dec", "plan"), ex("impl", "x" * 500)]}]
        with gzip.open(f"{d}/t.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        data, skipped, problems = T.examples(f"{d}/t.jsonl.gz", Tok(), max_len=100)
        assert (len(data), skipped, problems) == (5, 1, 2)          # a: first 4 only; b: the long one skipped
        assert all(Tok().decode(ids[n:]).startswith("r") or Tok().decode(ids[n:]).startswith("plan") for ids, n in data)
        order, used = T.schedule(data, budget=10 ** 9)
        assert sorted(order) == sorted(list(range(5)) * 2) and used == 2 * sum(len(x[0]) for x in data)
        order, used = T.schedule(data, budget=3 * len(data[0][0]))
        assert len(order) <= 3 and used <= 3 * len(data[0][0])
    print("test_train_data ok")


if __name__ == "__main__":
    test_lcb_data()
    test_decomp()
    test_grader()
    test_report()
    test_dev_report()
    test_end_to_end()
    test_cc_data()
    test_collect()
    test_train_data()
