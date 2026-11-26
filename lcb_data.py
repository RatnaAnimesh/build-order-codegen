"""LiveCodeBench problems for the DecompRL comparison: the paper's window, 2024-08-01 to 2025-02-01 (279 problems).

    python lcb_data.py --raw test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl --out lcb/

Reads livecodebench/code_generation_lite release files and writes the files below. Releases are snapshots, so a
file can hold problems dated inside a later window; every file of a release is read and stats.json counts each
problem's source. Measured on Kaggle (2026-11-10): all of release_v5 gives exactly the paper's 279 (test3 12,
test4 100, test5 167); test4+test5 alone give 267, and adding test6 (release_v6) gives 317.
  * lcb/problems.jsonl   one row per problem: id, title, platform, date, difficulty, content, starter_code, fn_name,
                         public tests (shown in no prompt; LCB's prompts don't include them separately)
  * lcb/tests/<id>.json  {"fn_name", "testtype", "inputs", "outputs", "n_public"}: public tests first, then private,
                         the order lcb_runner grades them in
  * lcb/stats.json
The window filter is lcb_runner's: start <= contest_date <= end, with dates parsed at midnight.

Private tests are base64(zlib(pickle(json string))). They are unpickled with a loader that refuses every global, so the
pickle can only produce plain values (it must be a str), never call anything.
"""
import argparse, base64, io, json, pickle, re, zlib
from datetime import datetime
from pathlib import Path

START, END = "2024-08-01", "2025-02-01"


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused")


def decode_private(s):
    try:
        return json.loads(s)
    except Exception:
        raw = _PlainUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(s.encode("utf-8"))))).load()
        if not isinstance(raw, str):
            raise ValueError(f"private tests unpickled to {type(raw).__name__}, expected str")
        return json.loads(raw)


def safe_id(qid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", qid)


def in_window(date, start=START, end=END):
    d = datetime.fromisoformat(date)
    return datetime.strptime(start, "%Y-%m-%d") <= d <= datetime.strptime(end, "%Y-%m-%d")


def convert(row):
    """One raw LCB row -> (problem dict, tests dict)."""
    public = json.loads(row["public_test_cases"])
    private = decode_private(row["private_test_cases"])
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    tests = public + private
    types = {t["testtype"] for t in tests}
    if len(types) != 1:
        raise ValueError(f"{row['question_id']}: mixed test types {types}")
    p = {"id": row["question_id"], "title": row["question_title"], "platform": row["platform"], "source": None,
         "date": row["contest_date"], "difficulty": row["difficulty"], "content": row["question_content"],
         "starter_code": row["starter_code"] or "", "fn_name": meta.get("func_name"), "testtype": types.pop(),
         "n_public": len(public), "n_tests": len(tests)}
    t = {"fn_name": p["fn_name"], "testtype": p["testtype"], "inputs": [x["input"] for x in tests],
         "outputs": [x["output"] for x in tests], "n_public": len(public)}
    if p["testtype"] == "functional" and not p["fn_name"]:
        raise ValueError(f"{p['id']}: functional problem without func_name")
    return p, t


def build(raw_files, out, start=START, end=END):
    out = Path(out)
    (out / "tests").mkdir(parents=True, exist_ok=True)
    probs, seen, skipped = [], set(), []
    for f in raw_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["question_id"] in seen or not in_window(row["contest_date"], start, end):
                    continue
                seen.add(row["question_id"])
                try:
                    p, t = convert(row)
                    p["source"] = Path(f).name
                except Exception as e:                    # report, never guess
                    skipped.append(f"{row['question_id']}: {type(e).__name__}: {e}"[:300])
                    continue
                (out / "tests" / f"{safe_id(p['id'])}.json").write_text(json.dumps(t))
                probs.append(p)
    probs.sort(key=lambda p: (p["date"], p["id"]))
    with open(out / "problems.jsonl", "w") as fh:
        for p in probs:
            fh.write(json.dumps(p) + "\n")
    stats = {"window": [start, end], "problems": len(probs), "skipped": skipped,
             "by_platform": _count(p["platform"] for p in probs), "by_source": _count(p["source"] for p in probs), "by_difficulty": _count(p["difficulty"] for p in probs),
             "by_testtype": _count(p["testtype"] for p in probs),
             "tests_per_problem": _quantiles([p["n_tests"] for p in probs]),
             "test_bytes": _quantiles([(out / "tests" / f"{safe_id(p['id'])}.json").stat().st_size for p in probs])}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return probs, stats


def load(lcb_dir):
    return [json.loads(l) for l in open(Path(lcb_dir) / "problems.jsonl")]


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"min": xs[0], "p50": q(0.5), "p90": q(0.9), "max": xs[-1], "sum": sum(xs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()
    _, s = build(a.raw, a.out, a.start, a.end)
    print(json.dumps({k: v for k, v in s.items() if k != "skipped"}, indent=1), "\nskipped:", len(s["skipped"]))
    for x in s["skipped"][:20]:
        print("  ", x)
