"""CodeContests training problems for Stage B, in lcb_data's format (problems.jsonl + tests/<id>.json).

    python cc_data.py --raw train-*.parquet --out cc_train/ [--n 1500]
    python cc_data.py --hf /tmp/cc --out cc_train/     # the 39 train files from the Hub, one at a time (7.2 GB)

Keeps Codeforces problems rated 800-1600 with an untranslated statement, stdin/stdout (no input or output file) and
a Python 3 solution. Tests: public, private, then generated, the first 30 under 1 MB each. A problem stays only if
its first Python 3 solution passes our grader on those tests, so the tests and the grader agree; then a seeded --n
of them. The solutions are human-written ones from the dataset, and like everything the grader runs they run only
inside a Kaggle session. CodeContests ends in 2021, so no problem here can be a LiveCodeBench one.
"""
import argparse, json, os, random
from collections import Counter
from pathlib import Path

from lcb_data import safe_id
from lcb_exec import Grader

COLS = ["name", "description", "public_tests", "private_tests", "generated_tests", "source", "solutions",
        "cf_contest_id", "cf_index", "cf_rating", "is_description_translated", "input_file", "output_file"]
CODEFORCES, PYTHON3, MAX_TESTS, MAX_BYTES = 2, 3, 30, 1 << 20


def convert(row):
    """One CodeContests row -> (problem, tests, reference solution), or None when it doesn't qualify."""
    if (row["source"] != CODEFORCES or not 800 <= (row["cf_rating"] or 0) <= 1600 or row["is_description_translated"]
            or row["input_file"] or row["output_file"]):
        return None
    sols = row["solutions"]
    sol = next((s for lang, s in zip(sols["language"], sols["solution"]) if lang == PYTHON3), None)
    tests, n_pub = [], 0
    for k in ("public_tests", "private_tests", "generated_tests"):
        for i, o in zip(row[k]["input"], row[k]["output"]):
            if len(tests) < MAX_TESTS and len(i) + len(o) < MAX_BYTES:
                tests.append((i, o))
                n_pub += k == "public_tests"
    if sol is None or not tests:
        return None
    p = {"id": f"cf{row['cf_contest_id']}{row['cf_index']}", "title": row["name"], "platform": "codeforces",
         "source": "code_contests", "date": "", "difficulty": "easy" if row["cf_rating"] <= 1200 else "medium",
         "rating": row["cf_rating"], "content": row["description"], "starter_code": "", "fn_name": None,
         "testtype": "stdin", "n_public": n_pub, "n_tests": len(tests)}
    t = {"fn_name": None, "testtype": "stdin", "inputs": [i for i, _ in tests], "outputs": [o for _, o in tests],
         "n_public": n_pub}
    return p, t, sol


def build(raw_files, out, n=1500, workers=3, seed=0, trusted=False):
    """raw_files: parquet paths, read one at a time (an iterator may download each and delete it after)."""
    import pyarrow.parquet as pq
    out = Path(out)
    (out / "tests").mkdir(parents=True, exist_ok=True)
    cand = {}                                             # id -> (problem, reference solution); tests go to disk
    for f in raw_files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=COLS):
            for row in batch.to_pylist():
                r = convert(row)
                if r and r[0]["id"] not in cand:
                    (out / "tests" / f"{safe_id(r[0]['id'])}.json").write_text(json.dumps(r[1]))
                    cand[r[0]["id"]] = (r[0], r[2])
    g = Grader(out, workers=workers, trusted=trusted)
    gids = {g.submit(q, [sol]): q for q, (_, sol) in cand.items()}
    g.wait()
    g.close()
    ok = sorted(gids[gid] for gid, r in g.done.items() if r["passed"])
    keep = set(random.Random(seed).sample(ok, min(n, len(ok))))
    for q in cand:
        if q not in keep:
            (out / "tests" / f"{safe_id(q)}.json").unlink()
    with open(out / "problems.jsonl", "w") as fh:
        for q in sorted(keep):
            fh.write(json.dumps(cand[q][0]) + "\n")
    stats = {"candidates": len(cand), "reference_passed": len(ok), "kept": len(keep),
             "reference_statuses": dict(sum((g.done[gid]["statuses"] for gid in gids), Counter())),
             "by_rating": dict(Counter(cand[q][0]["rating"] for q in keep)), "grader_errors": g.errors[:20]}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def hub_files(tmp, repo="deepmind/code_contests"):
    """Download the train files one at a time, deleting each once read."""
    from huggingface_hub import HfApi, hf_hub_download
    for f in sorted(x for x in HfApi().list_repo_files(repo, repo_type="dataset") if x.startswith("data/train-")):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=tmp)
        print("read", f, flush=True)
        yield p
        os.remove(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", nargs="+")
    src.add_argument("--hf", help="download folder for the Hub's train files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    print(json.dumps(build(a.raw or hub_files(a.hf), a.out, a.n, a.workers), indent=1))
