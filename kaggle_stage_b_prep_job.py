"""Kaggle job (CPU only, internet on): Stage B's data. Uses no GPU quota.

Inputs: dataset animeshratna/qwen7b-decomp-code (this folder's *.py); notebook output animeshratna/qwen7b-lcb-prep
(lcb/problems.jsonl: the 279, which the dev set must not contain).
  1. tests_qwen.py here: the grader runs only the hand-written programs in that file;
  2. dev set: LiveCodeBench release_v6 (test6.jsonl), problems dated from 2025-02-02 -> /kaggle/working/lcb_dev/;
  3. training pool: CodeContests train, filtered and checked with reference solutions (cc_data.py)
     -> /kaggle/working/cc_train/.
"""
import glob, json, os, shutil, subprocess, sys, time

CODE, RAW, DEV, CC = "/kaggle/working/code", "/tmp/raw", "/kaggle/working/lcb_dev", "/kaggle/working/cc_train"
t0 = time.time()
for d in (CODE, RAW):
    os.makedirs(d, exist_ok=True)
src = [p for p in glob.glob("/kaggle/input/**/cc_data.py", recursive=True) if "qwen7b-decomp-code" in p]
if len(src) != 1:
    sys.exit(f"want one cc_data.py in the code dataset, found {src}")
for f in glob.glob(f"{os.path.dirname(src[0])}/*.py"):
    shutil.copy(f, CODE)
test_ids = {json.loads(l)["id"] for f in glob.glob("/kaggle/input/**/lcb/problems.jsonl", recursive=True) for l in open(f)}
print("the 279:", len(test_ids), flush=True)
if len(test_ids) != 279:
    sys.exit("missing input: the 279 from qwen7b-lcb-prep")


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=CODE)
    print(f"[exit {r.returncode}, {time.time() - t0:.0f}s]", flush=True)
    return r.returncode


run(f"{sys.executable} --version; nproc; free -g; df -h /tmp /kaggle/working | tail -2")
if run(f"{sys.executable} tests_qwen.py") != 0:
    sys.exit("tests failed")
from huggingface_hub import hf_hub_download  # noqa: E402
raw6 = hf_hub_download(repo_id="livecodebench/code_generation_lite", repo_type="dataset", filename="test6.jsonl",
                       local_dir=RAW)
if run(f"{sys.executable} lcb_data.py --raw {raw6} --out {DEV} --start 2025-02-02 --end 2100-01-01") != 0:
    sys.exit("lcb_data failed")
dev_ids = {json.loads(l)["id"] for l in open(f"{DEV}/problems.jsonl")}
print("dev problems:", len(dev_ids), "shared with the 279:", sorted(dev_ids & test_ids), flush=True)
if dev_ids & test_ids:
    sys.exit("the dev set overlaps the 279")
if run(f"{sys.executable} cc_data.py --hf {RAW}/cc --out {CC} --n 1500") != 0:
    sys.exit("cc_data failed")
run(f"du -sh {DEV} {CC}")
print("DONE", flush=True)
