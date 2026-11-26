"""Kaggle job (CPU only, internet on): LiveCodeBench data for Stage A, and this code's tests in Kaggle's environment.

Inputs: dataset animeshratna/qwen7b-decomp-code (this folder's *.py).
  1. download livecodebench/code_generation_lite test.jsonl..test5.jsonl (release_v5, about 4.3 GB) from the Hub;
  2. keep DecompRL's window, 2024-08-01..2025-02-01 (the paper reports 279 problems): /kaggle/working/lcb/;
  3. run tests_qwen.py here (Linux, Kaggle's Python): the grader runs only the hand-written programs in that file;
  4. rehearse the GPU job's install: a Python 3.11 venv made with uv (Kaggle's Python is 3.13, which vLLM 0.8.5
     doesn't support; 3.11 is also what DecompRL's sandbox ran), the pinned vLLM in it, and the tests again under it.
Uses no GPU quota. The Stage A notebook reads /kaggle/working/lcb from this notebook's output.
"""
import glob, json, os, shutil, subprocess, sys, time

CODE, RAW, OUT, VENV = "/kaggle/working/code", "/tmp/lcb_raw", "/kaggle/working/lcb", "/tmp/venv"
VLLM = "vllm==0.8.5.post1"
TRANSFORMERS = "transformers==4.51.3"   # vLLM 0.8.5 sets no upper bound; 5.x tokenizers lack all_special_tokens_extended
SETUPTOOLS = "setuptools==80.9.0"       # Triton imports it; vLLM lists it only above 3.11 and uv venvs come without it
FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl")   # all of release_v5
t0 = time.time()
for d in (CODE, RAW):
    os.makedirs(d, exist_ok=True)
src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/lcb_data.py", recursive=True))[-1])
for f in glob.glob(f"{src}/*.py"):
    shutil.copy(f, CODE)


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=CODE)
    print(f"[exit {r.returncode}, {time.time() - t0:.0f}s]", flush=True)
    return r.returncode


run(f"{sys.executable} --version; nproc; free -g; df -h /tmp /kaggle/working | tail -2")
from huggingface_hub import hf_hub_download  # noqa: E402
paths = []
for f in FILES:
    paths.append(hf_hub_download(repo_id="livecodebench/code_generation_lite", repo_type="dataset", filename=f,
                                 local_dir=RAW))
    print(f, os.path.getsize(paths[-1]) // 2**20, "MiB", flush=True)
if run(f"{sys.executable} lcb_data.py --raw {' '.join(paths)} --out {OUT}") != 0:
    sys.exit("lcb_data failed")
stats = json.load(open(f"{OUT}/stats.json"))
print("problems in window:", stats["problems"], "(paper: 279)", flush=True)
if run(f"{sys.executable} tests_qwen.py") != 0:
    sys.exit("tests failed")
run(f"du -sh {OUT}")
if run(f"pip install -q uv && uv venv {VENV} --python 3.11 && uv pip install -q --python {VENV}/bin/python '{VLLM}' '{TRANSFORMERS}' '{SETUPTOOLS}'") != 0:
    sys.exit("venv / vllm install failed")
run(f"{VENV}/bin/python -c 'import sys, vllm, torch, xformers; "
    f"print(sys.version.split()[0], vllm.__version__, torch.__version__, torch.version.cuda, xformers.__version__)'")
if run(f"{VENV}/bin/python tests_qwen.py") != 0:
    sys.exit("tests failed under the venv")
print("DONE", flush=True)
