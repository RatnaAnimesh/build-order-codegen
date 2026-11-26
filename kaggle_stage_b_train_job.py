"""Kaggle job (2 x T4, internet on): Stage B training, one LoRA per arm (train_lora.py). Nothing is sampled or run.

Inputs: dataset animeshratna/qwen7b-decomp-code (this folder's *.py); notebook output
animeshratna/qwen7b-stage-b-collect (stage_b_collect/train_<arm>.jsonl.gz). Two lanes, one per GPU: decomp then
plain on GPU 0, idea on GPU 1. Every arm stops at train_lora's --max_hours, so the job ends within about
2 x 1.8 h of training plus setup; HARD_HOURS kills anything still running.
Outputs: /kaggle/working/lora_<arm>/ (adapter, train_stats.json), train_<arm>.log, job_status.json.
"""
import glob, json, os, shutil, subprocess, sys, threading, time

T0 = time.time()
HARD_HOURS = float(os.environ.get("HARD_HOURS", 4.4))
MODEL_ID, MODEL, CODE, OUT, VENV = "Qwen/Qwen2.5-7B-Instruct", "/tmp/qwen7b", "/kaggle/working/code", "/kaggle/working", "/tmp/venv"
PY = f"{VENV}/bin/python"
PKGS = "'torch==2.6.0' 'transformers==4.51.3' 'peft==0.15.2' 'bitsandbytes==0.45.5' 'accelerate==1.6.0' 'setuptools==80.9.0'"
LANES = [("0", ["decomp", "plain"]), ("1", ["idea"])]
os.makedirs(CODE, exist_ok=True)
src = [p for p in glob.glob("/kaggle/input/**/train_lora.py", recursive=True) if "qwen7b-decomp-code" in p]
if len(src) != 1:
    sys.exit(f"want one train_lora.py in the code dataset, found {src}")
for f in glob.glob(f"{os.path.dirname(src[0])}/*.py"):
    shutil.copy(f, CODE)
data = {os.path.basename(p)[6:-9]: p for p in glob.glob("/kaggle/input/**/stage_b_collect/train_*.jsonl.gz", recursive=True)}
print("training data:", data, flush=True)
if sorted(data) != ["decomp", "idea", "plain"]:
    sys.exit("missing input: stage_b_collect/train_<arm>.jsonl.gz for all three arms")


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    code = subprocess.run(cmd, shell=True, cwd=CODE).returncode
    print(f"[exit {code}, {(time.time() - T0) / 60:.1f} min]", flush=True)
    return code


run("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h /tmp | tail -1")
if run(f"pip install -q uv && uv venv {VENV} --python 3.11 && uv pip install -q --python {PY} {PKGS}") != 0:
    sys.exit("install failed")
if run(f"{PY} -c \"from huggingface_hub import snapshot_download; "
       f"snapshot_download('{MODEL_ID}', local_dir='{MODEL}', allow_patterns=['*.json', '*.safetensors', '*.txt'])\"") != 0:
    sys.exit("model download failed")
codes = {}


def lane(gpu, arms):
    for arm in arms:
        with open(f"{OUT}/train_{arm}.log", "w") as log:
            p = subprocess.Popen(f"{PY} train_lora.py --model {MODEL} --data {data[arm]} --out {OUT}/lora_{arm}",
                                 shell=True, cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                codes[arm] = p.wait(timeout=max(T0 + HARD_HOURS * 3600 - time.time(), 0))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                codes[arm] = "killed at HARD_HOURS"
        print(f"{arm} on GPU {gpu}: exit {codes[arm]} at {(time.time() - T0) / 60:.1f} min", flush=True)


threads = [threading.Thread(target=lane, args=x) for x in LANES]
for t in threads:
    t.start()
for t in threads:
    t.join()
for arm in data:
    print(f"\n== {arm}\n" + "".join(open(f"{OUT}/train_{arm}.log").readlines()[-15:]), flush=True)
stats = {arm: json.load(open(f)) for arm in data if os.path.exists(f := f"{OUT}/lora_{arm}/train_stats.json")}
json.dump({"exit": codes, "hours": (time.time() - T0) / 3600, "trained": sorted(stats)}, open(f"{OUT}/job_status.json", "w"))
print("DONE" if len(stats) == 3 else f"STOPPED: adapters for {sorted(stats)} only", flush=True)
