"""Kaggle job (2 x T4, internet on): Stage A, Qwen2.5-7B-Instruct, plain vs DecompRL inference vs the idea. No training.

Inputs (private to the account):
  * dataset        animeshratna/qwen7b-decomp-code -> this folder's *.py
  * notebook output animeshratna/qwen7b-lcb-prep    -> lcb/ (problems.jsonl, tests/), from kaggle_prep_job.py
  * later sessions only: the previous session's notebook output (stage_a/state.json, raw_*.jsonl.gz, logs). Its
    presence makes this a resumed session; one whose rounds are all done exits before installing anything.
Steps: a Python 3.11 venv made with uv (Kaggle's Python is 3.13; vLLM 0.8.5 needs < 3.13), the pinned vLLM in it
(0.8.5.post1: its V0 engine runs on Turing; V1 needs compute capability 8.0), download Qwen/Qwen2.5-7B-Instruct from the Hub (15.2 GB, fp16 on T4s, tensor parallel 2), run the CPU tests, then
run_stage_a.py with hours counted from this script's start. Outputs in /kaggle/working/stage_a.
Generated programs run only in lcb_exec's sandboxed children. A run still going HARD_HOURS after the start is killed
(the quota is shared) and the report is rebuilt from state.json, which holds the chunks that finished. Kaggle stops
a GPU session at 12 h, so one session is at most 10.7 h, and all sessions together stay within PLAN_HOURS.
The job exits 0 whenever a state.json exists, so the next session can take this output as its input; the outcome
is in stage_a/job_status.json and the log.
Stage B collection runs the same job with RUNNER=collect.py, DATA=cc_train (qwen7b-stage-b-prep's output),
OUT_NAME=stage_b_collect and EXTRA="--arm_hours 1.5", set at the top of that notebook's job.py.
Stage B dev evaluation: DATA=lcb_dev; a trained arm also sets ADAPTER=lora_<arm> (in qwen7b-stage-b-train's output,
merged into fp16 weights before vLLM starts, so it runs as fast as the base) and EXTRA="--arms <arm>".
"""
import glob, json, os, shutil, signal, subprocess, sys, threading, time

T0 = time.time()
PLAN_HOURS = float(os.environ.get("PLAN_HOURS", 21.0))     # the whole Stage A run, every session (13.5 for sessions 1-2)
GEN_HOURS, TOTAL_HOURS = float(os.environ.get("GEN_HOURS", 10.0)), float(os.environ.get("TOTAL_HOURS", 10.6))
VLLM, MODEL_ID, MODEL = "vllm==0.8.5.post1", "Qwen/Qwen2.5-7B-Instruct", "/tmp/qwen7b"
TRANSFORMERS = "transformers==4.51.3"   # vLLM 0.8.5 sets no upper bound; 5.x tokenizers lack all_special_tokens_extended
SETUPTOOLS = "setuptools==80.9.0"       # Triton imports it; vLLM lists it only above 3.11 and uv venvs come without it
RUNNER, DATA, OUT_NAME = (os.environ.get(k, v) for k, v in (("RUNNER", "run_stage_a.py"), ("DATA", "lcb"),
                                                             ("OUT_NAME", "stage_a")))
EXTRA, ADAPTER = os.environ.get("EXTRA", ""), os.environ.get("ADAPTER", "")
PEFT = "'peft==0.15.2' 'accelerate==1.6.0'" if ADAPTER else ""   # train_lora.py --merge; the training job's pins
CODE, OUT, VENV = "/kaggle/working/code", f"/kaggle/working/{OUT_NAME}", "/tmp/venv"
PY = f"{VENV}/bin/python"
LIMIT = int(os.environ.get("LIMIT", 0))                   # 0 = all problems; a pilot takes the first LIMIT of the shuffle
CHUNK = int(os.environ.get("CHUNK", 48))                  # problems per chunk-round; smaller cuts time finer
for d in (CODE, OUT):
    os.makedirs(d, exist_ok=True)
prev = glob.glob(f"/kaggle/input/**/{OUT_NAME}/state.json", recursive=True)
if len(prev) > 1:
    sys.exit(f"want at most one earlier session's state.json, found {prev}")
RESUME = bool(prev)
if RESUME:
    for f in glob.glob(f"{os.path.dirname(prev[0])}/*"):
        if os.path.basename(f) in ("state.json", "run.log", "job.log") or f.endswith(".jsonl.gz"):
            shutil.copy(f, OUT)
    st = json.load(open(f"{OUT}/state.json"))
    used = sum(s["hours"] + 0.1 for s in st["sessions"])  # + container start and teardown
    print(f"resuming from {prev[0]}: {len(st['sessions'])} session(s), {used:.2f} h used of {PLAN_HOURS}", flush=True)
    if st.get("all_rounds_done"):
        print("all rounds were done in an earlier session; nothing to run", flush=True)
        sys.exit(0)
    TOTAL_HOURS = min(TOTAL_HOURS, PLAN_HOURS - used - 0.1)
    GEN_HOURS = min(GEN_HOURS, TOTAL_HOURS - 0.6)
    if GEN_HOURS < 0.75:
        print(f"{PLAN_HOURS - used:.2f} h left of the plan: too little for setup and a chunk-round", flush=True)
        sys.exit(0)
HARD_HOURS = TOTAL_HOURS + 0.1
print(f"GEN_HOURS {GEN_HOURS:.2f}, TOTAL_HOURS {TOTAL_HOURS:.2f}, hard stop {HARD_HOURS:.2f} h", flush=True)
src = [p for p in glob.glob("/kaggle/input/**/run_stage_a.py", recursive=True) if "qwen7b-decomp-code" in p]
if len(src) != 1:                                         # the prep notebook's output also holds an old code/ copy
    sys.exit(f"want one run_stage_a.py in the code dataset, found {src}")
src = os.path.dirname(src[0])
for f in glob.glob(f"{src}/*.py"):
    shutil.copy(f, CODE)
print("code from", src, flush=True)
hits = sorted(glob.glob(f"/kaggle/input/**/{DATA}/problems.jsonl", recursive=True))
if not hits:
    sys.exit(f"missing input: {DATA}/problems.jsonl")
LCB = os.path.dirname(hits[-1])
log = open(f"{OUT}/job.log", "a")


def run(cmd, env=None, deadline=None):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    kill = threading.Timer(max(deadline - time.time(), 0), lambda: os.killpg(p.pid, signal.SIGKILL)) if deadline else None
    if kill:
        kill.daemon = True
        kill.start()
    for line in p.stdout:
        if "it/s]" in line or "Warning" in line:
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    code = p.wait()
    if kill:
        kill.cancel()
    print(f"[exit {code}, {(time.time() - T0) / 60:.1f} min]", flush=True)
    return code


run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g; df -h /tmp | tail -1")
if run(f"pip install -q uv && uv venv {VENV} --python 3.11 && uv pip install -q --python {PY} '{VLLM}' '{TRANSFORMERS}' '{SETUPTOOLS}' {PEFT}") != 0:
    sys.exit("vllm install failed")
if run(f"{PY} -c \"from huggingface_hub import snapshot_download; "
       f"snapshot_download('{MODEL_ID}', local_dir='{MODEL}', allow_patterns=['*.json', '*.safetensors', '*.txt'])\" "
       f"&& sync && du -sh {MODEL}") != 0:             # sync: 15 GB of dirty pages stall the grader's file writes
    sys.exit("model download failed")
if ADAPTER:
    hits = glob.glob(f"/kaggle/input/**/{ADAPTER}/adapter_config.json", recursive=True)
    if len(hits) != 1:
        sys.exit(f"want one {ADAPTER}/adapter_config.json in the inputs, found {hits}")
    if run(f"{PY} train_lora.py --merge --model {MODEL} --adapter {os.path.dirname(hits[0])} --out {MODEL}_merged "
           f"&& rm -rf {MODEL} && mv {MODEL}_merged {MODEL} && sync && du -sh {MODEL}") != 0:
        sys.exit("merge failed")
if run(f"{PY} -c 'import llm; llm.preflight(\"{MODEL}\")'") != 0:
    sys.exit("preflight failed: the environment can't start vLLM")
tests = f"{PY} -c \"import tests_qwen as t; t.test_lcb_data(); t.test_decomp(); t.test_grader()\""
if run(tests) != 0 and run(tests) != 0:                   # a second try for timing flakes; a real bug fails both
    sys.exit("tests failed twice")
env = {"VLLM_USE_V1": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
code = run(f"{PY} {RUNNER} --model {MODEL} --lcb {LCB} --out {OUT} --t0 {T0} --limit {LIMIT} "
           f"--chunk {CHUNK} --gen_hours {GEN_HOURS} --total_hours {TOTAL_HOURS} {EXTRA}" + (" --resume" if RESUME else ""),
           env, deadline=T0 + HARD_HOURS * 3600)
if code != 0 and os.path.exists(f"{OUT}/state.json") and RUNNER == "run_stage_a.py":
    run(f"{PY} -c \"import json, pathlib, report; from lcb_data import load; "
        f"r = report.write(json.load(open('{OUT}/state.json')), load('{LCB}'), pathlib.Path('{OUT}')); "
        f"print(report.markdown(r))\"")
done = os.path.exists(f"{OUT}/state.json") and json.load(open(f"{OUT}/state.json")).get("all_rounds_done", False)
json.dump({"run_stage_a_exit": code, "resumed": RESUME, "hours": (time.time() - T0) / 3600, "all_rounds_done": done},
          open(f"{OUT}/job_status.json", "w"))
if not os.path.exists(f"{OUT}/state.json"):
    sys.exit(f"run_stage_a failed ({code}) before saving any state")
print("DONE" if code == 0 else f"STOPPED: run_stage_a exit {code}; state.json holds the chunk-rounds that finished",
      "(all rounds done)" if done else "(more rounds left: resume in a new session)", flush=True)
