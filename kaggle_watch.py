"""Runs on the laptop, never on Kaggle: pushes the Stage A full run and walks it through its sessions.

Session 1 (notebook qwen7b-stage-a, staged in ~/slm-runs/qwen7b/kaggle_stage_a_kernel) is pushed once the weekly
GPU quota has reset (2026-11-14 00:00 UTC) and at least S1_QUOTA hours are left. Each later session (qwen7b-stage-a-s2,
then -s3, each taking the one before's output as input and resuming) is pushed only when all of these hold:
  * the session before was seen running, then showed the same final status on two checks at least 10 min apart;
  * the output files of that run (version created at or after the push) include state.json (the CLI lists
    base names only; the job writes it nowhere but stage_a/);
  * its stage_a/job_status.json says rounds are left, and run_stage_a ended by itself (exit 0) or by the job's
    hard stop (-9); only job_status.json and report.md/json are downloaded, each time into a new folder;
  * the quota still holds what the plan leaves (PLAN_HOURS - the earlier sessions' hours) plus 0.5 h.
Session 3 was added on 2026-11-18, after session 1, to finish the 40k and 50k rounds (PLAN_HOURS 13.5 -> 21).
Anything else stops with NEEDS USER. Steps are kept in ~/slm-runs/qwen7b/full_run/state.json, so the script can be
stopped and started again; a lock keeps two copies from acting at once. Log: full_run/watch.log.

  --watch    loop until a session starts running, all is done, or something needs the user
  --once     as far as it can get now (a scheduled fallback); waits 10 min when a final status needs a second look
  --dry-run  check the gates without pushing or saving; --ignore-time checks them before the reset as well
Exit: 0 done, 1 nothing to do yet, 2 needs the user, 3 past DEADLINE, 4 another copy holds the lock,
10 a session has started running.
"""
import argparse, fcntl, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.expanduser("~/slm-runs/qwen7b")
RUN = f"{ROOT}/full_run"
OWNER = "animeshratna"
S1 = {"key": "s1", "kernel": f"{OWNER}/qwen7b-stage-a", "dir": f"{ROOT}/kaggle_stage_a_kernel", "max_hours": 13.0}
S2 = {"key": "s2", "kernel": f"{OWNER}/qwen7b-stage-a-s2", "dir": f"{ROOT}/kaggle_stage_a_s2_kernel", "max_hours": 5.0}
S3 = {"key": "s3", "kernel": f"{OWNER}/qwen7b-stage-a-s3", "dir": f"{ROOT}/kaggle_stage_a_s3_kernel", "max_hours": 10.0}
SESSIONS = [S1, S2, S3]
RESET = datetime(2026, 11, 14, tzinfo=timezone.utc).timestamp()
DEADLINE = datetime(2026, 11, 21, tzinfo=timezone.utc).timestamp()   # the next quota refresh
PLAN_HOURS = 21.0          # kaggle_stage_a_job.py's cap over all sessions
S1_QUOTA = 11.5            # session 1 hard-stops 10.7 h after its start, plus container start and teardown
CONFIRM = 600              # seconds between the two looks at a final status
FINAL, LIVE = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"), ("QUEUED", "RUNNING")
REPORTS = r"^(stage_a/)?(job_status\.json|report\.(md|json))$"
EXIT = {"done": 0, "dry": 0, "user": 2, "milestone": 10}
DRY = False


class Transient(Exception):
    pass


def kaggle(*args, timeout=300):
    exe = shutil.which("kaggle") or "/opt/anaconda3/bin/kaggle"
    p = subprocess.run([exe, *args], cwd=os.path.expanduser("~"), capture_output=True, text=True, timeout=timeout)
    return p.returncode, "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "Warning" not in l)


def status(kernel):
    code, out = kaggle("kernels", "status", kernel)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    if m:
        return m.group(1)
    if "Cannot access kernel" in out:
        return "MISSING"
    raise Transient(f"status {kernel}: {out[-300:]}")


def gpu_left():
    code, out = kaggle("quota")
    for line in out.splitlines():
        f = line.split()
        if len(f) == 5 and f[0] == "GPU":
            return float(f[2].rstrip("h")), f[4]
    raise Transient(f"quota: {out[-300:]}")


def files(kernel):
    """{base name: creation time of the run's version} for the output files of the kernel's latest run."""
    names, token = {}, None
    for _ in range(200):
        code, out = kaggle("kernels", "files", kernel, "--page-size", "200", *(["--page-token", token] if token else []))
        token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token = "):
                token = line.split("= ", 1)[1].strip()
                continue
            f = re.split(r"\s{2,}", line.strip())
            if len(f) == 3 and f[0] != "name" and not f[0].startswith("-"):
                when = datetime.strptime(f[2], "%I:%M %p, %A %d %B %Y UTC").replace(tzinfo=timezone.utc)
                names[f[0]] = when.timestamp()
        if code != 0:
            raise Transient(f"files {kernel}: {out[-300:]}")
        if not token:
            return names
    raise Transient(f"files {kernel}: more than 200 pages")


def download(kernel, key):
    d = f"{RUN}/{key}_out_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    os.makedirs(d)
    code, out = kaggle("kernels", "output", kernel, "-p", d, "--file-pattern", REPORTS, "-q", timeout=900)
    if code != 0:
        raise Transient(f"output {kernel}: {out[-300:]}")
    return d


def job_status(d):
    try:
        js = json.load(open(glob.glob(f"{d}/**/job_status.json", recursive=True)[0]))
    except (OSError, ValueError, IndexError):
        return None
    ok = isinstance(js, dict) and isinstance(js.get("hours"), (int, float)) and isinstance(js.get("all_rounds_done"), bool)
    return js if ok else None


def load():
    try:
        return json.load(open(f"{RUN}/state.json"))
    except FileNotFoundError:
        return {}


def save(st):
    if DRY:
        return
    json.dump(st, open(f"{RUN}/state.json.tmp", "w"), indent=1)
    os.replace(f"{RUN}/state.json.tmp", f"{RUN}/state.json")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}"
    print(line, flush=True)
    if not DRY:
        with open(f"{RUN}/watch.log", "a") as f:
            f.write(line + "\n")


def push(st, sess, now, note):
    if DRY:
        return "dry", f"would push {sess['kernel']} from {sess['dir']} ({note})"
    st[sess["key"]] = {"pushing_at": now, "note": note}  # a crash mid-push must not lead to a second push
    save(st)
    code, out = kaggle("kernels", "push", "-p", sess["dir"], timeout=600)
    m = re.search(r"Kernel version (\d+) successfully pushed", out)
    if not m:
        return "user", f"push of {sess['kernel']} gave no version (exit {code}): {out[-400:]}"
    st[sess["key"]].update(pushed_at=now, version=int(m.group(1)))
    save(st)
    return "pushed", f"pushed {sess['kernel']} v{m.group(1)} ({note})"


def track(st, sess, now):
    """Moves a pushed session along: seen running, final on two looks, output checked. None once checked."""
    s, kernel = st[sess["key"]], sess["kernel"]
    if s.get("checked"):
        return None
    if "pushed_at" not in s:
        return "user", f"a push of {kernel} began at {fmt(s['pushing_at'])} and was never confirmed"
    age = (now - s["pushed_at"]) / 3600
    k = status(kernel)
    if not s.get("running_at"):
        ran = k in FINAL and any(t >= s["pushed_at"] - 300 for t in files(kernel).values())
        if k in LIVE or ran:
            s["running_at"] = now
            save(st)
            return "milestone", f"{kernel} v{s['version']} is {k} ({age * 60:.0f} min after the push)"
        if age > 0.5:
            return "user", f"{kernel} not seen running 30 min after the push (status {k})"
        return "wait", f"{kernel} status {k}, waiting for the new version to start"
    if k not in FINAL:
        if s.pop("final", None):
            save(st)
        if age > sess["max_hours"]:
            return "user", f"{kernel} still {k} {age:.1f} h after the push"
        return "wait", f"{kernel} {k}, {age:.1f} h after the push"
    first = s.get("final")
    if not first or first[0] != k:
        s["final"] = [k, now]
        save(st)
        return "look", f"{kernel} shows {k}; second look in {CONFIRM // 60} min"
    if now - first[1] < CONFIRM:
        return "look", f"{kernel} shows {k}; second look due {fmt(first[1] + CONFIRM)}"
    mine = {n for n, t in files(kernel).items() if t >= s["pushed_at"] - 300}
    if "state.json" not in mine:
        return "user", f"{kernel} ended {k}, but this run's output has no state.json ({len(mine)} files)"
    d = download(kernel, sess["key"])
    s.update(checked=now, status=k, out_dir=d, job_status=job_status(d))
    save(st)
    return "progress", f"{kernel} ended {k} (confirmed); job_status {s['job_status']}; reports in {d}"


def step(st, now, ignore_time=False):
    if "s1" not in st:
        if now < RESET and not ignore_time:
            return "wait", f"waiting for the quota reset at {fmt(RESET)}"
        left, refresh = gpu_left()
        if left < S1_QUOTA:
            return "wait", f"{left:.2f} GPU-h left (refresh {refresh}); session 1 needs {S1_QUOTA}"
        k = status(S1["kernel"])
        if k not in FINAL:
            return "user", f"{S1['kernel']} is {k} before the push"
        return push(st, S1, now, f"{left:.2f} GPU-h left; ntl-poollora {status(f'{OWNER}/ntl-poollora')}")
    for n, (sess, nxt) in enumerate(zip(SESSIONS, SESSIONS[1:] + [None]), 1):
        r = track(st, sess, now)
        if r:
            return r
        js, out = st[sess["key"]]["job_status"], st[sess["key"]]["out_dir"]
        if js is None:
            return "user", f"session {n}'s job_status.json is missing or unreadable in {out}"
        if js["all_rounds_done"]:
            return "done", f"all rounds done in session {n}; reports in {out}"
        if nxt is None:
            return "done", f"all {n} sessions finished with rounds left (the plan's hours are used); reports in {out}"
        if nxt["key"] in st:
            continue
        if js.get("run_stage_a_exit") not in (0, -9):
            return "user", f"session {n}'s run_stage_a exited {js.get('run_stage_a_exit')}; not resuming blindly"
        left, refresh = gpu_left()
        used = sum(st[s["key"]]["job_status"]["hours"] + 0.1 for s in SESSIONS[:n])
        if left < PLAN_HOURS - used + 0.5:
            return "user", f"{left:.2f} GPU-h left, session {n + 1} may take {PLAN_HOURS - used:.2f} h plus margin"
        k = status(nxt["kernel"])
        if k != "MISSING" and k not in FINAL:
            return "user", f"{nxt['kernel']} is {k} before its push"
        return push(st, nxt, now, f"{left:.2f} GPU-h left; sessions 1-{n} took {used:.2f} h")


def fmt(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def pause(kind, st, now):
    if kind == "look":
        return CONFIRM + 5
    if kind == "pushed" or any(st.get(k, {}).get("pushed_at", 0) > now - 1800 and not st[k].get("running_at")
                               for k in [s["key"] for s in SESSIONS]):
        return 120
    if "s1" not in st and now < RESET:
        return max(30, min(600, RESET + 60 - now))
    return 600


def main():
    global DRY
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ignore-time", action="store_true")
    a = ap.parse_args()
    DRY = a.dry_run
    os.makedirs(RUN, exist_ok=True)
    lock = open(f"{RUN}/lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another copy holds the lock (the watcher is running); nothing to do", flush=True)
        return 4
    st, errors, last = load(), 0, ("", 0.0)
    for i in range(10 ** 6):
        now = time.time()
        if now > DEADLINE:
            log(f"NEEDS USER: past the deadline {fmt(DEADLINE)} with state {sorted(st)}")
            return 3
        try:
            kind, msg = step(st, now, a.ignore_time)
            errors = 0
        except (Transient, subprocess.TimeoutExpired, OSError) as e:
            errors += 1
            kind, msg = ("user", f"{errors} failed Kaggle calls in a row: {e}") if errors > 6 else ("wait", f"failed: {e}")
        if kind != "wait" or re.sub(r"[\d.]+", "#", msg) != last[0] or now - last[1] > 3600:
            log(("NEEDS USER: " if kind == "user" else f"{kind}: ") + msg)
            if kind == "wait":
                last = (re.sub(r"[\d.]+", "#", msg), now)
        if kind in EXIT:
            return EXIT[kind]
        if kind == "wait" and not a.watch:
            return 1
        if kind != "progress":
            if a.once and i > 8:
                return 1
            time.sleep(pause(kind, st, time.time()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
