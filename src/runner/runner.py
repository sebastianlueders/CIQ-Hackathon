"""Experiment runner: trial config -> Fuzzball workflow -> metrics in results/.

Every experiment the agent runs goes through here, so every experiment is a
Fuzzball workflow on Fuzzball volumes backed by VAST.

    from runner import Trial, run_all
    results = run_all([Trial("a0", {"lr": 2e-4}), Trial("a1", {"lr": 5e-4})], concurrency=2)

CLI:  python src/runner/runner.py trial.json [trial2.json ...] --concurrency 2
      (each JSON file: {"id": "...", "config": {...}})
"""
import argparse, json, os, subprocess, sys, tempfile, time
from dataclasses import dataclass, field

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FB = os.path.join(REPO, "fuzzball")
RESULTS = os.path.join(REPO, "results", "trials")
TERMINAL = {"Finished", "Errored", "Canceled", "Failed"}

# Execution environment for training trials: NGC PyTorch (Triton works; the slim image lacks a C
# compiler) + venv on the envs volume with transformers/peft. Chosen from the stage-0 probes.
ENV = {
    "image": os.environ.get("TRIAL_IMAGE", "docker://nvcr.io/nvidia/pytorch:26.10-py3"),
    "python": os.environ.get("TRIAL_PYTHON", "/envs/train-pt2610/bin/python"),
    "cores": int(os.environ.get("TRIAL_CORES", 16)), "memory": os.environ.get("TRIAL_MEMORY", "80GiB"),  # measured peak 64.8 GiB incl. GPU
}


@dataclass
class Trial:
    id: str
    config: dict
    script: str = "src/train/train_lora.py"
    timeout: str = "3h"
    nodes: int = 1          # >1: Fuzzball multinode job, data parallel (one GB10 GPU per node)
    workflow_id: str = ""
    status: str = ""
    records: list = field(default_factory=list)


def fb(*args, check=True):
    r = subprocess.run([FB, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"fuzzball {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def render(t: Trial) -> dict:
    """Fuzzfile for one trial. The trainer source and the config are inlined, so the
    workflow alone records exactly what ran."""
    cfg = {"out": f"/runs/trials/{t.id}", **t.config}
    code = open(os.path.join(REPO, t.script)).read()
    return {
        "version": "v4",
        "files": {"train.py": code, "trial.json": json.dumps(cfg, indent=1),
                  "fused_ce.py": open(os.path.join(REPO, "src/kernels/fused_ce.py")).read()},
        "jobs": {
            "trial": {
                "image": {"uri": ENV["image"]},
                "env": ["HOME=/tmp/home", "HF_HUB_OFFLINE=1", "PYTHONUNBUFFERED=1"],
                "files": {"/opt/train.py": "file://train.py", "/opt/trial.json": "file://trial.json",
                          "/opt/fused_ce.py": "file://fused_ce.py"},
                "mounts": {"/models": {"volume": "models"}, "/data": {"volume": "data"},
                           "/runs": {"volume": "runs"}, "/envs": {"volume": "envs"}},
                "policy": {"timeout": {"execute": t.timeout},
                           # GPU nodes power down 7 PM-9 AM: re-run a job whose node disappears.
                           **({"restart": "on-node-fault"} if os.environ.get("TRIAL_RETRY") else {})},
                "resource": {"cpu": {"cores": ENV["cores"]}, "memory": {"size": ENV["memory"]},
                             "devices": {"nvidia.com/gpu": 1}},
                "script": launch_script(t),
            }
        },
        "volumes": {v: {"name": v, "use": "team3"} for v in ("models", "data", "runs", "envs")},
    } if t.nodes == 1 else _multinode(t)


def _multinode(t: Trial) -> dict:
    t2 = Trial(t.id, t.config, t.script, t.timeout, nodes=1)
    wf = render(t2)
    job = wf["jobs"]["trial"]
    job["multinode"] = {"nodes": t.nodes, "implementation": "generic"}
    job["script"] = launch_script(t)
    wf["files"]["multinode_launch.sh"] = open(os.path.join(REPO, "src/runner/multinode_launch.sh")).read()
    job["files"]["/opt/multinode_launch.sh"] = "file://multinode_launch.sh"
    job["env"] = job["env"] + ["NCCL_DEBUG=INFO"]
    return wf


def launch_script(t: Trial) -> str:
    py = ENV["python"]
    if t.nodes == 1:
        return f"#!/bin/bash\nmkdir -p /tmp/home\nexec {py} /opt/train.py --config /opt/trial.json\n"
    # Same script on every node (src/runner/multinode_launch.sh): diagnostics, TCP check, then torchrun.
    variant = t.config.get("dp_launch", "ip")
    timeout = 1800 if t.config.get("skip_eval") else 0  # perf probes only; real training runs have no timeout
    return (f"#!/bin/bash\nexport MN_NODES={t.nodes} MN_VARIANT={variant} MN_TIMEOUT={timeout} PY={py} "
            f"TRAIN_ARGS='/opt/train.py --config /opt/trial.json'\nexec bash /opt/multinode_launch.sh\n")


def launch(t: Trial):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{t.id}.fuzzfile.yaml")
    json.dump(render(t), open(path, "w"), indent=1)  # JSON is valid YAML
    fb("workflow", "validate", path)
    out = fb("workflow", "start", path, "--name", t.id)
    t.workflow_id = out.strip().splitlines()[-1].strip()
    t.status = "Started"
    print(f"[runner] launched {t.id} -> {t.workflow_id}", flush=True)


def status(wid: str) -> str:
    for line in fb("workflow", "list", check=False).splitlines():
        if line.startswith(wid):
            return line.split("|")[3].strip()
    return "Unknown"


def collect(t: Trial):
    log = fb("workflow", "log", t.workflow_id, "trial", check=False)
    t.records = [json.loads(l.split("RESULT ", 1)[1]) for l in log.splitlines() if "RESULT {" in l]
    with open(os.path.join(RESULTS, f"{t.id}.jsonl"), "w") as f:
        for r in t.records:
            f.write(json.dumps({**r, "trial": t.id, "workflow": t.workflow_id}) + "\n")
    if t.status != "Finished":
        open(os.path.join(RESULTS, f"{t.id}.log"), "w").write(log)


def run_all(trials, concurrency=2, poll=20):
    """Run trials with at most `concurrency` in flight (one per GB10 node)."""
    pending, active, done = list(trials), [], []
    while pending or active:
        while pending and len(active) < concurrency:
            t = pending.pop(0); launch(t); active.append(t)
        time.sleep(poll)
        for t in list(active):
            t.status = status(t.workflow_id)
            if t.status in TERMINAL:
                collect(t); active.remove(t); done.append(t)
                ev = [r for r in t.records if r.get("kind") == "eval"]
                print(f"[runner] {t.id} {t.status}: evals={[(r['tokens'], r['eval_loss']) for r in ev]}", flush=True)
    return done


def final_eval(t: Trial):
    ev = [r for r in t.records if r.get("kind") == "eval"]
    return ev[-1]["eval_loss"] if ev else float("inf")


def submit_only(trials, manifest):
    """Submit every trial now (no local driver needed) and record them for a later collect."""
    m = json.load(open(manifest)) if os.path.exists(manifest) else []
    for t in trials:
        launch(t); m.append({"id": t.id, "workflow": t.workflow_id, "job": "trial", "kind": "trial"})
        json.dump(m, open(manifest, "w"), indent=1)


def collect_manifest(manifest):
    """Pull status + RESULT records of every workflow in a manifest into results/trials/."""
    for e in json.load(open(manifest)):
        t = Trial(e["id"], {}); t.workflow_id = e["workflow"]; t.status = status(e["workflow"])
        log = fb("workflow", "log", e["workflow"], e.get("job", "trial"), check=False)
        os.makedirs(RESULTS, exist_ok=True)
        import re
        log = re.sub(r"(FB_TOKEN|TOKEN|token)=\S+", r"\1=<redacted>", log)
        open(os.path.join(RESULTS, f"{e['id']}.log"), "w").write(log)
        t.records = [json.loads(l.split("RESULT ", 1)[1]) for l in log.splitlines() if "RESULT {" in l]
        with open(os.path.join(RESULTS, f"{e['id']}.jsonl"), "w") as f:
            for r in t.records: f.write(json.dumps({**r, "trial": t.id, "workflow": t.workflow_id}) + "\n")
        print(f"{e['id']:40s} {t.status:10s} records={len(t.records)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="*"); ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--submit-only", metavar="MANIFEST"); ap.add_argument("--collect", metavar="MANIFEST")
    a = ap.parse_args()
    if a.collect:
        collect_manifest(a.collect); sys.exit(0)
    ts = [Trial(d["id"], d.get("config", {}), nodes=d.get("nodes", 1)) for d in (json.load(open(p)) for p in a.trials)]
    if a.submit_only:
        submit_only(ts, a.submit_only); sys.exit(0)
    for t in run_all(ts, a.concurrency):
        print(json.dumps({"id": t.id, "status": t.status, "final_eval_loss": final_eval(t)}))
