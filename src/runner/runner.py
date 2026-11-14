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
    "cores": int(os.environ.get("TRIAL_CORES", 8)), "memory": os.environ.get("TRIAL_MEMORY", "80GiB"),  # measured peak 64.8 GiB incl. GPU
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
        "files": {"train.py": code, "trial.json": json.dumps(cfg, indent=1)},
        "jobs": {
            "trial": {
                "image": {"uri": ENV["image"]},
                "env": ["HOME=/tmp/home", "HF_HUB_OFFLINE=1", "PYTHONUNBUFFERED=1"],
                "files": {"/opt/train.py": "file://train.py", "/opt/trial.json": "file://trial.json"},
                "mounts": {"/models": {"volume": "models"}, "/data": {"volume": "data"},
                           "/runs": {"volume": "runs"}, "/envs": {"volume": "envs"}},
                "policy": {"timeout": {"execute": t.timeout}},
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
    job["env"] = job["env"] + ["NCCL_DEBUG=INFO"]
    return wf


def launch_script(t: Trial) -> str:
    py = ENV["python"]
    if t.nodes == 1:
        return f"#!/bin/bash\nmkdir -p /tmp/home\nexec {py} /opt/train.py --config /opt/trial.json\n"
    # Same script runs on every node; rank = position of this host in the sorted host list.
    return (
        "#!/bin/bash\nmkdir -p /tmp/home\n"
        "HOSTS=$(echo \"$MULTINODE_HOSTLIST_NOSLOTS\" | tr ',' '\\n' | sort)\n"
        "MASTER=$(echo \"$HOSTS\" | head -1); RANK=$(echo \"$HOSTS\" | grep -nx \"$(hostname)\" | cut -d: -f1)\n"
        "echo \"host=$(hostname) master=$MASTER node_rank=$((RANK-1)) hosts=$(echo $HOSTS)\"\n"
        f"exec {py} -m torch.distributed.run --nnodes {t.nodes} --nproc-per-node 1 --node-rank $((RANK-1)) "
        "--master-addr $MASTER --master-port 29500 /opt/train.py --config /opt/trial.json\n")


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+"); ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    ts = [Trial(d["id"], d.get("config", {}), nodes=d.get("nodes", 1)) for d in (json.load(open(p)) for p in a.trials)]
    for t in run_all(ts, a.concurrency):
        print(json.dumps({"id": t.id, "status": t.status, "final_eval_loss": final_eval(t)}))
