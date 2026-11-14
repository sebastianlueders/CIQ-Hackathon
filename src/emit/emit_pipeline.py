"""Phase D: write the end-to-end Fuzzball workflow from what the searches measured.

Inputs:  results/search/recipe.json    (phase A winner; falls back to trainer defaults)
         results/serve/best.json       (phase C winner; falls back to BF16)
Output:  workflows/pipeline.yaml       (ingest -> train -> serve-and-verify)

The source code is uploaded once to the Fuzzball object cache under the current git
commit and pulled into each job by ingress, so the workflow pins exactly what ran.

Usage: python src/emit/emit_pipeline.py [--upload]
"""
import argparse, json, os, subprocess, textwrap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FB = os.path.join(REPO, "fuzzball")
CODE = ["src/ingest/ingest.py", "src/prep/fuse_experts.py", "src/train/train_lora.py", "src/kernels/fused_ce.py",
        "src/serve/serve_bench.py"]


def load(path, default):
    p = os.path.join(REPO, path)
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--upload", action="store_true")
    ap.add_argument("--train-tokens", type=int, default=8_000_000)
    ap.add_argument("--out", default="workflows/pipeline.yaml")
    a = ap.parse_args()
    if "validate" in a.out:  # short end-to-end check: small budget, fast eval set
        a.train_tokens = min(a.train_tokens, 500_000)
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    prefix = f"fuzzball://group/code/{sha}"
    if a.upload:
        for f in CODE:
            subprocess.run([FB, "object", "put", os.path.join(REPO, f), f"{prefix}/{os.path.basename(f)}"], check=True)

    import sys; sys.path.insert(0, os.path.join(REPO, "src", "perf"))
    import policy; policy.decide()  # re-derive adopted optimizations (torch.compile first) from latest results
    r = load("results/search/recipe.json", {})
    s = load("results/serve/best.json", {"config": "bf16", "args": []})
    train_cfg = {
        "model": "/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-fused", "data": "/data/tok/lammps",
        "out": "/runs/pipeline/train",
        "lr": r.get("lr", 2e-4), "lora_r": r.get("lora_r", 16), "lora_alpha": r.get("lora_alpha", 32),
        "targets": r.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "seq_len": r.get("seq_len", 4096), "micro_batch": r.get("micro_batch", 1), "grad_accum": r.get("grad_accum", 8),
        "warmup_tokens": 250_000, "schedule": "cosine", "train_tokens": a.train_tokens,
        "eval_every_tokens": 2_000_000, "eval_set": "eval_small.bin" if "validate" in a.out else "eval.bin",
        "save_adapter": True,
    }
    serve_cfg = [{"name": "final", "args": [*s.get("args", []), "--enable-lora", "--max-lora-rank",
                                            str(max(16, train_cfg["lora_r"])),
                                            "--lora-modules", "ft=/runs/pipeline/train/adapter"],
                  "served_model": "ft"}]  # requests to "ft" use the fine-tuned adapter
    # Phase B: use 2-node data parallel only if it measured >= 1.6x the single-node tokens/s.
    def med_tok_s(name):
        p = os.path.join(REPO, "results", "trials", f"{name}.jsonl")
        t = sorted(json.loads(l)["tok_s"] for l in open(p) if '"kind": "train"' in l and '"step": 1,' not in l) if os.path.exists(p) else []
        return t[len(t) // 2] if t else None
    one, two = med_tok_s("s4-perf-experts-grouped"), med_tok_s("s4-perf-dp2")
    dp_nodes = 2 if (one and two and two / one >= 1.6) else 1
    train_cfg["experts_impl"] = "grouped_mm"  # phase B: 1.90x over the per-expert loop
    train_cfg.update(load("results/perf/defaults.json", {}).get("adopted", {}))  # e.g. compile, fused_ce
    provenance = (f"recipe: {r.get('config', 'trainer defaults (search not run)')}, "
                  f"held-out loss {r.get('eval_loss', 'n/a')}; experts grouped_mm; "
                  f"train nodes {dp_nodes} ({one} -> {two} tok/s); serving: {s.get('config')}")

    ingress = "\n".join(
        f"      - source: {{uri: {prefix}/{os.path.basename(f)}}}\n        destination: {{uri: file://{os.path.basename(f)}}}"
        for f in CODE)
    if dp_nodes > 1:
        multinode = f"\n    multinode: {{nodes: {dp_nodes}, implementation: generic}}   # data parallel: one GB10 GPU per node"
        train_script = textwrap.indent(textwrap.dedent(f"""\
            #!/bin/bash
            # Same script on every node; node rank = position of this host in the sorted host list.
            HOSTS=$(echo "$MULTINODE_HOSTLIST_NOSLOTS" | tr ',' '\\n' | sort)
            MASTER=$(echo "$HOSTS" | head -1); RANK=$(echo "$HOSTS" | grep -nx "$(hostname)" | cut -d: -f1)
            exec /envs/train-pt2610/bin/python -m torch.distributed.run --nnodes {dp_nodes} --nproc-per-node 1 \\
              --node-rank $((RANK-1)) --master-addr $MASTER --master-port 29500 /code/train_lora.py --config /opt/train.json"""), "      ")
    else:
        multinode = ""
        train_script = "      #!/bin/bash\n      exec /envs/train-pt2610/bin/python /code/train_lora.py --config /opt/train.json"
    y = f"""\
# End-to-end fine-tune-and-serve workflow, generated by src/emit/emit_pipeline.py
# from the agent's measured search results (git {sha}). Settings chosen:
#   {provenance}
#
# Bring your data, base model, and hardware: every stage below runs on GB10 nodes
# through Fuzzball, and all state lives on Fuzzball volumes backed by VAST, shared
# by every node with no copying. Edit the PARAMETERS block, then:
#   fuzzball workflow start workflows/pipeline.yaml --name pipeline
version: v4

files:
  # PARAMETERS: training recipe (from phase A) and run length.
  train.json: |
{textwrap.indent(json.dumps(train_cfg, indent=1), '    ')}
  # PARAMETERS: serving configuration (from phase C), with the fine-tuned adapter attached.
  serve.json: |
{textwrap.indent(json.dumps(serve_cfg, indent=1), '    ')}

volumes:
  models: {{name: models, use: team3}}   # base weights (VAST)
  data:   {{name: data, use: team3}}     # raw repository + tokenized shards (VAST)
  runs:   {{name: runs, use: team3}}     # adapters, metrics, serving results (VAST)
  envs:   {{name: envs, use: team3}}     # Python environments shared by all jobs (VAST)
  code:                                  # this repo's code, pinned to git {sha}
    use: ephemeral
    size: 1GiB
    ingress:
{ingress}

jobs:
  # 1. Turn the private repository into a training set, locally. Whole groups of
  #    related files are held out for evaluation so near-copies never leak.
  ingest:
    image: {{uri: docker://python:3.12-slim}}
    mounts: {{/data: {{volume: data}}, /models: {{volume: models}}, /code: {{volume: code}}}}
    resource: {{cpu: {{cores: 8}}, memory: {{size: 16GiB}}}}
    script: |
      #!/bin/bash
      set -euo pipefail
      pip install --quiet --no-cache-dir tokenizers numpy
      python3 /code/ingest.py --repo /data/raw/lammps \\
        --tokenizer /models/Qwen/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json --out /data/tok/lammps

  # 2. One-time: rewrite MoE experts into the layout the trainer keeps in memory,
  #    so loading is a straight read from VAST instead of a ~10 min conversion.
  fuse-experts:
    image: {{uri: docker://nvcr.io/nvidia/pytorch:26.10-py3}}
    mounts: {{/models: {{volume: models}}, /envs: {{volume: envs}}, /code: {{volume: code}}}}
    resource: {{cpu: {{cores: 8}}, memory: {{size: 48GiB}}}}
    script: |
      #!/bin/bash
      D=/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-fused
      [ -f $D/model.safetensors.index.json ] && exit 0   # already converted
      /envs/train-pt2610/bin/python -I /code/fuse_experts.py --src /models/Qwen/Qwen3-Coder-30B-A3B-Instruct --dst $D

  # 3. Fine-tune with the recipe the search picked (LoRA, BF16, one GB10 node),
  #    evaluating held-out loss as it goes; saves the adapter to VAST.
  train:
    depends-on: [{{name: ingest, status: FINISHED}}, {{name: fuse-experts, status: FINISHED}}]{multinode}
    image: {{uri: docker://nvcr.io/nvidia/pytorch:26.10-py3}}
    files: {{/opt/train.json: file://train.json}}
    mounts: {{/models: {{volume: models}}, /data: {{volume: data}}, /runs: {{volume: runs}}, /envs: {{volume: envs}}, /code: {{volume: code}}}}
    policy: {{timeout: {{execute: 12h}}}}
    resource: {{cpu: {{cores: 16}}, memory: {{size: 112GiB}}, devices: {{nvidia.com/gpu: 1}}}}
    script: |
{train_script}

  # 4. Serve the fine-tuned model with the serving config the search picked, and
  #    verify it: throughput, latency, and held-out quality on the same tokens.
  serve-and-verify:
    depends-on: [{{name: train, status: FINISHED}}]
    image: {{uri: docker://vllm/vllm-openai:v0.33.0}}
    files: {{/opt/serve.json: file://serve.json}}
    mounts: {{/models: {{volume: models}}, /data: {{volume: data}}, /runs: {{volume: runs}}, /code: {{volume: code}}}}
    resource: {{cpu: {{cores: 16}}, memory: {{size: 112GiB}}, devices: {{nvidia.com/gpu: 1}}}}
    env: [HOME=/tmp]
    script: |
      #!/bin/bash
      exec python3 /code/serve_bench.py --configs /opt/serve.json --out /runs/pipeline/serve
"""
    path = os.path.join(REPO, a.out)
    open(path, "w").write(y)
    print(path)


if __name__ == "__main__":
    main()
