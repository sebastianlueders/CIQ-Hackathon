# Crucible — Local Autoresearch & `sm_121` Performance-Engineering Agent

> **Bring your private codebase, your base model, and your hardware.**  
> Crucible runs the ML research, training-systems, and inference-serving experiments directly on your own **CIQ Fuzzball** cluster over **VAST Data** shared volumes on **NVIDIA GB10 (`sm_121`)** nodes — and hands back **one reproducible Fuzzball workflow** (`ingest -> train -> eval -> serve`). **Your raw data never leaves the room.**

**Team**: Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania · **Track**: Multi-Node Training

---

## Why Crucible

Organizations holding sensitive code and scientific data (national labs, hospitals, financial institutions, defense) cannot upload their corpus to cloud fine-tuning APIs. Running local fine-tuning well requires three scarce disciplines that are normally done manually by trial and error:

1. **Stage A — ML Recipe Search**: Parallel short LoRA trials across cluster nodes scheduled by a deterministic **Successive Halving** optimizer, stopping weak trials early and picking the winner strictly by **sealed whole-file held-out evaluation loss**.
2. **Stage B — Training Performance Search**: Searches micro-batching, activation checkpointing, single- vs. multi-node `torchrun`/NCCL DDP, and our **custom Triton/CUDA kernels for Grace Blackwell GB10 (`sm_121`)** — accepting an optimization **only when the loss curve matches the reference trajectory within tolerance**.
3. **Stage C — Inference Performance Search**: Benchmarks vLLM across precision (`BF16`, `FP8`, Blackwell `NVFP4`), data replicas vs. tensor parallelism (`2xTP1` vs. `1xTP2`), and concurrency — rejecting any configuration that violates the **held-out C++ quality floor** or P99 latency ceiling.
4. **Stage D — Turnkey Emission**: Outputs a single, documented CIQ Fuzzball workflow (`final-crucible-workflow.yaml`) plus a measured Markdown/JSON report (`run_report.md`).

An optional **LLM Advisor Layer** sits strictly above a cryptographic **Privacy Schema Firewall** (`src/crucible/firewall.py`): it inspects only hyperparameter identifiers and scalar metrics, blocking any payload containing code or dataset text and recording SHA-256 digests in `privacy_audit.jsonl`.

---

## Quickstart (Runs Out-of-the-Box Offline or on GB10 Cluster)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui,test]"

# 1. Verify custom sm_121 kernels (runs Triton on CUDA/GB10 or exact PyTorch fallback on CPU)
crucible bench-kernels

# 2. Run the full 4-stage autoresearch loop (offline calibrated GB10 simulator)
crucible run --config configs/demo_lammps_30b.yaml --backend simulator --output-dir runs/demo

# 3. Validate the emitted turnkey CIQ Fuzzball workflow
crucible validate-yaml runs/demo/final-crucible-workflow.yaml

# 4. Launch the interactive Web Studio on http://127.0.0.1:8080
crucible ui --run-dir runs/demo --port 8080
```

### Running Live on the CIQ Fuzzball GB10 Cluster

```bash
# Step 0 (once): Populate /vast/models/Qwen3-Coder-30B-A3B and /vast/data/lammps
fuzzball workflow validate workflows/00-fetch.yaml
fuzzball workflow start workflows/00-fetch.yaml -n crucible-00-fetch

# Step 1a: Auto-discover active Fuzzball cluster, VAST volumes, and GB10 nodes
crucible discover

# Step 1b: Launch Crucible natively via CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve --allow-write --allow-exec`)
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball_mcp --output-dir runs/live-01

# (Alternative) Launch via direct `fuzzball workflow` CLI subcommands
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball --output-dir /vast/runs/live-01

# Step 2: Run the emitted production workflow end-to-end
fuzzball workflow start /vast/runs/live-01/final-crucible-workflow.yaml -n crucible-lammps-prod
```

---

## Custom `sm_121` Kernel Library (`src/crucible/kernels/`)

| Kernel | Hardware Bottleneck Solved | Fallback & Tests |
| --- | --- | --- |
| `chunked_cross_entropy` | Eliminates `~4.9 GB` FP32 `[B*T, 151936]` logit materialization via tiled online log-sum-exp | Autograd forward + backward parity (`< 1e-4`) |
| `grouped_moe_gemm` | Coalesces ragged top-8-of-128 expert projections in `Qwen3-Coder-30B-A3B` | Numerical parity vs. per-expert loop |
| `fused_lora` | Fuses frozen BF16 base projection + scaled low-rank adapter `Wx + (alpha/r) B(Ax)` | Exact `dX, dA, dB` autograd parity |
| `low_prec_decode` | Block-scaled FP8 & Blackwell NVFP4 (`E2M1` 16-element micro-blocks) decode dequant-GEMV | `3.2x` weight memory reduction + quality floor |

---

## Repository Structure

- `configs/` — Cluster topology (`cluster.default.yaml`), primary 30B MoE demo (`demo_lammps_30b.yaml`), and 2-node 80B stretch config (`stretch_lammps_80b.yaml`)
- `workflows/` — `00-fetch.yaml` VAST bootstrap + `examples/final-lammps-qwen3-30b.yaml` sample turnkey output
- `src/crucible/` — Core library (`data`, `eval`, `kernels`, `trainer`, `serving`, `fuzzball`, `optimizer`, `firewall`, `server`, `cli`)
- `tests/` — Unit and end-to-end test suite (`pytest`)
- `HANDOFF.md` — Cluster volume paths, Fuzzfile `v4`/`v1` notes, and hardware pitfalls
