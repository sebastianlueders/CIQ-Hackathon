# Alchemy — Enterprise Client Studio & Fuzzball Remote GPU Local LLM Forge (Team 3)

> **Run Alchemy Studio on your client business machine. Dispatch private tuning to your CIQ Fuzzball 2xGB10 GPU server. Receive a turnkey local LLM & branded copilot app.**  
> Alchemy runs its client-facing UI on a thin business workstation (`http://127.0.0.1:8000`), stages proprietary corpora into leak-free Train / Validation / Sealed Test splits, dispatches 4-stage + Continuous Remote-Teacher LoRA, `sm_121` kernel, and vLLM serving optimization to a remote **CIQ Fuzzball** cluster (`api.fb-hack.civilfritz.net:443`, `team3@fb-hack.local`) over **VAST Data** shared volumes on **2x NVIDIA GB10 (`sm_121`)** nodes — and packages a ready-to-run local LLM build (`scripts/run_company_app.py` + `Modelfile.ollama` + Python Client SDK + `fuzzfile.yaml`). **Zero raw bytes ever leave your local/VAST boundary.**

**Team 3**: Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania · **Track**: Multi-Node Training (NVIDIA GB10 · CIQ Fuzzball · VAST Data)

---

## Live Team 3 Cluster Target (`definitionId: team3`)

Alchemy is pre-configured out of the box for **Team 3** (`team3@fb-hack.local`) on Grace Blackwell DGX Spark nodes:

| Node | Substrate ID | GPU (`sm_121`) | CPU (`cpu/arm64/sve`) | Unified RAM | Fast-Schedule Container Alloc |
| --- | --- | --- | --- | --- | --- |
| `CIQ-CHI-SPARK05` | `192.168.99.14/7331` | `1x NVIDIA-GB10` (`nvidia.com/gpu: 1`) | 20 cores | 127.9 GiB | `8–16 cores / 80–96GiB` |
| `CIQ-CHI-SPARK06` | `192.168.99.15/7331` | `1x NVIDIA-GB10` (`nvidia.com/gpu: 1`) | 20 cores | 127.9 GiB | `8–16 cores / 80–96GiB` |
| **Team 3 Total** | `use: team3` | **2x `nvidia.com/gpu`** | **40 cores** | **255.8 GiB** | Shared VAST volumes: `runs`, `models`, `data`, `envs` |

- **Pre-Cached ARM64 + `sm_121` Container Images**: `docker://nvcr.io/nvidia/pytorch:26.10-py3` (CUDA 13.4 + Triton 3.8 + `sm_121` verified on GB10; `99.1` BF16 / `198.6` FP8 TFLOP/s) and `docker://vllm/vllm-openai:v0.33.0`.
- **Pre-Loaded Team 3 VAST Models (`/models`)**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (`56.9 GiB`), `Qwen/Qwen3-4B`, and `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`3.1 GiB`), plus 52 catalog presets across 6 categories and 1-click VAST staging for any HuggingFace ID.

---

## One-Command Launch on Team 3 (`http://127.0.0.1:8000`)

```bash
# Start the interactive Enterprise Client Studio at http://127.0.0.1:8000 (auto-detects team3 & live Wi-Fi):
./scripts/run_team3.sh --studio-only

# Or launch AND immediately run live Fuzzball MCP optimization on Team 3 GPUs:
./scripts/run_team3.sh --model Qwen/Qwen2.5-Coder-1.5B-Instruct --backend fuzzball_mcp --time-budget 5 --cost-budget 3.50

# Or start the FastAPI Studio directly via the CLI:
alchemy serve --port 8000
```

From `http://127.0.0.1:8000`:
1. **Pick Any Base Model**: Filter 52 open-weight presets (`Min: 1 GPU` / `Min: 2 GPUs`, `Pre-Loaded`, `Preset Opt` with company `sm_121` warm-start dollar savings) or paste any HuggingFace `org/model` ID.
2. **Company Workspace & Private Corpus**: Set your enterprise copilot brand (default: **`Apex Dynamics`**) and drop proprietary code/JSONL/docs (or use the built-in LAMMPS C++ benchmark) for deterministic local Train / Validation / Sealed Test splitting.
3. **Continuous Remote-Teacher & Budget**: Set your Time Budget (minutes) and Cost Cap (`$ USD`), dispatch to **`CIQ Fuzzball 4.2 MCP Server · Live Hardware`**, watch the **Live Dual-GPU Usage & 200GbE RoCE NCCL Fabric**, real-time loss-curve equivalence (`Δ <= 0.025`), and Teacher rounds stream live, then test your tuned LLM side-by-side or download the turnkey `.zip`!

### Unzip & Launch the Built Company Copilot App Locally

```bash
cd ~/Downloads
unzip -o apex-dynamics-private-llm-*.zip -d apex-ai && python3 apex-ai/scripts/run_company_app.py
```
Opens a standalone, zero-dependency **`Apex Dynamics — Private Engineering Copilot`** web app at `http://127.0.0.1:8765` on the client laptop that executes prompts directly on your private Team 3 Fuzzball GB10 cluster (alongside `fuzzfile.yaml`, `Modelfile.ollama`, `scripts/launch_fuzzball.sh`, `scripts/client_query_llm.py`, `BUILD_REPORT.md`, and `privacy_audit.jsonl`).

---

## 4-Stage + Continuous Remote-Teacher Loop & Privacy Schema Firewall

1. **Stage A — ML Recipe Search**: Parallel LoRA trials across `CIQ-CHI-SPARK05` and `CIQ-CHI-SPARK06` scheduled by deterministic **Successive Halving**, pruned by Validation loss and ranked by **Sealed Held-Out Test perplexity**.
2. **Stage B — Training Performance Search**: Warm-starts from verified company `sm_121` hardware presets or sweeps micro-batching, activation checkpointing, 1- vs. 2-node `torchrun`/NCCL DDP (`1,011 -> 2,006 tok/s`, `99.2%` scaling efficiency), and custom `sm_121` kernels — accepted **only when the loss curve matches within `Δ <= 0.025`**.
3. **Stage C — Inference Performance Search**: Sweeps vLLM precision (`BF16`, `FP8`, Blackwell `NVFP4`), data replicas vs. tensor parallelism (`2xTP1` vs. `1xTP2`), and concurrency — gated on the **Held-Out Test quality floor** and P99 ceiling.
4. **Continuous Remote-Teacher Rounds + Turnkey Emission**: Iteratively proposes firewalled joint intelligence + throughput mutations within your Time & `$` Cost budgets, then emits `final-crucible-workflow.yaml` + the downloadable client `.zip`.

All Remote Teacher / Advisor requests pass through `PrivacySchemaFirewall` (`src/crucible/firewall.py`), proving with SHA-256 audit records that `0` raw code/dataset bytes ever leave `/vast`.

---

## Custom `sm_121` Kernel Library (`src/crucible/kernels/`)

| Kernel | Hardware Bottleneck Solved on GB10 `sm_121` | Verification |
| --- | --- | --- |
| `chunked_cross_entropy` | Eliminates `~6.96 GB -> 0.34 GB` FP32 `[B*T, 151936]` logit temporary via tiled online log-sum-exp (`2.35x` faster) | Forward + backward parity (`< 1e-4`) |
| `grouped_moe_gemm` | Coalesces ragged top-8-of-128 expert projections in `Qwen3-Coder-30B-A3B` (`1.90x` step speedup) | Numerical parity vs. per-expert loop |
| `fused_lora` | Fuses frozen BF16 base projection + scaled low-rank adapter `Wx + (alpha/r) B(Ax)` | Exact `dX, dA, dB` autograd parity |
| `low_prec_decode` | Block-scaled FP8 & Blackwell NVFP4 (`E2M1` 16-element micro-blocks) decode dequant-GEMV | `3.2x` weight compression + quality floor |
