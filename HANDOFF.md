# Crucible Cluster & Architecture Handoff (`HANDOFF.md`)

**Team**: Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania  
**Track**: Multi-Node Training (NVIDIA GB10 · CIQ Fuzzball · VAST Data)

---

## 1. Cluster Facts & VAST Volume Layout

- **Compute Nodes**: 2x NVIDIA GB10 Grace Blackwell nodes (`aarch64` 20-core Grace CPU + Blackwell `sm_121` GPU with 128 GB unified LPDDR5X memory per node; ~112 GB safely allocatable to containers).
- **Shared Storage (VAST Data)**: Persistent Fuzzball volume `crucible-vast` (`provisioner: vast`) mounted at `/vast` across every trial and both nodes with zero per-trial data copying:
  - `/vast/models/Qwen3-Coder-30B-A3B` — 57 GB BF16 base weights (fetched by `workflows/00-fetch.yaml`)
  - `/vast/data/lammps` — 15,605-file LAMMPS C++ repository (fetched by `workflows/00-fetch.yaml`)
  - `/vast/data/lammps-packed` — deterministically split (`train` / `val` / sealed `held_out`) packed chunks + SHA-256 manifest
  - `/vast/runs/<run_id>/` — per-trial Fuzzfiles, JSONL step metrics, LoRA adapters, benchmark logs, privacy audit trail, and final emitted workflow
- **Overriding Cluster Identifiers**: Edit `configs/cluster.default.yaml` or set `CRUCIBLE_VAST_VOLUME`, `CRUCIBLE_VAST_PROVISIONER`, `CRUCIBLE_TRAIN_IMAGE`, `CRUCIBLE_SERVE_IMAGE`, or `CRUCIBLE_FUZZFILE_VERSION` (`v4` or `v1`).

---

## 2. CIQ Fuzzball Syntax Rules & Hardware Pitfalls

1. **Unique Stage Namespace**: In Fuzzball `v4`, volumes, jobs, and services share one allocation namespace — never give a job and a service or volume the same name (`crucible validate-yaml` enforces this automatically).
2. **`mounts` Map (`v4` vs `v1`)**:
   - `version: v4` maps container path to volume key: `mounts: { /vast: vast-shared }` and uses `depends-on: [{ name: ..., status: FINISHED }]`.
   - `version: v1` maps volume key to `{ location: /vast }` and uses `requires: [...]`.
   - `crucible` renders `v4` by default and supports `--fuzzfile-version v1`.
3. **GB10 (`sm_121`) Unified Memory & Logit Spikes**:
   - `Qwen3-Coder-30B-A3B` has a `151,936`-entry vocabulary. Materializing full `[B, T, 151936]` FP32 logits for cross-entropy consumes `~4.9 GB` at `B=4, T=2048`, starving unified memory bandwidth and forcing small micro-batches or heavy activation checkpointing.
   - Our **Kernel 1 (`chunked_cross_entropy`)** tiles hidden states and computes online log-sum-exp in Triton on `sm_121` without ever materializing the `[B*T, V]` logit tensor, unlocking larger micro-batches without activation checkpointing while matching PyTorch loss to `< 1e-4`.
4. **Multi-Node Training (`torchrun` + NCCL)**:
   - Multi-node Fuzzball jobs specify `multinode: { nodes: 2, implementation: generic, procs-per-node: 1 }` and coordinate rank-0 rendezvous over shared VAST (`/vast/runs/<run_id>/rdzv`) so `torchrun` initializes NCCL reliably without hardcoding node IPs.

---

## 3. Custom `sm_121` Kernel Library (`src/crucible/kernels/`)

| # | Kernel | Bottleneck Addressed on GB10 `sm_121` | Verification |
|---|---|---|---|
| 1 | `chunked_cross_entropy` | Eliminates 152k-vocab FP32 logit materialization; tiles projection + online softmax CE | Forward + autograd backward vs `F.cross_entropy` |
| 2 | `grouped_moe_gemm` | Coalesces top-8-of-128 active expert projections in `Qwen3-Coder-30B-A3B` | Numerical parity vs per-expert reference loop |
| 3 | `fused_lora` | Fuses base projection + low-rank adapter `Wx + (alpha/r) B(Ax)` to cut HBM/LPDDR5X round-trips | Forward + `dX, dA, dB` gradient parity |
| 4 | `low_prec_decode` | Block-scaled FP8 / NVFP4 weight dequant-GEMV for memory-bound batch-decode serving | Bounded quantization error + quality-floor check |

Every kernel auto-detects Triton + CUDA `sm_121` at runtime and falls back cleanly to exact reference PyTorch kernels on non-Blackwell GPUs or CPU CI environments.

---

## 4. Privacy Guarantee (Hard Schema Firewall)

The LLM advisor (`src/crucible/optimizer/advisor.py`) communicates exclusively through `src/crucible/firewall.py`:
- Only strongly-typed numeric metrics, enum identifiers, and hyperparameter grids can cross the boundary.
- Any payload containing C++ source tokens, filesystem snippets, long free-text strings, or non-allowlisted keys raises `DataIsolationViolation` before serialization and is logged to `privacy_audit.jsonl` with SHA-256 payload digests.

---

## 5. Quick Commands

```bash
# Run full 4-stage autoresearch loop in offline calibrated simulator mode (completes in seconds)
crucible run --config configs/demo_lammps_30b.yaml --backend simulator --output-dir runs/demo

# Run against live CIQ Fuzzball cluster
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball --output-dir /vast/runs/live-01

# Benchmark custom kernels (runs Triton on GPU or PyTorch verification on CPU)
crucible bench-kernels

# Validate any generated Fuzzfile
crucible validate-yaml runs/demo/final-crucible-workflow.yaml

# Launch interactive live dashboard
crucible ui --run-dir runs/demo --port 8080
```
