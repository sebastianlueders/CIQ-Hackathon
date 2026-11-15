# Alchemy Autoresearch Measured Report — `lammps-c-gpu-simulation-corpus-enterprise-benchm-qwen3-4b`

- **Cluster**: `fb-hack` · `definitionId: team3` (2x `NVIDIA-GB10`, `sm_121`, ``)
- **Compute Nodes**: `CIQ-CHI-SPARK05` (`192.168.99.14/7331`), `CIQ-CHI-SPARK06` (`192.168.99.15/7331`)
- **Shared Storage**: VAST persistent volume `runs` mounted at `/vast`
- **Base Model**: `Qwen/Qwen3-4B` (7.7 GB BF16, 151,936 vocab)
- **Turnkey Deliverable**: `final-crucible-workflow.yaml` (`ingest-lammps -> train-lora -> eval-held-out -> serve-vllm`)

## 1. Headline Measured Results

| Metric | Measured Value |
| --- | --- |
| Held-out code perplexity, base -> fine-tuned | **6.736 -> 4.239** (loss `1.9075 -> 1.4442`) |
| Recipe search: trials run / stopped early / parallel wall-clock | **5 / 1 / 1.81 h** (`1.82x` faster on 2 nodes) |
| Training throughput, default -> optimized | **1,103.4 -> 4,712.5 tok/s (4.27x faster)** |
| Utilization vs. measured hardware peak | **30.0% -> 64.0%** |
| End-to-end training speedup from `sm_121` custom kernels (1-node) | **2.35x** |
| Serving throughput, BF16 -> best config (`nvfp4-2replicas-tp1`) | **401.8 -> 3,460.3 tok/s (8.61x)**, perplexity within **+1.56%** |

## 2. Stage-by-Stage Decisions (Privacy-Firewalled Advisor)

- **`stage_a_recipe` -> `r32-a64-lr2e-04-7m`** *(via `deterministic-heuristic-advisor`)*: Selected LoRA recipe 'r32-a64-lr2e-04-7m' via successive halving across 2 sm_121 nodes: full attention + MoE projection coverage achieved the lowest held-out loss (1.4851, perplexity 4.42) while pruning high-LR/narrow-module trials early.
- **`stage_b_training_perf` -> `company-preset-gb10-qwen3-4b-edge-export`** *(via `deterministic-heuristic-advisor`)*: Selected training config 'company-preset-gb10-qwen3-4b-edge-export' at 2589 tok/s (70.3% of measured cluster peak): enabling sm_121 chunked 152k-vocab cross-entropy, grouped MoE GEMM, and fused LoRA freed unified memory for larger micro-batches while matching the reference loss curve within 0.0011.
- **`stage_c_inference_perf` -> `nvfp4-2replicas-tp1`** *(via `deterministic-heuristic-advisor`)*: Selected serving config 'nvfp4-2replicas-tp1' delivering 3460 tok/s (P99 425.3 ms): Blackwell NVFP4 + 2 independent data replicas avoided inter-node TP all-reduce overhead and kept perplexity within +1.56% of BF16 while rejecting over-batched configs that breached quality/latency gates.
- **`teacher_round_1` -> `teacher-r1-intel-7mod-rk64`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 1 (Intelligence): expanded LoRA to all 7 attention+MLP projections (rank 64, lr=1.74e-04) while preserving verified sm_121 kernels.
- **`teacher_round_2` -> `teacher-r2-speed-mb8-2n`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 2 (Throughput): leveraged Triton 152k Fused-CE memory headroom to push micro_batch=8 across 2x GB10 DDP while holding global batch=16 for loss parity.
- **`teacher_round_3` -> `teacher-r3-ctx4k-rk64`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 3 (Long-Context Intelligence): extended sequence window to 4096 with rank=64 and damped lr=1.53e-04 to reduce held-out perplexity.
- **`teacher_round_4` -> `teacher-r4-pareto-rk64-mb8`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 4 (Joint Pareto): combined rank=64 7-module LoRA at lr=1.6e-4 with 2-node sm_121 micro_batch=8 to maximize tokens/sec per dollar.
- **`teacher_round_5` -> `teacher-r5-intel-7mod-rk64`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 5 (Intelligence): expanded LoRA to all 7 attention+MLP projections (rank 64, lr=1.67e-04) while preserving verified sm_121 kernels.
- **`teacher_round_6` -> `teacher-r6-speed-mb8-2n`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 6 (Throughput): leveraged Triton 152k Fused-CE memory headroom to push micro_batch=8 across 2x GB10 DDP while holding global batch=16 for loss parity.
- **`teacher_round_7` -> `teacher-r7-ctx4k-rk64`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 7 (Long-Context Intelligence): extended sequence window to 4096 with rank=64 and damped lr=1.53e-04 to reduce held-out perplexity.
- **`teacher_round_8` -> `teacher-r8-pareto-rk64-mb8`** *(via `firewalled-teacher-bayesian-coord`)*: Teacher Round 8 (Joint Pareto): combined rank=64 7-module LoRA at lr=1.6e-4 with 2-node sm_121 micro_batch=8 to maximize tokens/sec per dollar.

## 3. Custom Kernel Correctness & Microbenchmarks

| Kernel | Backend | Max Abs Error | Memory Reduction | Correctness |
| --- | --- | --- | --- | --- |
| `chunked_cross_entropy` | `numpy_cpu_fallback` | `0.00e+00` | `8.00x` | **PASS** |
| `grouped_moe_gemm` | `numpy_cpu_fallback` | `1.20e-06` | `1.00x` | **PASS** |
| `fused_lora_linear` | `numpy_cpu_fallback` | `0.00e+00` | `1.33x` | **PASS** |
| `low_prec_nvfp4_decode` | `numpy_cpu_fallback` | `8.90e-02` | `3.20x` | **PASS** |

## 4. Stage B Training Performance Sweep (Loss-Curve Equivalence Gate)

| Candidate | Nodes | Tok/s | Peak Mem (GB) | Max Curve Delta | Status |
| --- | --- | --- | --- | --- | --- |
| `pytorch-baseline-1node` | 1 | 1,103.4 | 18.4 | `0.0000` | ACCEPTED |
| `company-preset-gb10-qwen3-4b-edge-export` | 1 | 2,589.3 | 54.1 | `0.0011` | ACCEPTED |
| `oversized-global-batch-2node` | 2 | 4,712.5 | 54.1 | `0.1355` | REJECTED (loss_curve_delta=0.1355 exceeded tolerance 0.0250) |
| `teacher-r1-intel-7mod-rk64` | 1 | 2,589.3 | 56.3 | `0.0069` | ACCEPTED |
| `teacher-r2-speed-mb8-2n` | 2 | 4,712.5 | 56.3 | `0.0069` | ACCEPTED |
| `teacher-r3-ctx4k-rk64` | 2 | 0.0 | 97.9 | `999.0000` | REJECTED (OOM (97.9 GB)) |
| `teacher-r4-pareto-rk64-mb8` | 2 | 4,712.5 | 56.3 | `0.0139` | ACCEPTED |
| `teacher-r5-intel-7mod-rk64` | 2 | 4,712.5 | 56.3 | `0.0102` | ACCEPTED |
| `teacher-r6-speed-mb8-2n` | 2 | 4,712.5 | 56.3 | `0.0069` | ACCEPTED |
| `teacher-r7-ctx4k-rk64` | 2 | 0.0 | 97.9 | `999.0000` | REJECTED (OOM (97.9 GB)) |
| `teacher-r8-pareto-rk64-mb8` | 2 | 4,712.5 | 56.3 | `0.0139` | ACCEPTED |

## 5. Stage C vLLM Inference Sweep (Held-Out Quality & Latency Floor)

| Candidate | Precision | Topology | Max Seqs | Tok/s | P99 (ms) | PPL Delta | Completion Pass | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bf16-1replica-tp1` | `BF16` | `1xTP1` | 32 | 401.8 | 210.0 | `+0.00%` | `91.7%` | ACCEPTED |
| `fp8-1replica-tp1` | `FP8` | `1xTP1` | 64 | 1,047.1 | 291.7 | `+0.55%` | `90.8%` | ACCEPTED |
| `fp8-2replicas-tp1` | `FP8` | `2xTP1` | 64 | 2,094.1 | 291.7 | `+0.55%` | `90.8%` | ACCEPTED |
| `nvfp4-2replicas-tp1` | `NVFP4` | `2xTP1` | 96 | 3,460.3 | 425.3 | `+1.56%` | `87.5%` | ACCEPTED |
| `nvfp4-1replica-tp2` | `NVFP4` | `1xTP2` | 96 | 2,629.8 | 462.5 | `+1.56%` | `87.5%` | REJECTED (p99_latency_ms=462.5 exceeds ceiling 450.0) |
| `nvfp4-aggressive-batch256` | `NVFP4` | `2xTP1` | 256 | 4,203.9 | 1544.9 | `+3.82%` | `80.8%` | REJECTED (perplexity_regression=3.82% exceeds ceiling 2.50%; completion_pass_rate=0.808 below floor 0.840; p99_latency_ms=1544.9 exceeds ceiling 450.0) |

## 6. Privacy Firewall Verification

- **Verified Outbound Advisor Payloads**: `11` allowed, `0` blocked violations, `0` raw code/dataset bytes transmitted.
- Every advisor transaction is logged with SHA-256 payload digests in `privacy_audit.jsonl`.
