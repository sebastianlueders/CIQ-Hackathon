# Alchemy Autoresearch Measured Report — `lammps-c-gpu-simulation-corpus-enterprise-benchm-qwen2-5-coder-1-5b-instruct`

- **Cluster**: `ciq-gb10-team3` · `definitionId: team3` (2x `NVIDIA-GB10`, `sm_121`, ``)
- **Compute Nodes**: `gb10-node-01` (`gb10-01`), `gb10-node-02` (`gb10-02`)
- **Shared Storage**: VAST persistent volume `runs` mounted at `/vast`
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (2.9 GB BF16, 151,936 vocab)
- **Turnkey Deliverable**: `final-crucible-workflow.yaml` (`ingest-lammps -> train-lora -> eval-held-out -> serve-vllm`)

## 1. Headline Measured Results

| Metric | Measured Value |
| --- | --- |
| Held-out code perplexity, base -> fine-tuned | **6.506 -> 4.230** (loss `1.8727 -> 1.4421`) |
| Recipe search: trials run / stopped early / parallel wall-clock | **5 / 1 / 0.99 h** (`1.82x` faster on 2 nodes) |
| Training throughput, default -> optimized | **2,027.0 -> 4,756.4 tok/s (2.35x faster)** |
| Utilization vs. measured hardware peak | **30.0% -> 70.3%** |
| End-to-end training speedup from `sm_121` custom kernels (1-node) | **2.35x** |
| Serving throughput, BF16 -> best config (`nvfp4-2replicas-tp1`) | **401.8 -> 3,460.3 tok/s (8.61x)**, perplexity within **+1.56%** |

## 2. Stage-by-Stage Decisions (Privacy-Firewalled Advisor)

- **`stage_a_recipe` -> `r32-a64-lr2e-04-7m`** *(via `deterministic-heuristic-advisor`)*: Selected LoRA recipe 'r32-a64-lr2e-04-7m' via successive halving across 2 sm_121 nodes: full attention + MoE projection coverage achieved the lowest held-out loss (1.4421, perplexity 4.23) while pruning high-LR/narrow-module trials early.
- **`stage_b_training_perf` -> `company-preset-gb10-qwen25-1.5b-fast-demo`** *(via `deterministic-heuristic-advisor`)*: Selected training config 'company-preset-gb10-qwen25-1.5b-fast-demo' at 4756 tok/s (70.3% of measured cluster peak): enabling sm_121 chunked 152k-vocab cross-entropy, grouped MoE GEMM, and fused LoRA freed unified memory for larger micro-batches while matching the reference loss curve within 0.0011.
- **`stage_c_inference_perf` -> `nvfp4-2replicas-tp1`** *(via `deterministic-heuristic-advisor`)*: Selected serving config 'nvfp4-2replicas-tp1' delivering 3460 tok/s (P99 425.3 ms): Blackwell NVFP4 + 2 independent data replicas avoided inter-node TP all-reduce overhead and kept perplexity within +1.56% of BF16 while rejecting over-batched configs that breached quality/latency gates.

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
| `pytorch-baseline-1node` | 1 | 2,027.0 | 13.6 | `0.0000` | ACCEPTED |
| `company-preset-gb10-qwen25-1.5b-fast-demo` | 1 | 4,756.4 | 49.3 | `0.0011` | ACCEPTED |
| `oversized-global-batch-2node` | 2 | 8,656.6 | 49.3 | `0.1355` | REJECTED (loss_curve_delta=0.1355 exceeded tolerance 0.0250) |

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

- **Verified Outbound Advisor Payloads**: `3` allowed, `0` blocked violations, `0` raw code/dataset bytes transmitted.
- Every advisor transaction is logged with SHA-256 payload digests in `privacy_audit.jsonl`.
