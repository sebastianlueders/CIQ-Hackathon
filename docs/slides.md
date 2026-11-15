---
title: "Alchemy"
subtitle: "Turning Private Data into Fast, Fine-Tuned Local Models"
author: "Tushar Sonthalia · Kunlin Han · Sebastian Lueders · Akshat Singhania"
date: "Multi-Node Training Track"
---

# {.plain}

```{=latex}
\vfill
{\Large Part 1\par}\vspace{6pt}
{\Huge\bfseries The problem and our system\par}
\vfill
```

# Private data has to stay local

- Proprietary data can't be sent to cloud LLM providers
- Local hardware can't host frontier open-source models
    - e.g. Kimi K3 (~2.8T parameters) needs GB300-class systems
- So institutions run smaller models: **3B–80B parameters**
- Smaller models fall short of frontier quality
- **Fine-tuning on their own data** is the best way to close the gap

# Fine-tuning well is hard

1. **ML research**
    - Hyperparameter search for the fine-tuning runs
2. **Training performance engineering**
    - Older or consumer-grade GPUs, mixed hardware
    - Missing kernels, or kernels not tuned for this hardware
3. **Inference engineering**
    - Serving that wins on both throughput and latency

Deeply technical, and not most institutions' core skill

# Alchemy

- **Hyperparameter search** on its own
- **Training performance engineering** loop
- **Inference engineering** loop
- **Out:** one Fuzzball workflow, ending in a vLLM server and a URL

# Hyperparameter search

- Many LoRA experiments in parallel
    - LoRA ranks, learning rates, batch sizes
- Weak runs stopped early, strong runs get more data
- Winner: best loss on a held-out eval set

# Training performance engineering

- **Baseline the hardware:** compute FLOPs, memory bandwidth, network bandwidth
- **Kernels:** write custom Triton kernels, patch and tune existing ones
- **Parallelism:** data, expert, tensor parallel; whichever fits the hardware
- **Overlap and fusion:** compute/communication overlap, CUDA streams, CUDA graphs
- Keep a change only if it's faster **and** the loss holds

# Inference engineering

- Custom kernels for serving
- Best serving path for the hardware
- Best precision the hardware supports: FP8, BF16
- Stands up a vLLM server

# The output: one Fuzzball workflow

- Hyperparameter search
- Training performance engineering
- Inference engineering
- **Ends with a vLLM server and a URL you can use**

# {.plain}

```{=latex}
\vfill
{\Large Part 2\par}\vspace{6pt}
{\Huge\bfseries The experiment\par}
\vfill
```

# Setup

- **Data:** LAMMPS, a large C++ simulation codebase
    - 18.4M training tokens; 374 whole files held out
- **Model:** Qwen3-Coder-30B-A3B (mixture of experts)
- **Hardware:** 2 × NVIDIA DGX Spark (GB10)
- **Platform:** CIQ Fuzzball, VAST Data storage

# Hardware baseline (measured)

- **Compute:** 99 TFLOP/s BF16, 199 TFLOP/s FP8 per node
- **Memory bandwidth:** 209 GB/s per node
- **Storage (VAST):** 1.64 GB/s per stream
- **Network:** 2-node gradient sync in 0.05–0.5 s per step

# Hyperparameter search

- 4 configs: learning rate 1e-4 to 1e-3, LoRA rank 8 to 64
- Successive halving: 4 → 2 → 1
- **Winner:** learning rate 1e-3, rank 8
- Held-out loss **1.175 → 1.037** after 750k tokens

# Training performance

- Stock: **356** tokens/s
- Grouped expert GEMMs: **675** (1.9×)
- \+ torch.compile: **1,011** (1.5×)
- \+ no activation checkpointing: **1,366** (1.35×)
- \+ second node: **2,858** tokens/s, **8.0× overall**
- Model load per run: 10 min → 62 s

# Kernels we wrote

- **Fused linear + cross-entropy** (Triton, 152k vocabulary)
    - Exact loss; 2.1–2.35× faster; 20–38× less memory
    - **Adopted:** the memory it frees makes room to drop checkpointing
- **Grouped expert GEMM** (Triton, sm_121)
    - Correct; 1.9× faster than a per-expert loop
    - 0.4–0.7× of PyTorch's grouped GEMM, so **not adopted**

# Inference engineering

- **Base model, BF16 → FP8:** 113 → 203 tokens/s (1.8×), quality within 0.2%
- **Fine-tuned, FP8 + LoRA:** 192 tokens/s at 16 users, 441 at 64
- Fine-tuned held-out loss **1.293 → 1.145**

# End to end

- One `pipeline.yaml`, on 2 nodes, in **66 min**
- Held-out loss **0.730 → 0.589 (−19.3%)**
- Training **8.0× faster**, serving **1.8× faster**

# Next

- Tune the grouped expert GEMM to beat PyTorch
- Tuned MoE kernels for serving
- An LLM layer that explains every decision

# {.plain}

```{=latex}
\vfill
\begin{center}{\Huge\bfseries Thank you}\end{center}
\vfill
```
