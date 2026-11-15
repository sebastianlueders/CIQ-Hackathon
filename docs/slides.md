---
title: "Alchemy"
subtitle: "Turning Private Data into Fast, Fine-Tuned Local Models"
author: "Tushar Sonthalia · Kunlin Han · Sebastian Lueders · Akshat Singhania"
date: "Multi-Node Training Track"
---

# The problem

- Private data can't go to a cloud API
- Local hardware means smaller models
- Fine-tuning closes the gap, but takes three scarce skills:
    - ML research
    - Training performance engineering
    - Inference performance engineering

# Alchemy

- **In:** your data, your base model, your hardware
- **The agent runs the experiments** on your cluster
- **Out:** one Fuzzball workflow that fine-tunes and serves the model
- Data never leaves the room

# How it works

1. **Recipe search:** find the best fine-tuning recipe
2. **Training performance:** make that recipe fast
3. **Inference performance:** serve it fast, without losing quality
4. **Emit:** one tuned Fuzzball workflow

Every experiment is a Fuzzball workflow the agent launches itself

# How the agent decides

- Successive halving: stop weak trials early
- Keep a change only if it's faster **and** the loss holds
- Reject serving configs below a quality floor
- Every number is measured, never assumed

# Demo

- **Codebase:** LAMMPS, standing in for a lab's private code
- **Model:** Qwen3-Coder-30B-A3B (MoE)
- **Hardware:** 2 × NVIDIA GB10
- Whole files held out for evaluation

# Results

- **Quality:** held-out loss **−19.3%**
- **Training:** **8.0× faster** (356 → 2,858 tokens/s)
- **Serving:** FP8 **1.8× faster**, quality within 0.2%
- **End to end:** raw code → served model in ~66 min, one workflow

# Where the speed came from

- Grouped expert GEMMs: **1.9×**
- torch.compile: **1.5×**
- No activation checkpointing: **1.35×**
- Second node: **1.98×**
- Custom Triton kernel: **up to 38× less memory** for the loss

# What the second node unlocked

- Data-parallel training across both GB10 nodes
- **1.98× speedup, 99% scaling efficiency**
- Parallel experiments on one shared copy of the data

# Built on

- **CIQ Fuzzball:** every experiment, and the final deliverable
- **VAST Data:** shared weights, data, and results across nodes
- **NVIDIA:** GB10, custom sm_121 kernels, NCCL, FP8

# Next

- Grouped expert GEMM kernel for sm_121
- Tuned MoE kernels for serving
- An LLM layer that explains every decision

# {.plain}

\vfill
\begin{center}{\Huge\bfseries Thank you}\end{center}
\vfill
