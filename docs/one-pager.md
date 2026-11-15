# Alchemy: An Autoresearch Agent for Local Fine-Tuning and Serving

**Bring your data, your base model, and your hardware. Our agent runs the experiments on your own cluster and hands back one Fuzzball workflow that fine-tunes and serves the best model your hardware allows, without your data ever leaving the room.**

**Team:** Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania · **Track:** Multi-Node Training

### The problem
National labs, hospitals, banks, and defense contractors hold data that cannot go to a cloud API. Fine-tuning locally is the answer, but doing it well takes three scarce skills: **ML research** (which recipe actually improves the model on this data), **training performance engineering** (running that recipe near the hardware's peak), and **inference performance engineering** (serving the result fast without losing quality).

### What we built
An agent that does all three as experiments on the customer's own cluster. **Every experiment is a Fuzzball workflow the agent writes and launches itself.**

1. **Recipe search.** Many short LoRA trials run in parallel, one per node. A successive-halving optimizer stops weak trials early and gives survivors more data. The winner is chosen by held-out eval loss.
2. **Training performance search.** For the winning recipe, the agent searches batch sizes, activation checkpointing, single- vs. multi-node training, and **our custom kernels for Blackwell GB10 (sm_121)**. It accepts a change only if the loss curve is unchanged.
3. **Inference performance search.** It searches precision (BF16, FP8) and serving settings, and rejects any configuration that falls below a quality floor.
4. **Emit.** The result is **one clean, commented Fuzzball workflow** (ingest → train → eval → serve) tuned to this model, this data, and this cluster.

A deterministic optimizer decides which experiments run and a measured adoption rule decides which optimizations stick, so the loop always completes; an LLM layer that explains each decision from configs and metrics only is next.

**Custom kernels are part of the search, not a side project.** Each optimization is an option the agent benchmarks against the PyTorch default as a Fuzzball experiment: torch.compile first (1.49×, adopted), the MoE expert implementation (1.90×), our Triton fused linear + cross-entropy over the 152k vocabulary (exact vs PyTorch, adopted), and a grouped expert GEMM written for sm_121 (benchmark running). Every adopted kernel passed a correctness test against PyTorch.

### Demo: a coding assistant for a private HPC codebase
**LAMMPS**, a large public C++ simulation code, stands in for a lab's private code. The base model is **Qwen3-Coder-30B-A3B** (MoE, BF16), with whole LAMMPS files held out for evaluation.

| Result (all measured on GB10, logged in `results/`) | Value |
|----------------------------|------------------------------------------|
| Held-out LAMMPS loss (perplexity), base → fine-tuned | **0.730 (2.08) → 0.600 (1.82), −17.8%** on all 374 held-out files after 2M tokens in the final 2-node `pipeline.yaml` run; recipe search (4 → 2 → 1 configs): 1.175 → 1.037 on a fast eval subset |
| Training throughput (stock → best) | 356 → **2,858 tokens/s (8.0×)** inside the final `pipeline.yaml` run on 2 nodes: grouped experts 1.9×, torch.compile 1.5×, no activation checkpointing 1.35×, 2 nodes 1.98×; loss unchanged. **~19% of measured BF16 peak** (2 × 99 TFLOP/s; 4 × active params per token) |
| `pipeline.yaml` end to end | ran in ~35 min: ingest → train → FP8 + LoRA serving in vLLM |
| Custom Triton fused linear + cross-entropy (152k vocab) | 2.1–2.35× faster and 7.0 → 0.34 GiB alone; in training −6.2 GiB peak memory, exact loss |
| Serving (vLLM): base BF16 → FP8; fine-tuned FP8 + LoRA | 113 → 203 tokens/s at 16 concurrent (1.79×, loss +0.2%); fine-tuned: 188 at 16, **441 at 64 concurrent**, held-out loss 1.293 → **1.168** |

**What did the second node unlock?** Data-parallel training across both GB10 nodes as one Fuzzball multinode job: 1,011 → 1,998 tokens/s on the same config (1.98×, 99% scaling efficiency; gradient sync 0.05–0.5 s of a 33 s step), and 2,807 tokens/s with all adopted optimizations; plus parallel search trials sharing one copy of weights and data on VAST.

### Built on NVIDIA, CIQ, and VAST Data
**CIQ Fuzzball:** the agent's entire experiment loop runs as Fuzzball workflows, and its deliverable *is* a Fuzzball workflow. **VAST Data:** Fuzzball volumes on VAST hold the weights, dataset, adapters, checkpoints, and results, shared by every trial on every node with no copying. That shared storage is what makes a parallel search cheap. **NVIDIA:** GB10 Grace Blackwell nodes, custom kernels written for sm_121, NCCL, and FP8 serving on Blackwell. **Next:** the agent generates and validates new kernels for hardware it hasn't seen.
