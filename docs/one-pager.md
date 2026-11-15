# Alchemy: Turning Private Data into Fast, Fine-Tuned Local Models

Bring your data, your base model, and your hardware. Our agent runs the experiments on your own hardware and hands back one Fuzzball workflow that fine-tunes the model and serves it as fast as your hardware allows, without your data ever leaving the room.

**Team:** Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania \
**Track:** Multi-Node Training

### The problem

National labs, hospitals, banks, and defense contractors hold data that cannot go to a cloud API. Local models solve this, but these institutions often lack the hardware to run frontier open-source models and must use smaller ones (3B–70B parameters), which fall short of frontier quality. Fine-tuning a local model on the use case is the best way to close that gap, but doing it well takes three scarce skills:

- **ML research** (which recipe actually improves the model on this data),
- **training performance engineering** (running that recipe near the hardware's peak), and
- **inference performance engineering** (serving the result fast without losing quality).

### What we built

An agent that does all three as experiments on the customer's own hardware. Every experiment is a **Fuzzball workflow** the agent writes and launches itself.

1. **Recipe search.** Many short LoRA trials run in parallel, one per node. A successive-halving optimizer stops weak trials early and gives the remaining trials more data. The winner is chosen by held-out eval loss.
2. **Training performance search.** For the winning recipe, the agent searches batch sizes, activation checkpointing, single- vs. multi-node training, and **our custom kernels for Blackwell GB10 (sm_121)**. It accepts a change only if the loss curve is unchanged.
3. **Inference performance search.** It explores precision (BF16, FP8) and serving settings, and rejects any configuration that falls below a quality floor.
4. **Emit.** The result is one clean, commented Fuzzball workflow tuned to this model, this data, and this hardware: it runs the full optimized fine-tuning job and stands up the inference server.

**Custom kernels are integral to the search.** Local hardware often lacks kernels tuned for its architecture, so we write Triton kernels for the hardware and model (a fused linear + cross-entropy over the 152k vocabulary, and a grouped expert GEMM for sm_121) and the agent benchmarks them against the PyTorch defaults like any other option.

### Demo: a coding assistant for a private HPC codebase
**LAMMPS**, a large public C++ simulation code, stands in for a lab's private code. The base model is **Qwen3-Coder-30B-A3B** (MoE, BF16), with whole LAMMPS files held out for evaluation.

| **Result** | **Value** (all measured on GB10; logged in `results/`) |
|------------------|--------------------------------------------------|
| Held-out LAMMPS loss (perplexity), base → fine-tuned | **0.730 (2.08) → 0.600 (1.82), −17.8%** on all 374 held-out files after 2M tokens in the final 2-node `pipeline.yaml` run; recipe search (4 → 2 → 1 configs): 1.175 → 1.037 on a fast eval subset |
| Training throughput (stock → best) | 356 → **2,858 tokens/s (8.0×)** inside the final `pipeline.yaml` run on 2 nodes: grouped experts 1.9×, torch.compile 1.5×, no activation checkpointing 1.35×, 2 nodes 1.98×; loss unchanged. **~19% of measured BF16 peak** (2 × 99 TFLOP/s; 4 × active params per token) |
| Custom Triton fused linear + cross-entropy (152k vocab) | 2.1–2.35× faster and 7.0 → 0.34 GiB alone; in training −6.2 GiB peak memory, exact loss |
| Serving (vLLM): base BF16 → FP8; fine-tuned FP8 + LoRA | 113 → 203 tokens/s at 16 concurrent (1.79×, loss +0.2%); fine-tuned: 188 at 16, **441 at 64 concurrent**, held-out loss 1.293 → **1.168** |
| `pipeline.yaml` end to end | ingest → train → FP8 + LoRA serving in vLLM; a 500k-token validation run took ~35 min |

**What did the second node unlock?** Data-parallel training across both GB10 nodes as one Fuzzball multinode job: 1,011 → 1,998 tokens/s on the same config (1.98×, 99% scaling efficiency; gradient sync 0.05–0.5 s of a 33 s step), and 2,858 tokens/s with all adopted optimizations; plus parallel search trials sharing one copy of weights and data on VAST.

### Built on NVIDIA, CIQ, and VAST Data

- **CIQ Fuzzball:** the agent's entire experiment loop runs as Fuzzball workflows, and its deliverable *is* a Fuzzball workflow.
- **VAST Data:** Fuzzball volumes on VAST hold the weights, dataset, adapters, checkpoints, and results, shared by every trial on every node with no copying. That shared storage is what makes a parallel search cheap.
- **NVIDIA:** GB10 Grace Blackwell nodes, custom kernels written for sm_121, NCCL, and FP8 serving on Blackwell.
