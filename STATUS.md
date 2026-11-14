# Morning checklist (overnight queue submitted Sat ~4:20 PM)

All overnight jobs are listed in `results/overnight/manifest.json` (Fuzzball workflow IDs). In the morning:

```bash
python3 src/runner/runner.py --collect results/overnight/manifest.json   # pull logs + RESULT records into results/trials/
python3 src/perf/policy.py                                              # re-decide adopted optimizations (compile is already adopted)
python3 src/report/make_report.py                                       # regenerate results/summary.md
bash docs/build/build-pdfs.sh                                           # one-pager PDF (docs/ is local, git-ignored)
```

| Overnight job | What to look at |
|---|---|
| `s4-perf-dp2-ip-2`, `s4-perf-dp2-c10d-2` | **2-node DP throughput** (tok_s is cluster-wide; compare to 1011 single-node with compile). `DIAG` lines and `== worker log` show both ranks. `allreduce_s` = gradient sync time per step. (`s4-perf-dp2-ip` / `-c10d` without `-2` were the broken launcher: ignore.) |
| `s4-train-dp2-winner` | **Winning recipe trained on 2 nodes** (4M tokens, cosine schedule, evals every 1M): the "second node unlocked" result. |
| `s4-perf-dp2-stack-nockpt` / `s4-perf-dp2-stack` / `s4-perf-compile-stack` | **Stacked** best config: compile + fused CE + micro-batch 2 (+ checkpointing off, 110 GiB) on 2 nodes, plus the same stack on 1 node. Expect the nockpt one to OOM if activations exceed memory; the `-stack` one is the fallback. Judge batch changes by held-out eval, not per-step loss. |
| `s4-perf-compile-{fused-ce,mb2,nockpt,seq8192}` | What stacks on top of compile (1011 tok/s). Run `policy.py` after adding them to `CANDIDATES`. |
| `s5-serve-ft-adapter` | Fine-tuned model served: FP8 vs BF16 + LoRA at C=1/16/64, held-out loss. |
| `s5-tune-moe-gb10` | vLLM fused-MoE tile tuning for GB10 (`/runs/serve/moe-configs/`), then FP8 serving re-measured with tuned tiles (compare to 203 tok/s at C=16). |
| `s6-pipeline-full` | **`workflows/pipeline.yaml` end to end**: 8M tokens (winning recipe + grouped_mm + compile), then FP8 + LoRA serving. |
| `s6-pipeline-validate-2` | Short end-to-end check (500k tokens). Note: it shares `/runs/pipeline/train` with `s6-pipeline-full`; trust the full run's adapter. |

**2-node root cause (fixed):** with `implementation: generic`, Fuzzball runs the job script on the head node only; workers must be started with `"$MULTINODE_SSH_WRAPPER" "$host" "<one shell string>"` (see `/multinode/generic-helper` inside any multinode job). Multinode containers do see RDMA devices (`/dev/infiniband/uverbs0-3`), so NCCL over RoCE is worth testing (`c10d` variant leaves IB enabled).

# Status (updated Sat Nov 14, ~4:20 PM CST)

An agent loop for local fine-tuning and serving on GB10 nodes via Fuzzball, with VAST-backed volumes.
Every number comes from `results/`; regenerate the table with `python src/report/make_report.py` (→ `results/summary.md`).

## Measured
| What | Value | Source |
|---|---|---|
| GB10 BF16 / FP8 GEMM peak (NGC PyTorch 26.10) | 99.1 / 198.6 TFLOP/s; 209 GB/s memory read | `results/env_probe_ngc.json` |
| VAST read, one stream | 1.64 GB/s | `results/vast_read.json` |
| Training data (LAMMPS) | 18.4M train / 0.9M held-out tokens, held out by file group | `results/ingest.json` |
| Model load per trial | ~10 min → **62 s** (fused-experts checkpoint + direct loader) | `results/fuse_experts.json`, smoke trials |
| Held-out loss, base → recipe-search winner (750k tokens) | **1.175 → 1.037** (ppl 3.24 → 2.82, −11.7%), lr 1e-3, rank 8 (runner-up 1.039) | `results/search/recipe.json` |
| Training throughput: per-expert loop → grouped_mm → + torch.compile | **356 → 675 → 1011 tokens/s (2.84×)**, 13.5% of BF16 peak; compile adopted (loss within 0.6%) | `results/trials/s4-perf-*`, `results/perf/defaults.json` |
| Custom Triton fused linear + CE (V=151,936) | exact loss, grad rel. err 5e-6; **2.1–2.35× faster, 7.0 → 0.34 GiB** at 4k tokens | `results/kernel_fused_ce_*.json` |
| Serving (vLLM 0.33), BF16 → FP8 at 16 concurrent | **113 → 203 tokens/s (1.79×)**, held-out loss +0.2% | `results/serve/` |

## Open work for the team (in priority order)
1. **Grouped expert GEMM in Triton for sm_121.** Expert GEMMs are small (about 256 tokens per expert per layer) and dominate step time. Register it in transformers' `ExpertsInterface` and set `experts_impl` in a trial config; compare against `grouped_mm` with `configs/perf-experts-*.json`. Biggest training lever (we're at 9% of peak).
2. **Tune vLLM's fused-MoE kernel for GB10.** The FP8 server logs `Using default MoE config ... Config file not found ... E=128,N=768,device_name=NVIDIA_GB10` (`results/serve/finding_moe_configs.json`). Run vLLM's `benchmarks/kernels/benchmark_moe.py --tune` (v0.33.0) for E=128, N=768 (auto + fp8_w8a8), save the JSONs to `/runs/serve/moe-configs`, serve with `VLLM_TUNED_CONFIG_FOLDER=/runs/serve/moe-configs`, and re-measure with `src/serve/serve_bench.py`.
3. **2-node data parallel.** Launcher fixed (see the morning checklist); check `s4-perf-dp2-*-2` and `s4-train-dp2-winner`. Fuzzball multinode pools hold at most 2 nodes.
4. **Enable the fused CE kernel in the pipeline** (`"fused_ce": true` in the train config) once `s4-perf-fused-ce` shows end-to-end gains. Same for torch.compile.
5. **Full pipeline run** with `train_tokens` 8M (`workflows/pipeline.yaml`, about 3.5 h on one node at 675 tok/s).

## How to run
- One trial: `python src/runner/runner.py configs/<trial>.json` (renders a named Fuzzball workflow, collects metrics into `results/trials/`).
- Recipe search: `python src/search/recipe_search.py --n 4 --budgets 250000,750000 --concurrency 4` (resume with `--continue-from-rung K --keep N`).
- Serving search: `scripts/fbrun s5-serve-<name> --image docker://vllm/vllm-openai:v0.33.0 ... -- python3 /opt/serve_bench.py --configs <json>` (see `results/serve/`).
- Final workflow: `python src/emit/emit_pipeline.py --upload` → `workflows/pipeline.yaml`, then `./fuzzball workflow start workflows/pipeline.yaml --name pipeline`.
- Cluster notes: jobs request 8 cores / 80 GiB (bigger requests sit in the queue on a busy cluster); GPU memory counts toward the container memory limit on GB10.
