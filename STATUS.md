# Status (updated Sat Nov 14, ~3:55 PM CST)

An agent loop for local fine-tuning and serving on GB10 nodes via Fuzzball, with VAST-backed volumes.
Every number comes from `results/`; regenerate the table with `python src/report/make_report.py` (→ `results/summary.md`).

## Measured
| What | Value | Source |
|---|---|---|
| GB10 BF16 / FP8 GEMM peak (NGC PyTorch 26.10) | 99.1 / 198.6 TFLOP/s; 209 GB/s memory read | `results/env_probe_ngc.json` |
| VAST read, one stream | 1.64 GB/s | `results/vast_read.json` |
| Training data (LAMMPS) | 18.4M train / 0.9M held-out tokens, held out by file group | `results/ingest.json` |
| Model load per trial | ~10 min → **62 s** (fused-experts checkpoint + direct loader) | `results/fuse_experts.json`, smoke trials |
| Held-out loss, base → best after 250k tokens | **1.175 → 1.066** (ppl 3.24 → 2.90), lr 1e-3, rank 8 | `results/search/recipe_decisions.jsonl` |
| Training throughput, per-expert loop → grouped_mm | **356 → 675 tokens/s (1.90×)**, 9% of BF16 peak | `results/trials/s4-perf-experts-*` |
| Custom Triton fused linear + CE (V=151,936) | exact loss, grad rel. err 5e-6; **2.1–2.35× faster, 7.0 → 0.34 GiB** at 4k tokens | `results/kernel_fused_ce_*.json` |
| Serving (vLLM 0.33), BF16 → FP8 at 16 concurrent | **113 → 203 tokens/s (1.79×)**, held-out loss +0.2% | `results/serve/` |

## Running / automatic
- Recipe search round 2 (`s3-recipe-r1-*`): top 2 configs to 750k tokens → `results/search/recipe.json`.
- After that, automatically: re-emit `workflows/pipeline.yaml` and start `s6-pipeline-validate` (ingest → train 1M tokens → serve FP8 + LoRA and measure).
- Queued, low priority: `s4-perf-compile` (torch.compile per layer), `s4-perf-fused-ce` (the kernel inside real training).

## Open work for the team (in priority order)
1. **Grouped expert GEMM in Triton for sm_121.** Expert GEMMs are small (about 256 tokens per expert per layer) and dominate step time. Register it in transformers' `ExpertsInterface` and set `experts_impl` in a trial config; compare against `grouped_mm` with `configs/perf-experts-*.json`. Biggest training lever (we're at 9% of peak).
2. **Tune vLLM's fused-MoE kernel for GB10.** The FP8 server logs `Using default MoE config ... Config file not found ... E=128,N=768,device_name=NVIDIA_GB10` (`results/serve/finding_moe_configs.json`). Run vLLM's `benchmarks/kernels/benchmark_moe.py --tune` (v0.33.0) for E=128, N=768 (auto + fp8_w8a8), save the JSONs to `/runs/serve/moe-configs`, serve with `VLLM_TUNED_CONFIG_FOLDER=/runs/serve/moe-configs`, and re-measure with `src/serve/serve_bench.py`.
3. **2-node data parallel.** The code is ready (trainer + runner `nodes: 2`), but `s4-perf-dp2` hung at rendezvous: no output after `host=trial-0 master=trial-0 node_rank=0`. Check that `trial-0:29500` is reachable between the containers (try `--master-addr` as an IP, set `NCCL_SOCKET_IFNAME`). Fuzzball multinode pools hold at most 2 nodes.
4. **Enable the fused CE kernel in the pipeline** (`"fused_ce": true` in the train config) once `s4-perf-fused-ce` shows end-to-end gains. Same for torch.compile.
5. **Full pipeline run** with `train_tokens` 8M (`workflows/pipeline.yaml`, about 3.5 h on one node at 675 tok/s).

## How to run
- One trial: `python src/runner/runner.py configs/<trial>.json` (renders a named Fuzzball workflow, collects metrics into `results/trials/`).
- Recipe search: `python src/search/recipe_search.py --n 4 --budgets 250000,750000 --concurrency 4` (resume with `--continue-from-rung K --keep N`).
- Serving search: `scripts/fbrun s5-serve-<name> --image docker://vllm/vllm-openai:v0.33.0 ... -- python3 /opt/serve_bench.py --configs <json>` (see `results/serve/`).
- Final workflow: `python src/emit/emit_pipeline.py --upload` → `workflows/pipeline.yaml`, then `./fuzzball workflow start workflows/pipeline.yaml --name pipeline`.
- Cluster notes: jobs request 8 cores / 80 GiB (bigger requests sit in the queue on a busy cluster); GPU memory counts toward the container memory limit on GB10.
