# Status (updated Sat Nov 14, ~2:55 PM CST)

Agent loop for local fine-tuning and serving on 2× GB10 via Fuzzball, VAST-backed volumes.
Every number in the report comes from `results/` (`python src/report/make_report.py`).

## Measured so far
| What | Value | Source |
|---|---|---|
| BF16 / FP8 GEMM peak (NGC PyTorch 26.10) | 99.1 / 198.6 TFLOP/s | `results/env_probe_ngc.json` |
| GPU memory read bandwidth | 209 GB/s | same |
| VAST read (one stream) | 1.64 GB/s | `results/vast_read.json` |
| Training data (LAMMPS) | 18.4M train / 0.9M held-out tokens, group-level split | `results/ingest.json` |
| Model load per trial | ~10 min (from_pretrained) → **61.8 s** (fused checkpoint + direct loader) | `results/trials/s2-smoke-direct*` |
| Base model held-out loss (eval_small) | 1.175 (ppl 3.24) | smoke trial |
| LoRA training throughput (1 node, seq 4096) | ~670 tokens/s, peak 64.8 GiB | smoke trial |

## Running (Fuzzball workflow names)
- `s3-recipe-*`: recipe search, 4 configs × 250k tokens → best 2 to 750k (successive halving). Decisions: `results/search/recipe_decisions.jsonl`.
- `s5-serve-base`: vLLM 0.33 serving search, BF16 vs FP8: tok/s at C=1/16 and held-out loss.
- Next, automatically: `s4-perf-experts-{grouped,eager,batched}`, comparing training throughput across MoE expert implementations.

## How to run things
- Trial: `python src/runner/runner.py configs/<trial>.json` (renders a Fuzzfile, launches, collects metrics into `results/trials/`).
- Recipe search: `python src/search/recipe_search.py --n 4 --budgets 250000,750000`.
- Final workflow: `python src/emit/emit_pipeline.py [--upload]` → `workflows/pipeline.yaml`.
- One-off jobs with a name: `scripts/fbrun <name> <fuzzball run flags>`.

## Not done yet (owners welcome)
1. **Custom kernels** (Triton, sm_121; Triton works in the NGC image): fused chunked cross-entropy, then a grouped expert GEMM registered via transformers' `ExpertsInterface` (`config._experts_implementation`), compared against `grouped_mm` with `configs/perf-experts-*.json`.
2. **Serving with the fine-tuned adapter** (vLLM `--enable-lora`) and NVFP4.
3. **Run `workflows/pipeline.yaml` end to end** after re-emitting it with the search winners.
4. Fill the one-pager table from `results/summary.md`.
