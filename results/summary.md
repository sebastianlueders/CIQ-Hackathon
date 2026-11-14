| Result | Value |
|---|---|
| Held-out LAMMPS loss (perplexity), base -> fine-tuned | 1.175 (3.24) -> not measured (not measured) |
| Training throughput, eager experts -> best (grouped) | 356 -> 675 tokens/s (1.90x) |
| Utilization vs measured BF16 peak (4 x active params FLOPs/token) | 9.0% of 99.1 TFLOP/s |
| Serving bf16: out tok/s at C=1 / C=16; held-out loss | 28.4 / 113.4; 1.29053 |
| Serving fp8: out tok/s at C=1 / C=16; held-out loss | 47.4 / 203.3; 1.2934 |

Formula: utilization = tokens/s x 4 x 3.3e+09 FLOPs / measured BF16 GEMM peak (results/env_probe_ngc.json). Base and fine-tuned loss use the same held-out eval_small tokens.
