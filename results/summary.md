| Result | Value |
|---|---|
| Held-out LAMMPS loss (perplexity), base -> fine-tuned | 1.175 (3.24) -> 1.037 (2.82) |
| Recipe search: configs tried / stopped early / winner | 4 / 3 / c1-lr1e-3-r8 |
| Training throughput, eager experts -> best (grouped) | 356 -> 675 tokens/s (1.90x) |
| Training: compile vs grouped_mm baseline (adopted=True) | 1011 tokens/s (1.49x), loss within 0.6% |
| Training throughput, stock (eager experts) -> adopted optimizations | 356 -> 1011 tokens/s (2.84x) |
| Utilization vs measured BF16 peak, with adopted optimizations | 13.5% |
| Utilization vs measured BF16 peak (4 x active params FLOPs/token) | 9.0% of 99.1 TFLOP/s |
| Serving bf16: out tok/s at C=1 / C=16; held-out loss | 28.4 / 113.4; 1.29053 |
| Serving fp8: out tok/s at C=1 / C=16; held-out loss | 47.4 / 203.3; 1.2934 |
| Custom Triton fused linear+CE (V=151,936): correctness vs FP32 PyTorch | loss err 0.0, grad rel err 5e-06 (pass) |
| Fused linear+CE at N=4096: time / peak memory vs PyTorch | 214.93 -> 102.17 ms (2.1x); 6.96 -> 0.34 GiB |
| Fused linear+CE at N=8192: time / peak memory vs PyTorch | 481.4 -> 204.85 ms (2.35x); 13.91 -> 0.36 GiB |
| 2-node data-parallel training | not measured (rendezvous between nodes never connected) |

Formula: utilization = tokens/s x 4 x 3.3e+09 FLOPs / measured BF16 GEMM peak (results/env_probe_ngc.json). Base and fine-tuned loss use the same held-out eval_small tokens.
