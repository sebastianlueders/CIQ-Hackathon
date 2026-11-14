"""Correctness test + benchmark: fused linear+CE vs PyTorch, at the trainer's real shapes.

Prints RESULT lines: correctness (loss / grad error vs FP32 reference) and, for N tokens,
time per forward+backward and peak memory for both paths.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from fused_ce import fused_linear_ce, reference_ce  # noqa: E402


def out(**kw): print("RESULT " + json.dumps(kw), flush=True)


def run(fn, h, W, t):
    h = h.detach().requires_grad_(True)
    loss = fn(h, W, t); loss.backward()
    return loss.detach(), h.grad


def bench(fn, h, W, t, iters=5):
    run(fn, h, W, t); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(iters): run(fn, h, W, t)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    torch.manual_seed(0)
    H, V = 2048, 151936  # Qwen3-Coder-30B-A3B hidden size and vocabulary
    W = (torch.randn(V, H, device="cuda") * 0.02).to(torch.bfloat16)
    # correctness at a small N, against an FP32 reference
    N = 512
    h = torch.randn(N, H, device="cuda").to(torch.bfloat16)
    t = torch.randint(0, V, (N,), device="cuda"); t[::17] = -100
    l_ref, g_ref = run(reference_ce, h, W, t)
    l_f, g_f = run(fused_linear_ce, h, W, t)
    out(kind="kernel_correctness", kernel="fused_linear_ce", N=N, V=V,
        loss_ref=round(l_ref.item(), 6), loss_fused=round(l_f.item(), 6),
        loss_abs_err=abs(l_ref.item() - l_f.item()),
        grad_rel_err=round(((g_ref.float() - g_f.float()).norm() / g_ref.float().norm()).item(), 6),
        ok=abs(l_ref.item() - l_f.item()) < 1e-2 and ((g_ref.float() - g_f.float()).norm() / g_ref.float().norm()).item() < 2e-2)
    # benchmark at the trainer's shape: one 4096-token sequence
    for N in (4096, 8192):
        h = torch.randn(N, H, device="cuda").to(torch.bfloat16)
        t = torch.randint(0, V, (N,), device="cuda")
        ms_r, gb_r = bench(reference_ce, h, W, t)
        ms_f, gb_f = bench(fused_linear_ce, h, W, t)
        out(kind="kernel_bench", kernel="fused_linear_ce", N=N, V=V,
            torch_ms=round(ms_r, 2), fused_ms=round(ms_f, 2), speedup=round(ms_r / ms_f, 2),
            torch_peak_gib=round(gb_r, 2), fused_peak_gib=round(gb_f, 2),
            mem_saved_gib=round(gb_r - gb_f, 2))


if __name__ == "__main__":
    main()
