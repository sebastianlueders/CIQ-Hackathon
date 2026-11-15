"""Correctness + speed: Triton grouped expert GEMM vs torch._grouped_mm vs a per-expert loop.

Shapes are Qwen3-Coder-30B-A3B's experts (E=128, top-8, hidden 2048, expert dim 768) for one
4096-token micro-batch (S = 32768 routed rows) and 8192 tokens. Prints RESULT lines.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from grouped_gemm import grouped_linear  # noqa: E402


def out(**kw): print("RESULT " + json.dumps(kw), flush=True)


def routed(tokens, E=128, k=8, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, device="cuda", generator=g)[:k] for _ in range(tokens)]).reshape(-1)
    ids, _ = torch.sort(ids)
    offs = torch.cumsum(torch.bincount(ids, minlength=E), 0).to(torch.int32)
    return offs


def loop_ref(x, w, offs):
    y = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=torch.float32); s = 0
    for e, end in enumerate(offs.tolist()):
        y[s:end] = x[s:end].float() @ w[e].float().t(); s = end
    return y


def torch_grouped(x, w, offs):
    f = getattr(torch.nn.functional, "grouped_mm", None) or torch._grouped_mm
    return f(x, w.transpose(-2, -1), offs=offs)


def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


def main():
    torch.manual_seed(0)
    E, H, I = 128, 2048, 768
    w_up = (torch.randn(E, 2 * I, H, device="cuda") * 0.02).to(torch.bfloat16)   # gate_up_proj [E, 2I, H]
    w_dn = (torch.randn(E, H, I, device="cuda") * 0.02).to(torch.bfloat16)       # down_proj    [E, H, I]
    # correctness (forward and dX) at a small size
    offs = routed(512)
    S = int(offs[-1])
    x = torch.randn(S, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = grouped_linear(x, w_up, offs)
    ref = loop_ref(x.detach(), w_up, offs)
    fwd_err = ((y.float() - ref).norm() / ref.norm()).item()
    dy = torch.randn_like(y)
    y.backward(dy)
    ref_dx = torch.empty(S, H, device="cuda"); s = 0
    for e, end in enumerate(offs.tolist()):
        ref_dx[s:end] = dy[s:end].float() @ w_up[e].float(); s = end
    bwd_err = ((x.grad.float() - ref_dx).norm() / ref_dx.norm()).item()
    out(kind="kernel_correctness", kernel="grouped_gemm", rows=S, fwd_rel_err=round(fwd_err, 6),
        dx_rel_err=round(bwd_err, 6), ok=fwd_err < 1e-2 and bwd_err < 1e-2)
    # speed at training shapes
    for tokens in (4096, 8192):
        offs = routed(tokens, seed=1)
        S = int(offs[-1])
        for name, w, K in (("up (gate_up)", w_up, H), ("down", w_dn, I)):
            xx = torch.randn(S, K, device="cuda", dtype=torch.bfloat16)
            flops = 2 * S * K * w.shape[1]
            t_tri = timeit(lambda: grouped_linear(xx, w, offs))
            try:
                t_tg = timeit(lambda: torch_grouped(xx, w, offs))
            except Exception as ex:
                t_tg = None; out(kind="note", msg=f"torch grouped_mm failed: {str(ex)[:200]}")
            t_loop = timeit(lambda: loop_ref(xx, w, offs), iters=3)
            out(kind="kernel_bench", kernel="grouped_gemm", tokens=tokens, rows=S, proj=name,
                triton_ms=round(t_tri * 1e3, 3), torch_grouped_ms=t_tg and round(t_tg * 1e3, 3),
                loop_ms=round(t_loop * 1e3, 3),
                triton_tflops=round(flops / t_tri / 1e12, 1), torch_grouped_tflops=t_tg and round(flops / t_tg / 1e12, 1),
                speedup_vs_torch_grouped=t_tg and round(t_tg / t_tri, 2))


if __name__ == "__main__":
    main()
