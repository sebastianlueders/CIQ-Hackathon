"""Fused linear + cross-entropy for a 152k vocabulary (Triton, tested on GB10 / sm_121).

Standard path: logits = h @ W^T for all tokens (4096 x 151936), upcast to FP32 for the loss,
then a second FP32 tensor for its gradient: ~5 GB of temporaries per 4k-token sequence.

Fused path: process tokens in chunks; for each chunk compute BF16 logits, then ONE Triton
kernel per row does an online-softmax pass (max + sum-exp) and a second pass that writes
d(loss)/d(logits) in place over the logits buffer. The chunk's hidden-state gradient is
then logits_grad @ W. The lm_head is frozen under LoRA, so no weight gradient is needed.
Gradients are computed during forward (backward just scales them).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ce_fwd_grad_kernel(logits_ptr, stride, targets_ptr, loss_ptr, V, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ptr = logits_ptr + row * stride
    t = tl.load(targets_ptr + row)
    valid = t >= 0
    # pass 1: online max / sum-exp over the vocabulary
    m = float("-inf")
    d = 0.0
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(d)
    xt = tl.load(ptr + tl.where(valid, t, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - xt, 0.0))
    # pass 2: gradient (softmax - onehot) * scale, written in place
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.exp(x - lse)
        g = tl.where(cols == t, g - 1.0, g) * scale
        g = tl.where(valid, g, 0.0)
        tl.store(ptr + cols, g.to(logits_ptr.dtype.element_ty), mask=mask)


class FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, chunk=1024, block=4096):
        """h [N, H] (bf16), weight [V, H] (bf16, frozen), targets [N] (int64, -100 = ignore)."""
        N, V = h.shape[0], weight.shape[0]
        n_valid = max(int((targets >= 0).sum()), 1)
        loss = torch.empty(N, device=h.device, dtype=torch.float32)
        dh = torch.empty_like(h) if h.requires_grad else None
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            logits = h[s:e] @ weight.t()  # [c, V] bf16, reused as the gradient buffer
            _ce_fwd_grad_kernel[(e - s,)](logits, logits.stride(0), targets[s:e].contiguous(), loss[s:e],
                                          V, 1.0 / n_valid, BLOCK=block, num_warps=8)
            if dh is not None:
                dh[s:e] = logits @ weight
            del logits
        ctx.save_for_backward(dh)
        return loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad):
        (dh,) = ctx.saved_tensors
        return dh * grad.to(dh.dtype), None, None, None, None


def fused_linear_ce(h, weight, targets, chunk=1024):
    return FusedLinearCE.apply(h, weight, targets, chunk)


def reference_ce(h, weight, targets):
    logits = (h @ weight.t()).float()
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
