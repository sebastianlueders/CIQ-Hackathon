"""Kernel 1: Fused / Chunked Cross-Entropy over 152k Vocabulary for GB10 (sm_121).

Problem on GB10:
- Qwen3-Coder-30B-A3B has vocab_size = 151,936. Materializing full FP32 `[N, V]` logits consumes
  `N * 151936 * 4` bytes (~4.9 GB at B=4, T=2048), starving GB10 unified LPDDR5X bandwidth.
- We tile across vocabulary blocks, maintaining online `(running_max, running_sum_exp)` statistics
  (using a Triton row-reduction kernel when on CUDA/sm_121, or tiled PyTorch online LSE fallback)
  and recomputing chunk probabilities during backward so `[N, V]` is never materialized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _online_lse_chunk_sm121_kernel(
        logits_ptr,
        stride_row,
        m_ptr,
        s_ptr,
        n_cols,
        BLOCK_V: tl.constexpr,
    ):
        """Updates running `(m, s)` online softmax statistics for one vocab chunk on sm_121."""
        row = tl.program_id(0)
        row_start = logits_ptr + row * stride_row
        m_prev = tl.load(m_ptr + row)
        s_prev = tl.load(s_ptr + row)

        for offset in range(0, n_cols, BLOCK_V):
            cols = offset + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            vals = tl.load(row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
            block_max = tl.max(vals, axis=0)
            m_new = tl.maximum(m_prev, block_max)
            s_prev = s_prev * tl.exp(m_prev - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
            m_prev = m_new

        tl.store(m_ptr + row, m_prev)
        tl.store(s_ptr + row, s_prev)


def _update_online_lse(
    chunk_logits: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if HAS_TRITON and chunk_logits.is_cuda:
        n_rows, n_cols = chunk_logits.shape
        _online_lse_chunk_sm121_kernel[(n_rows,)](
            chunk_logits,
            chunk_logits.stride(0),
            running_max,
            running_sum,
            n_cols,
            BLOCK_V=1024,
            num_warps=8,
            num_stages=3,
        )
        return running_max, running_sum

    chunk_f32 = chunk_logits.float()
    chunk_max = chunk_f32.amax(dim=-1)
    new_max = torch.maximum(running_max, chunk_max)
    new_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
        chunk_f32 - new_max.unsqueeze(-1)
    ).sum(dim=-1)
    return new_max, new_sum


class _ChunkedCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        vocab_chunk_size: int,
        ignore_index: int,
    ) -> torch.Tensor:
        orig_dtype = hidden.dtype
        flat_h = hidden.reshape(-1, hidden.shape[-1])
        flat_t = targets.reshape(-1)
        n_tokens = flat_h.shape[0]
        vocab_size = lm_head_weight.shape[0]

        valid_mask = flat_t != ignore_index
        valid_count = valid_mask.sum().clamp(min=1)

        running_max = torch.full((n_tokens,), -float("inf"), dtype=torch.float32, device=hidden.device)
        running_sum = torch.zeros((n_tokens,), dtype=torch.float32, device=hidden.device)
        target_logit = torch.zeros((n_tokens,), dtype=torch.float32, device=hidden.device)

        for v_start in range(0, vocab_size, vocab_chunk_size):
            v_end = min(v_start + vocab_chunk_size, vocab_size)
            w_chunk = lm_head_weight[v_start:v_end]
            chunk_logits = flat_h @ w_chunk.t()
            running_max, running_sum = _update_online_lse(chunk_logits, running_max, running_sum)

            in_chunk = valid_mask & (flat_t >= v_start) & (flat_t < v_end)
            if in_chunk.any():
                rows = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                cols = flat_t[rows] - v_start
                target_logit[rows] = chunk_logits[rows, cols].float()

        lse = running_max + torch.log(running_sum)
        per_token_nll = torch.where(valid_mask, lse - target_logit, torch.zeros_like(lse))
        loss = per_token_nll.sum() / valid_count.float()

        ctx.save_for_backward(flat_h, lm_head_weight, flat_t, lse, valid_mask)
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.orig_shape = hidden.shape
        ctx.orig_dtype = orig_dtype
        ctx.valid_count = int(valid_count.item())
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        flat_h, lm_head_weight, flat_t, lse, valid_mask = ctx.saved_tensors
        vocab_chunk_size: int = ctx.vocab_chunk_size
        vocab_size = lm_head_weight.shape[0]
        scale = grad_output.float() / float(ctx.valid_count)

        need_h_grad = ctx.needs_input_grad[0]
        need_w_grad = ctx.needs_input_grad[1]

        grad_h = torch.zeros_like(flat_h, dtype=torch.float32) if need_h_grad else None
        grad_w = torch.zeros_like(lm_head_weight, dtype=torch.float32) if need_w_grad else None
        flat_h_f32 = flat_h.float()

        for v_start in range(0, vocab_size, vocab_chunk_size):
            v_end = min(v_start + vocab_chunk_size, vocab_size)
            w_chunk_f32 = lm_head_weight[v_start:v_end].float()
            chunk_logits = flat_h_f32 @ w_chunk_f32.t()
            probs = torch.exp(chunk_logits - lse.unsqueeze(-1))
            probs = torch.where(valid_mask.unsqueeze(-1), probs, torch.zeros_like(probs))

            in_chunk = valid_mask & (flat_t >= v_start) & (flat_t < v_end)
            if in_chunk.any():
                rows = torch.nonzero(in_chunk, as_tuple=False).squeeze(-1)
                cols = flat_t[rows] - v_start
                probs[rows, cols] -= 1.0

            d_logits = probs * scale
            if grad_h is not None:
                grad_h.add_(d_logits @ w_chunk_f32)
            if grad_w is not None:
                grad_w[v_start:v_end].add_(d_logits.t() @ flat_h_f32)

        out_grad_h = grad_h.to(ctx.orig_dtype).reshape(ctx.orig_shape) if grad_h is not None else None
        out_grad_w = grad_w.to(lm_head_weight.dtype) if grad_w is not None else None
        return out_grad_h, out_grad_w, None, None, None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    vocab_chunk_size: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-saving online-LSE cross-entropy over `[..., H]` hidden states and `[V, H]` head."""
    return _ChunkedCrossEntropyFn.apply(
        hidden, lm_head_weight, targets, vocab_chunk_size, ignore_index
    )


def reference_cross_entropy(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Materialized PyTorch baseline (`F.linear` + `F.cross_entropy` in FP32)."""
    logits = F.linear(hidden, lm_head_weight).float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
