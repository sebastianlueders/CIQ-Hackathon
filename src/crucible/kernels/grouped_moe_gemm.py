"""Kernel 2: Grouped Expert GEMM for Qwen3-Coder-30B-A3B MoE Routing on GB10 (sm_121).

Problem on GB10:
- Qwen3-Coder-30B-A3B routes each token to top-8 out of 128 fine-grained experts.
- Naive Python loops issue up to 128 tiny GEMMs per MoE layer, leaving Grace Blackwell SM
  pipelines under-occupied.
- We permute `(token, k)` assignments into contiguous expert buckets and execute either a
  ragged Triton grouped-GEMM kernel (`sm_121` 64x64x64 tiles) on GPU or a coalesced sorted
  expert projection on CPU/fallback, weighted by router probabilities.
"""

from __future__ import annotations

import torch

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _grouped_expert_linear_sm121_kernel(
        x_sorted_ptr,
        w_experts_ptr,
        y_sorted_ptr,
        expert_ids_ptr,
        total_assignments,
        in_features,
        out_features,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Per-assignment fused expert projection kernel tailored for ragged MoE routing."""
        row_idx = tl.program_id(0)
        col_block = tl.program_id(1)
        if row_idx >= total_assignments:
            return

        expert_id = tl.load(expert_ids_ptr + row_idx)
        offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < out_features

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, in_features, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < in_features
            x_vec = tl.load(
                x_sorted_ptr + row_idx * stride_xm + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            w_tile = tl.load(
                w_experts_ptr
                + expert_id * stride_we
                + offs_n[:, None] * stride_wn
                + offs_k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w_tile * x_vec[None, :], axis=1)

        tl.store(
            y_sorted_ptr + row_idx * stride_ym + offs_n * stride_yn,
            acc.to(y_sorted_ptr.dtype.element_ty),
            mask=mask_n,
        )


def grouped_moe_forward(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_gates: torch.Tensor,
) -> torch.Tensor:
    """Coalesced grouped MoE projection.

    Args:
        hidden: `[N, D_in]`
        expert_weights: `[E, D_out, D_in]`
        topk_indices: `[N, K]` integer expert IDs in `[0, E)`
        topk_gates: `[N, K]` normalized routing weights
    Returns:
        `[N, D_out]` combined expert outputs.
    """
    n_tokens, d_in = hidden.shape
    num_experts, d_out, _ = expert_weights.shape
    top_k = topk_indices.shape[1]

    flat_experts = topk_indices.reshape(-1)
    sort_order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sort_order]
    token_ids = torch.div(sort_order, top_k, rounding_mode="floor")
    sorted_x = hidden.index_select(0, token_ids).contiguous()
    sorted_gates = topk_gates.reshape(-1).index_select(0, sort_order)

    total_assignments = sorted_x.shape[0]
    if HAS_TRITON and hidden.is_cuda:
        sorted_y = torch.empty((total_assignments, d_out), dtype=hidden.dtype, device=hidden.device)
        grid = (total_assignments, triton.cdiv(d_out, 64))
        _grouped_expert_linear_sm121_kernel[grid](
            sorted_x,
            expert_weights,
            sorted_y,
            sorted_experts,
            total_assignments,
            d_in,
            d_out,
            sorted_x.stride(0),
            sorted_x.stride(1),
            expert_weights.stride(0),
            expert_weights.stride(1),
            expert_weights.stride(2),
            sorted_y.stride(0),
            sorted_y.stride(1),
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
        )
    else:
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        sorted_y = torch.empty((total_assignments, d_out), dtype=hidden.dtype, device=hidden.device)
        offset = 0
        for e_idx in range(num_experts):
            cnt = int(counts[e_idx].item())
            if cnt == 0:
                continue
            seg_x = sorted_x[offset : offset + cnt]
            sorted_y[offset : offset + cnt] = seg_x @ expert_weights[e_idx].t()
            offset += cnt

    weighted_y = sorted_y.float() * sorted_gates.float().unsqueeze(-1)
    out = torch.zeros((n_tokens, d_out), dtype=torch.float32, device=hidden.device)
    out.index_add_(0, token_ids, weighted_y)
    return out.to(hidden.dtype)


def reference_moe_forward(
    hidden: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_gates: torch.Tensor,
) -> torch.Tensor:
    """Unoptimized per-token, per-slot PyTorch baseline for numerical verification."""
    n_tokens = hidden.shape[0]
    d_out = expert_weights.shape[1]
    top_k = topk_indices.shape[1]
    out = torch.zeros((n_tokens, d_out), dtype=torch.float32, device=hidden.device)
    for t in range(n_tokens):
        x_t = hidden[t].float()
        for k in range(top_k):
            e_id = int(topk_indices[t, k].item())
            gate = float(topk_gates[t, k].item())
            out[t] += gate * (expert_weights[e_id].float() @ x_t)
    return out.to(hidden.dtype)
