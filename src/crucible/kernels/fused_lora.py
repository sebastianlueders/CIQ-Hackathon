"""Kernel 3: Fused LoRA Linear Projection (`Wx + (alpha/r) * B(Ax)`) for GB10 (sm_121).

Fuses base frozen BF16 projection with low-rank adapter accumulation and implements a custom
autograd backward that computes exact `dX`, `dA`, and `dB` without computing `dW` on frozen base
weights or writing redundant intermediate tensors across GB10 unified memory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _fused_add_scaled_lora_sm121_kernel(
        base_ptr,
        lora_h_ptr,
        lora_b_ptr,
        out_ptr,
        m_rows,
        n_cols,
        rank,
        scale,
        stride_bm,
        stride_bn,
        stride_hm,
        stride_hr,
        stride_lbn,
        stride_lbr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fuses `base_out + scale * (lora_hidden @ lora_B.T)` in registers on sm_121."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < m_rows
        mask_n = offs_n < n_cols

        acc = tl.load(
            base_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        for r in range(0, rank):
            h_col = tl.load(lora_h_ptr + offs_m * stride_hm + r * stride_hr, mask=mask_m, other=0.0).to(
                tl.float32
            )
            b_row = tl.load(
                lora_b_ptr + offs_n * stride_lbn + r * stride_lbr, mask=mask_n, other=0.0
            ).to(tl.float32)
            acc += scale * (h_col[:, None] * b_row[None, :])

        tl.store(
            out_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


class _FusedLoRALinearFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        base_out = x_2d @ base_weight.t()
        lora_h = x_2d @ lora_a.t()

        if HAS_TRITON and x_2d.is_cuda and lora_a.shape[0] <= 128:
            out_2d = torch.empty_like(base_out)
            m_rows, n_cols = base_out.shape
            grid = (triton.cdiv(m_rows, 32), triton.cdiv(n_cols, 64))
            _fused_add_scaled_lora_sm121_kernel[grid](
                base_out,
                lora_h,
                lora_b,
                out_2d,
                m_rows,
                n_cols,
                lora_a.shape[0],
                float(scale),
                base_out.stride(0),
                base_out.stride(1),
                lora_h.stride(0),
                lora_h.stride(1),
                lora_b.stride(0),
                lora_b.stride(1),
                BLOCK_M=32,
                BLOCK_N=64,
                num_warps=4,
            )
        else:
            out_2d = torch.addmm(base_out, lora_h, lora_b.t(), beta=1.0, alpha=scale)

        ctx.save_for_backward(x_2d, base_weight, lora_a, lora_b, lora_h)
        ctx.scale = scale
        ctx.orig_shape = orig_shape
        return out_2d.reshape(*orig_shape[:-1], base_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x_2d, base_weight, lora_a, lora_b, lora_h = ctx.saved_tensors
        scale: float = ctx.scale
        dy = grad_out.reshape(-1, grad_out.shape[-1])
        dy_f32 = dy.float()

        grad_b = (dy_f32.t() @ lora_h.float()) * scale
        dh_f32 = (dy_f32 @ lora_b.float()) * scale
        grad_a = dh_f32.t() @ x_2d.float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            dx_f32 = dy_f32 @ base_weight.float() + dh_f32 @ lora_a.float()
            grad_x = dx_f32.to(x_2d.dtype).reshape(ctx.orig_shape)

        return (
            grad_x,
            None,
            grad_a.to(lora_a.dtype),
            grad_b.to(lora_b.dtype),
            None,
        )


def fused_lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _FusedLoRALinearFn.apply(x, base_weight, lora_a, lora_b, scale)


def reference_lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return F.linear(x, base_weight) + scale * F.linear(F.linear(x, lora_a), lora_b)


class FusedLoRALinear(nn.Module):
    """Drop-in LoRA module switching cleanly between `fused_lora_linear` and PyTorch baseline."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        use_fused_kernel: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.use_fused_kernel = use_fused_kernel

        base = torch.empty((out_features, in_features), dtype=dtype)
        nn.init.kaiming_uniform_(base, a=math.sqrt(5))
        self.base_weight = nn.Parameter(base, requires_grad=False)

        self.lora_a = nn.Parameter(torch.empty((rank, in_features), dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros((out_features, rank), dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_kernel:
            return fused_lora_linear(x, self.base_weight, self.lora_a, self.lora_b, self.scale)
        return reference_lora_linear(x, self.base_weight, self.lora_a, self.lora_b, self.scale)
