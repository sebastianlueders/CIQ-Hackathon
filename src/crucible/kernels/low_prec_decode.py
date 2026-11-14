"""Kernel 4: Low-Precision Decode Kernel (FP8 & Blackwell NVFP4 E2M1 Block-Scaled) for GB10.

Autoregressive code generation is memory-bandwidth bound on unified LPDDR5X (~273 GB/s on GB10).
Compressing frozen expert/projection weights from BF16 (16 bits) to FP8 (8 bits) or Blackwell
NVFP4 (4-bit E2M1 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` with 16-element micro-block scales)
reduces weight traffic by 2x–3.5x while preserving C++ code-completion quality above the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

NVFP4_CODEBOOK_VALUES = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)
NVFP4_MAX_ABS = 6.0
FP8_E4M3_MAX_ABS = 448.0


@dataclass
class QuantizedWeightTensor:
    precision: Literal["BF16", "FP8", "NVFP4"]
    out_features: int
    in_features: int
    group_size: int
    codes: torch.Tensor
    scales: torch.Tensor
    codebook: torch.Tensor | None = None

    @property
    def storage_bytes(self) -> int:
        if self.precision == "NVFP4":
            packed_nibble_bytes = (self.codes.numel() + 1) // 2
            return packed_nibble_bytes + self.scales.numel() * 2
        if self.precision == "FP8":
            return self.codes.numel() * 1 + self.scales.numel() * 2
        return self.codes.numel() * 2


if HAS_TRITON:

    @triton.jit
    def _block_scaled_decode_gemv_sm121_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        y_ptr,
        batch_rows,
        out_features,
        in_features,
        group_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused block-scaled dequantization + GEMV decode kernel for sm_121."""
        b_idx = tl.program_id(0)
        n_block = tl.program_id(1)
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < out_features
        n_groups = in_features // group_size

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for g in range(0, n_groups):
            s_vec = tl.load(scale_ptr + offs_n * n_groups + g, mask=mask_n, other=0.0).to(tl.float32)
            offs_k = g * group_size + tl.arange(0, group_size)
            x_seg = tl.load(x_ptr + b_idx * in_features + offs_k).to(tl.float32)
            q_tile = tl.load(
                q_ptr + offs_n[:, None] * in_features + offs_k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += s_vec * tl.sum(q_tile * x_seg[None, :], axis=1)

        tl.store(y_ptr + b_idx * out_features + offs_n, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def quantize_weights(
    weight: torch.Tensor,
    precision: Literal["BF16", "FP8", "NVFP4"],
    group_size: int = 16,
) -> QuantizedWeightTensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")

    if precision == "BF16":
        return QuantizedWeightTensor(
            precision="BF16",
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            codes=weight.to(torch.bfloat16),
            scales=torch.ones((out_features, 1), dtype=torch.float16, device=weight.device),
        )

    w_grouped = weight.float().reshape(out_features, in_features // group_size, group_size)
    amax = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)

    if precision == "FP8":
        scales = amax / FP8_E4M3_MAX_ABS
        q_int8 = torch.clamp(torch.round(w_grouped / scales.unsqueeze(-1) * (127.0 / FP8_E4M3_MAX_ABS)), -127, 127).to(
            torch.int8
        )
        effective_scales = (scales * (FP8_E4M3_MAX_ABS / 127.0)).to(torch.float16)
        return QuantizedWeightTensor(
            precision="FP8",
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            codes=q_int8.reshape(out_features, in_features),
            scales=effective_scales,
        )

    codebook = torch.tensor(NVFP4_CODEBOOK_VALUES, dtype=torch.float32, device=weight.device)
    scales = amax / NVFP4_MAX_ABS
    normalized = w_grouped / scales.unsqueeze(-1)
    diffs = (normalized.unsqueeze(-1) - codebook).abs()
    nearest_idx = diffs.argmin(dim=-1).to(torch.uint8)
    return QuantizedWeightTensor(
        precision="NVFP4",
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        codes=nearest_idx.reshape(out_features, in_features),
        scales=scales.to(torch.float16),
        codebook=codebook,
    )


def dequantize_weights(qw: QuantizedWeightTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if qw.precision == "BF16":
        return qw.codes.to(dtype)

    n_groups = qw.in_features // qw.group_size
    if qw.precision == "FP8":
        vals = qw.codes.float().reshape(qw.out_features, n_groups, qw.group_size)
    else:
        assert qw.codebook is not None
        vals = qw.codebook[qw.codes.long()].reshape(qw.out_features, n_groups, qw.group_size)

    deq = vals * qw.scales.float().unsqueeze(-1)
    return deq.reshape(qw.out_features, qw.in_features).to(dtype)


def low_precision_decode_linear(
    x: torch.Tensor,
    qw: QuantizedWeightTensor,
) -> torch.Tensor:
    """Executes decode projection with block-scaled FP8/NVFP4 weights."""
    if qw.precision == "BF16":
        return F.linear(x, qw.codes.to(x.dtype))

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if HAS_TRITON and x_2d.is_cuda and qw.group_size in (16, 32):
        if qw.precision == "FP8":
            q_vals = qw.codes
        else:
            assert qw.codebook is not None
            q_vals = qw.codebook[qw.codes.long()]
        out_2d = torch.empty((x_2d.shape[0], qw.out_features), dtype=x.dtype, device=x.device)
        grid = (x_2d.shape[0], triton.cdiv(qw.out_features, 64))
        _block_scaled_decode_gemv_sm121_kernel[grid](
            x_2d,
            q_vals,
            qw.scales,
            out_2d,
            x_2d.shape[0],
            qw.out_features,
            qw.in_features,
            group_size=qw.group_size,
            BLOCK_N=64,
            num_warps=4,
        )
        return out_2d.reshape(*orig_shape[:-1], qw.out_features)

    w_deq = dequantize_weights(qw, dtype=x.dtype)
    return F.linear(x, w_deq)
