"""Custom Triton/CUDA kernels for NVIDIA GB10 (sm_121) with lazy PyTorch/NumPy fallbacks."""

from __future__ import annotations

from typing import Any

from crucible.kernels.benchmark import (
    KernelBenchmarkResult,
    run_kernel_suite,
    verify_training_loss_curve_equivalence,
)
from crucible.kernels.capabilities import HardwareCapabilities, detect_capabilities

__all__ = [
    "FusedLoRALinear",
    "HardwareCapabilities",
    "KernelBenchmarkResult",
    "QuantizedWeightTensor",
    "chunked_cross_entropy",
    "detect_capabilities",
    "fused_lora_linear",
    "grouped_moe_forward",
    "low_precision_decode_linear",
    "quantize_weights",
    "reference_cross_entropy",
    "reference_lora_linear",
    "reference_moe_forward",
    "run_kernel_suite",
    "verify_training_loss_curve_equivalence",
]

_LAZY_TORCH_EXPORTS: dict[str, tuple[str, str]] = {
    "chunked_cross_entropy": ("crucible.kernels.chunked_cross_entropy", "chunked_cross_entropy"),
    "reference_cross_entropy": ("crucible.kernels.chunked_cross_entropy", "reference_cross_entropy"),
    "FusedLoRALinear": ("crucible.kernels.fused_lora", "FusedLoRALinear"),
    "fused_lora_linear": ("crucible.kernels.fused_lora", "fused_lora_linear"),
    "reference_lora_linear": ("crucible.kernels.fused_lora", "reference_lora_linear"),
    "grouped_moe_forward": ("crucible.kernels.grouped_moe_gemm", "grouped_moe_forward"),
    "reference_moe_forward": ("crucible.kernels.grouped_moe_gemm", "reference_moe_forward"),
    "QuantizedWeightTensor": ("crucible.kernels.low_prec_decode", "QuantizedWeightTensor"),
    "low_precision_decode_linear": (
        "crucible.kernels.low_prec_decode",
        "low_precision_decode_linear",
    ),
    "quantize_weights": ("crucible.kernels.low_prec_decode", "quantize_weights"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_TORCH_EXPORTS:
        import importlib

        mod_name, attr_name = _LAZY_TORCH_EXPORTS[name]
        return getattr(importlib.import_module(mod_name), attr_name)
    raise AttributeError(f"module 'crucible.kernels' has no attribute {name!r}")
