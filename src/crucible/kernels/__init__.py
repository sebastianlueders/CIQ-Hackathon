"""Custom Triton/CUDA kernels for NVIDIA GB10 (sm_121) with exact PyTorch fallbacks."""

from crucible.kernels.benchmark import KernelBenchmarkResult, run_kernel_suite
from crucible.kernels.capabilities import HardwareCapabilities, detect_capabilities
from crucible.kernels.chunked_cross_entropy import chunked_cross_entropy, reference_cross_entropy
from crucible.kernels.fused_lora import FusedLoRALinear, fused_lora_linear, reference_lora_linear
from crucible.kernels.grouped_moe_gemm import grouped_moe_forward, reference_moe_forward
from crucible.kernels.low_prec_decode import (
    QuantizedWeightTensor,
    low_precision_decode_linear,
    quantize_weights,
)

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
]
