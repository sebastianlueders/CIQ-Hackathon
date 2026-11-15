"""Runtime detection for CUDA, Triton, and NVIDIA Grace Blackwell GB10 (sm_121)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareCapabilities:
    torch_available: bool
    cuda_available: bool
    triton_available: bool
    device_name: str
    compute_capability: tuple[int, int] | None
    sm_arch: str
    is_blackwell_sm121: bool
    active_kernel_backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_capabilities() -> HardwareCapabilities:
    torch_ok = False
    cuda_ok = False
    triton_ok = False
    device_name = "cpu"
    cc: tuple[int, int] | None = None
    sm_arch = "cpu"

    try:
        import torch

        torch_ok = True
        if torch.cuda.is_available():
            cuda_ok = True
            device_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            sm_arch = f"sm_{cc[0]}{cc[1]}"
    except ImportError:
        pass

    try:
        import triton  # type: ignore[import-not-found]  # noqa: F401

        triton_ok = True
    except ImportError:
        pass

    is_sm121 = cc == (12, 1)
    if cuda_ok and triton_ok and is_sm121:
        backend = "triton_sm121"
    elif cuda_ok and triton_ok:
        backend = f"triton_{sm_arch}"
    elif cuda_ok:
        backend = "pytorch_cuda_fallback"
    elif torch_ok:
        backend = "pytorch_cpu_fallback"
    else:
        backend = "numpy_cpu_fallback"

    return HardwareCapabilities(
        torch_available=torch_ok,
        cuda_available=cuda_ok,
        triton_available=triton_ok,
        device_name=device_name,
        compute_capability=cc,
        sm_arch=sm_arch,
        is_blackwell_sm121=is_sm121,
        active_kernel_backend=backend,
    )
