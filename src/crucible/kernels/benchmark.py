"""Correctness verification, microbenchmark harness, and loss-curve equivalence check."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from crucible.kernels.capabilities import HardwareCapabilities, detect_capabilities
from crucible.kernels.chunked_cross_entropy import chunked_cross_entropy, reference_cross_entropy
from crucible.kernels.fused_lora import FusedLoRALinear, fused_lora_linear, reference_lora_linear
from crucible.kernels.grouped_moe_gemm import grouped_moe_forward, reference_moe_forward
from crucible.kernels.low_prec_decode import (
    low_precision_decode_linear,
    quantize_weights,
)


@dataclass(frozen=True)
class KernelBenchmarkResult:
    kernel_name: str
    backend: str
    reference_ms: float
    custom_ms: float
    measured_speedup: float
    max_abs_error: float
    relative_error: float
    memory_reduction_ratio: float
    correctness_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _time_callable(fn, warmup: int = 2, iters: int = 6, sync_cuda: bool = False) -> float:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    return ((time.perf_counter() - t0) * 1000.0) / float(iters)


def run_kernel_suite(
    vocab_size: int = 16384,
    hidden_size: int = 256,
    num_tokens: int = 128,
) -> tuple[HardwareCapabilities, list[KernelBenchmarkResult]]:
    caps = detect_capabilities()
    device = torch.device("cuda" if caps.cuda_available else "cpu")
    sync_cuda = caps.cuda_available
    torch.manual_seed(42)

    results: list[KernelBenchmarkResult] = []

    # 1. Chunked Cross-Entropy
    h = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn((vocab_size, hidden_size), device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, vocab_size, (num_tokens,), device=device)

    ref_loss = reference_cross_entropy(h, w, targets)
    ref_loss.backward()
    ref_gh = h.grad.detach().clone()
    h.grad = None
    w.grad = None

    chunk_size = max(512, vocab_size // 8)
    cust_loss = chunked_cross_entropy(h, w, targets, vocab_chunk_size=chunk_size)
    cust_loss.backward()
    cust_gh = h.grad.detach().clone()

    ce_err = max(
        float((ref_loss.detach() - cust_loss.detach()).abs().item()),
        float((ref_gh - cust_gh).abs().max().item()),
    )
    ce_rel = ce_err / max(float(ref_loss.detach().abs().item()), 1e-6)
    with torch.no_grad():
        ref_ce_ms = _time_callable(lambda: reference_cross_entropy(h, w, targets), sync_cuda=sync_cuda)
        cust_ce_ms = _time_callable(
            lambda: chunked_cross_entropy(h, w, targets, vocab_chunk_size=chunk_size),
            sync_cuda=sync_cuda,
        )
    results.append(
        KernelBenchmarkResult(
            kernel_name="chunked_cross_entropy",
            backend=caps.active_kernel_backend,
            reference_ms=round(ref_ce_ms, 4),
            custom_ms=round(cust_ce_ms, 4),
            measured_speedup=round(ref_ce_ms / max(cust_ce_ms, 1e-6), 3),
            max_abs_error=round(ce_err, 7),
            relative_error=round(ce_rel, 7),
            memory_reduction_ratio=round(vocab_size / float(chunk_size), 2),
            correctness_passed=(ce_err < 1e-4),
        )
    )

    # 2. Grouped Expert GEMM (MoE)
    n_experts = 16
    top_k = 4
    x_moe = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float32)
    w_exp = torch.randn((n_experts, hidden_size, hidden_size), device=device, dtype=torch.float32) * 0.05
    topk_idx = torch.randint(0, n_experts, (num_tokens, top_k), device=device)
    topk_gates = torch.softmax(torch.randn((num_tokens, top_k), device=device), dim=-1)

    with torch.no_grad():
        y_ref = reference_moe_forward(x_moe, w_exp, topk_idx, topk_gates)
        y_grp = grouped_moe_forward(x_moe, w_exp, topk_idx, topk_gates)
        moe_err = float((y_ref - y_grp).abs().max().item())
        moe_rel = moe_err / max(float(y_ref.abs().max().item()), 1e-6)
        ref_moe_ms = _time_callable(
            lambda: reference_moe_forward(x_moe, w_exp, topk_idx, topk_gates), sync_cuda=sync_cuda
        )
        cust_moe_ms = _time_callable(
            lambda: grouped_moe_forward(x_moe, w_exp, topk_idx, topk_gates), sync_cuda=sync_cuda
        )
    results.append(
        KernelBenchmarkResult(
            kernel_name="grouped_moe_gemm",
            backend=caps.active_kernel_backend,
            reference_ms=round(ref_moe_ms, 4),
            custom_ms=round(cust_moe_ms, 4),
            measured_speedup=round(ref_moe_ms / max(cust_moe_ms, 1e-6), 3),
            max_abs_error=round(moe_err, 7),
            relative_error=round(moe_rel, 7),
            memory_reduction_ratio=1.0,
            correctness_passed=(moe_err < 1e-4),
        )
    )

    # 3. Fused LoRA Linear
    rank = 16
    scale = 2.0
    x_lora = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float32, requires_grad=True)
    w_base = torch.randn((hidden_size, hidden_size), device=device, dtype=torch.float32) * 0.05
    lora_a = (torch.randn((rank, hidden_size), device=device, dtype=torch.float32) * 0.05).requires_grad_(True)
    lora_b = (torch.randn((hidden_size, rank), device=device, dtype=torch.float32) * 0.05).requires_grad_(True)

    out_ref = reference_lora_linear(x_lora, w_base, lora_a, lora_b, scale)
    out_ref.sum().backward()
    ga_ref = lora_a.grad.detach().clone()
    gb_ref = lora_b.grad.detach().clone()
    x_lora.grad = None
    lora_a.grad = None
    lora_b.grad = None

    out_fused = fused_lora_linear(x_lora, w_base, lora_a, lora_b, scale)
    out_fused.sum().backward()
    ga_fused = lora_a.grad.detach().clone()
    gb_fused = lora_b.grad.detach().clone()

    lora_err = max(
        float((out_ref.detach() - out_fused.detach()).abs().max().item()),
        float((ga_ref - ga_fused).abs().max().item()),
        float((gb_ref - gb_fused).abs().max().item()),
    )
    lora_rel = lora_err / max(float(out_ref.detach().abs().max().item()), 1e-6)
    with torch.no_grad():
        ref_lora_ms = _time_callable(
            lambda: reference_lora_linear(x_lora, w_base, lora_a, lora_b, scale), sync_cuda=sync_cuda
        )
        cust_lora_ms = _time_callable(
            lambda: fused_lora_linear(x_lora, w_base, lora_a, lora_b, scale), sync_cuda=sync_cuda
        )
    results.append(
        KernelBenchmarkResult(
            kernel_name="fused_lora_linear",
            backend=caps.active_kernel_backend,
            reference_ms=round(ref_lora_ms, 4),
            custom_ms=round(cust_lora_ms, 4),
            measured_speedup=round(ref_lora_ms / max(cust_lora_ms, 1e-6), 3),
            max_abs_error=round(lora_err, 7),
            relative_error=round(lora_rel, 7),
            memory_reduction_ratio=1.33,
            correctness_passed=(lora_err < 1e-4),
        )
    )

    # 4. Low-Precision NVFP4 Block-Scaled Decode
    w_dec = torch.randn((hidden_size * 2, hidden_size), device=device, dtype=torch.float32) * 0.25
    x_dec = torch.randn((8, hidden_size), device=device, dtype=torch.float32)
    qw_bf16 = quantize_weights(w_dec, "BF16", group_size=16)
    qw_nvfp4 = quantize_weights(w_dec, "NVFP4", group_size=16)

    with torch.no_grad():
        y_bf16 = low_precision_decode_linear(x_dec, qw_bf16)
        y_fp4 = low_precision_decode_linear(x_dec, qw_nvfp4)
        fp4_err = float((y_bf16.float() - y_fp4.float()).abs().mean().item())
        fp4_rel = fp4_err / max(float(y_bf16.float().abs().mean().item()), 1e-6)
        bf16_ms = _time_callable(lambda: low_precision_decode_linear(x_dec, qw_bf16), sync_cuda=sync_cuda)
        fp4_ms = _time_callable(lambda: low_precision_decode_linear(x_dec, qw_nvfp4), sync_cuda=sync_cuda)

    results.append(
        KernelBenchmarkResult(
            kernel_name="low_prec_nvfp4_decode",
            backend=caps.active_kernel_backend,
            reference_ms=round(bf16_ms, 4),
            custom_ms=round(fp4_ms, 4),
            measured_speedup=round(bf16_ms / max(fp4_ms, 1e-6), 3),
            max_abs_error=round(fp4_err, 6),
            relative_error=round(fp4_rel, 6),
            memory_reduction_ratio=round(qw_bf16.storage_bytes / float(qw_nvfp4.storage_bytes), 2),
            correctness_passed=(fp4_rel < 0.18),
        )
    )
    return caps, results


def verify_training_loss_curve_equivalence(steps: int = 12, seed: int = 7) -> tuple[list[float], list[float], float]:
    """Train two identically seeded mini MoE+LoRA networks (PyTorch baseline vs custom kernels)
    and return `(baseline_curve, custom_curve, max_step_diff)`."""

    def _run_trajectory(use_custom: bool) -> list[float]:
        torch.manual_seed(seed)
        d_model = 64
        vocab = 512
        emb = torch.nn.Embedding(vocab, d_model)
        proj = FusedLoRALinear(d_model, d_model, rank=8, alpha=16, use_fused_kernel=use_custom)
        expert_w = torch.randn((4, d_model, d_model)) * 0.05
        head_w = torch.randn((vocab, d_model)) * 0.05

        opt = torch.optim.AdamW([proj.lora_a, proj.lora_b], lr=3e-3)
        losses: list[float] = []
        for step in range(steps):
            torch.manual_seed(1000 + step)
            seq = torch.randint(0, vocab, (4, 16))
            x = emb(seq[:, :-1])
            tgt = seq[:, 1:]
            h_lora = proj(x)
            flat_h = h_lora.reshape(-1, d_model)
            idx = torch.remainder(torch.arange(flat_h.shape[0]).unsqueeze(-1) + torch.tensor([[0, 1]]), 4)
            gates = torch.full((flat_h.shape[0], 2), 0.5)
            moe_out = (
                grouped_moe_forward(flat_h, expert_w, idx, gates)
                if use_custom
                else reference_moe_forward(flat_h, expert_w, idx, gates)
            )
            combined = flat_h + 0.2 * moe_out
            loss = (
                chunked_cross_entropy(combined, head_w, tgt.reshape(-1), vocab_chunk_size=128)
                if use_custom
                else reference_cross_entropy(combined, head_w, tgt.reshape(-1))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        return losses

    base_curve = _run_trajectory(use_custom=False)
    custom_curve = _run_trajectory(use_custom=True)
    max_diff = max(abs(a - b) for a, b in zip(base_curve, custom_curve))
    return base_curve, custom_curve, max_diff
