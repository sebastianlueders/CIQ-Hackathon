"""Numerical correctness & loss-curve equivalence tests for all 4 custom sm_121 kernels."""

from __future__ import annotations

import torch

from crucible.kernels.benchmark import (
    run_kernel_suite,
    verify_training_loss_curve_equivalence,
)
from crucible.kernels.chunked_cross_entropy import chunked_cross_entropy, reference_cross_entropy
from crucible.kernels.fused_lora import fused_lora_linear, reference_lora_linear
from crucible.kernels.grouped_moe_gemm import grouped_moe_forward, reference_moe_forward
from crucible.kernels.low_prec_decode import (
    low_precision_decode_linear,
    quantize_weights,
)


def test_chunked_cross_entropy_forward_backward_with_ignore_index() -> None:
    torch.manual_seed(11)
    h_ref = torch.randn((5, 7, 48), requires_grad=True)
    w_ref = torch.randn((777, 48), requires_grad=True)
    targets = torch.randint(0, 777, (5, 7))
    targets[0, 3] = -100
    targets[4, 0] = -100

    h_cust = h_ref.detach().clone().requires_grad_(True)
    w_cust = w_ref.detach().clone().requires_grad_(True)

    loss_ref = reference_cross_entropy(h_ref, w_ref, targets, ignore_index=-100)
    loss_ref.backward()

    loss_cust = chunked_cross_entropy(
        h_cust, w_cust, targets, vocab_chunk_size=128, ignore_index=-100
    )
    loss_cust.backward()

    assert torch.allclose(loss_ref, loss_cust, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h_ref.grad, h_cust.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w_ref.grad, w_cust.grad, atol=1e-5, rtol=1e-5)


def test_grouped_moe_gemm_matches_reference() -> None:
    torch.manual_seed(22)
    x = torch.randn((32, 64))
    experts = torch.randn((12, 48, 64)) * 0.1
    topk_idx = torch.randint(0, 12, (32, 4))
    topk_gates = torch.softmax(torch.randn((32, 4)), dim=-1)

    ref = reference_moe_forward(x, experts, topk_idx, topk_gates)
    grp = grouped_moe_forward(x, experts, topk_idx, topk_gates)
    assert torch.allclose(ref, grp, atol=1e-5, rtol=1e-5)


def test_fused_lora_linear_forward_and_gradients() -> None:
    torch.manual_seed(33)
    x1 = torch.randn((6, 10, 32), requires_grad=True)
    w = torch.randn((40, 32)) * 0.1
    a1 = (torch.randn((8, 32)) * 0.1).requires_grad_(True)
    b1 = (torch.randn((40, 8)) * 0.1).requires_grad_(True)

    x2 = x1.detach().clone().requires_grad_(True)
    a2 = a1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)

    y_ref = reference_lora_linear(x1, w, a1, b1, scale=2.0)
    (y_ref * torch.randn_like(y_ref)).sum().backward()
    # Replay with identical upstream grad
    torch.manual_seed(99)
    upstream = torch.randn_like(y_ref)
    x1.grad = a1.grad = b1.grad = None
    (reference_lora_linear(x1, w, a1, b1, scale=2.0) * upstream).sum().backward()

    y_fused = fused_lora_linear(x2, w, a2, b2, scale=2.0)
    (y_fused * upstream).sum().backward()

    assert torch.allclose(y_ref, y_fused, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a1.grad, a2.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-5, rtol=1e-5)


def test_low_precision_decode_fp8_and_nvfp4_compression() -> None:
    torch.manual_seed(44)
    w = torch.randn((128, 64)) * 0.2
    x = torch.randn((4, 64))

    q_bf16 = quantize_weights(w, "BF16", group_size=16)
    q_fp8 = quantize_weights(w, "FP8", group_size=16)
    q_fp4 = quantize_weights(w, "NVFP4", group_size=16)

    assert q_fp8.storage_bytes < q_bf16.storage_bytes
    assert q_fp4.storage_bytes < q_fp8.storage_bytes

    y_bf16 = low_precision_decode_linear(x, q_bf16).float()
    y_fp8 = low_precision_decode_linear(x, q_fp8).float()
    y_fp4 = low_precision_decode_linear(x, q_fp4).float()

    fp8_rel = (y_bf16 - y_fp8).abs().mean() / y_bf16.abs().mean()
    fp4_rel = (y_bf16 - y_fp4).abs().mean() / y_bf16.abs().mean()
    assert float(fp8_rel.item()) < 0.03
    assert float(fp4_rel.item()) < 0.16


def test_kernel_suite_and_multistep_loss_curve_equivalence() -> None:
    _, results = run_kernel_suite(vocab_size=2048, hidden_size=64, num_tokens=32)
    assert all(r.correctness_passed for r in results)

    base_curve, custom_curve, max_diff = verify_training_loss_curve_equivalence(steps=8)
    assert len(base_curve) == len(custom_curve) == 8
    assert max_diff < 1e-4
