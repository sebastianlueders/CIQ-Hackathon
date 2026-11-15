"""Grouped expert GEMM for MoE layers (Triton; written for GB10 / sm_121, mma.sync via tl.dot).

Rows of X are sorted by expert; `offs` holds cumulative row ends per expert (int32, like
torch._grouped_mm). For each expert e:   Y[rows_e] = X[rows_e] @ B_e
  forward  (linear):  B_e = W[e]^T   W: [E, out, in]   (Y = X W^T)
  backward (dX only): B_e = W[e]     dX = dY W         (expert weights are frozen under LoRA)
Both are the same kernel with different weight strides.

One launch covers every expert: a tile table maps program id -> (expert, first row). Its size is
a fixed upper bound (ceil(S/BM) + E), built on device with searchsorted, so there is no
GPU->CPU sync per call (hundreds of calls per training step) and it stays CUDA-graph friendly.

`triton_grouped_experts_forward` mirrors transformers' `grouped_mm_experts_forward`
(routing, sort, offsets, gate, combine) and swaps only the grouped linear. Register it with
`register()` and select it with config._experts_implementation = "triton_grouped".
"""
import torch
import triton
import triton.language as tl

BM = 64  # rows per tile (fixed: the tile table depends on it). ~256 rows/expert -> ~4 tiles.


@triton.autotune(
    configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=s)
             for bn in (64, 128) for bk in (32, 64) for w in (4, 8) for s in (2, 3, 4)],
    key=["N", "K"],
)
@triton.jit
def _grouped_gemm_kernel(X, B, Y, tile_e, tile_m0, offs, N, K, E,
                         sxm, sxk, sbe, sbk, sbn, sym, syn,
                         BM_: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(tile_e + pid_t)
    if e >= E:  # padding tile of the fixed-size table
        return
    m0 = tl.load(tile_m0 + pid_t)
    m_end = tl.load(offs + e)
    rm = m0 + tl.arange(0, BM_)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < m_end
    n_ok = rn < N
    x_ptr = X + rm[:, None] * sxm + rk[None, :] * sxk
    b_ptr = B + e.to(tl.int64) * sbe + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM_, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        k_ok = (k + rk) < K
        x = tl.load(x_ptr, mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        b = tl.load(b_ptr, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
        acc += tl.dot(x, b)
        x_ptr += BK * sxk
        b_ptr += BK * sbk
    y_ptr = Y + rm[:, None] * sym + rn[None, :] * syn
    tl.store(y_ptr, acc.to(Y.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


def tile_table(offs, num_rows, num_experts):
    """(tile_e, tile_m0) with a fixed upper-bound length; padding tiles have expert id E."""
    offs = offs.to(torch.int64)
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    ntiles = (offs - starts + BM - 1) // BM
    cum = torch.cumsum(ntiles, 0)
    max_tiles = (num_rows + BM - 1) // BM + num_experts
    t = torch.arange(max_tiles, device=offs.device)
    e = torch.searchsorted(cum, t, right=True)
    e_c = e.clamp(max=num_experts - 1)
    m0 = starts[e_c] + (t - (cum[e_c] - ntiles[e_c])) * BM
    return e.to(torch.int32), m0.to(torch.int32)


def _launch(x, b_view, offs, tables, out_dim, transposed_strides):
    E = offs.numel()
    S, K = x.shape
    y = torch.empty(S, out_dim, device=x.device, dtype=x.dtype)
    te, tm = tables
    sbe, sbk, sbn = transposed_strides
    grid = lambda meta: (te.numel(), triton.cdiv(out_dim, meta["BN"]))
    _grouped_gemm_kernel[grid](x, b_view, y, te, tm, offs, out_dim, K, E,
                               x.stride(0), x.stride(1), sbe, sbk, sbn, y.stride(0), y.stride(1), BM_=BM)
    return y


class GroupedLinear(torch.autograd.Function):
    """Y = X W[e]^T per expert group. W frozen: backward returns dX only."""

    @staticmethod
    def forward(ctx, x, weight, offs):
        x = x.contiguous()
        tables = tile_table(offs, x.shape[0], weight.shape[0])
        # B_e[k, n] = W[e, n, k]
        y = _launch(x, weight, offs, tables, weight.shape[1], (weight.stride(0), weight.stride(2), weight.stride(1)))
        ctx.save_for_backward(weight, offs, *tables)
        return y

    @staticmethod
    def backward(ctx, dy):
        weight, offs, te, tm = ctx.saved_tensors
        # dX = dY @ W[e]: B_e[k, n] = W[e, k, n] with K = out_dim, N = in_dim
        dx = _launch(dy.contiguous(), weight, offs, (te, tm), weight.shape[2],
                     (weight.stride(0), weight.stride(1), weight.stride(2)))
        return dx, None, None


def grouped_linear(x, weight, offs):
    assert not weight.requires_grad, "expert weights must be frozen (LoRA); dW is not implemented"
    return GroupedLinear.apply(x, weight, offs)


def triton_grouped_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Drop-in for transformers' grouped_mm_experts_forward (no bias, no expert parallelism)."""
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    x_g = hidden_states[perm // num_top_k]
    w_g = sample_weights[perm]
    counts = torch.histc(expert_ids_g.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    up = self.gate_up_proj if self.has_gate else self.up_proj
    if getattr(self, "is_transposed", False):  # weights stored [E, in, out]: view them as [E, out, in]
        up = up.transpose(-2, -1)
    h = grouped_linear(x_g, up, offs)
    h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
    down = self.down_proj.transpose(-2, -1) if getattr(self, "is_transposed", False) else self.down_proj
    out = grouped_linear(h, down, offs) * w_g.unsqueeze(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return out[inv].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def register(name="triton_grouped"):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS.register(name, triton_grouped_experts_forward)
    return name
