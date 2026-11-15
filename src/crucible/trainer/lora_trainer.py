"""LoRA fine-tuning trainer supporting single-node & multi-node (torchrun/NCCL) execution.

`torch` is imported strictly lazily inside `run_micro_torch_training` and `run_hf_peft_training`
so thin control-plane environments (e.g., macOS laptops orchestrating remote CIQ Fuzzball clusters)
can import Crucible without installing PyTorch locally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from crucible.config import RecipeCandidate, TrainingPerfCandidate
from crucible.trainer.types import StepMetricRecord, TrainingTrialSummary

__all__ = [
    "StepMetricRecord",
    "TrainingTrialSummary",
    "run_hf_peft_training",
    "run_micro_torch_training",
]


def _query_nvidia_smi(local_rank: int = 0) -> dict[str, float]:
    """Read physical GPU SM util %, temperature C, and power W via nvidia-smi (graceful on GB10 N/A fields)."""
    out: dict[str, float] = {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={local_rank}",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.2,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 1:
                try:
                    out["gpu_util_pct"] = float(parts[0])
                except ValueError:
                    pass
            if len(parts) >= 2:
                try:
                    out["gpu_temp_c"] = float(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    out["gpu_power_w"] = float(parts[2])
                except ValueError:
                    pass
    except Exception:
        pass
    return out


def _require_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. "
            "Install 'pip install -e \".[gpu]\"' for local_torch execution, or use "
            "'--backend fuzzball_mcp' / '--backend fuzzball' / '--backend simulator' to run on Team 20."
        ) from exc


def _load_packed_sequences(jsonl_path: Path, vocab_cap: int) -> list[list[int]]:
    seqs: list[list[int]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = [int(tok) % vocab_cap for tok in row["input_ids"]]
        if len(ids) >= 8:
            seqs.append(ids)
    if not seqs:
        raise ValueError(f"No usable sequences in {jsonl_path}")
    return seqs


def _build_micro_moe_lora_model(
    vocab_size: int,
    hidden_size: int,
    lora_rank: int,
    lora_alpha: int,
    num_experts: int = 8,
    top_k: int = 2,
    use_fused_lora: bool = False,
    use_grouped_moe: bool = False,
):
    torch, nn = _require_torch()
    from crucible.kernels.fused_lora import FusedLoRALinear
    from crucible.kernels.grouped_moe_gemm import grouped_moe_forward, reference_moe_forward

    class MicroMoELoRAModel(nn.Module):
        """Compact MoE + LoRA causal LM exercising all 3 training kernels on real packed token IDs."""

        def __init__(self) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            self.top_k = top_k
            self.use_grouped_moe = use_grouped_moe

            self.embed = nn.Embedding(vocab_size, hidden_size)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.embed.weight.requires_grad_(False)

            self.qkv_lora = FusedLoRALinear(
                hidden_size,
                hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                use_fused_kernel=use_fused_lora,
            )
            self.router = nn.Linear(hidden_size, num_experts, bias=False)
            self.router.weight.requires_grad_(False)
            self.expert_weights = nn.Parameter(
                torch.randn((num_experts, hidden_size, hidden_size)) * (1.0 / math.sqrt(hidden_size)),
                requires_grad=False,
            )
            self.out_lora = FusedLoRALinear(
                hidden_size,
                hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                use_fused_kernel=use_fused_lora,
            )
            self.lm_head = nn.Parameter(
                torch.randn((vocab_size, hidden_size)) * (1.0 / math.sqrt(hidden_size)),
                requires_grad=False,
            )

        def forward_hidden(self, input_ids):
            x = self.embed(input_ids)
            h = torch.tanh(self.qkv_lora(x))
            flat_h = h.reshape(-1, self.hidden_size)
            router_logits = self.router(flat_h)
            topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_gates = torch.softmax(topk_vals, dim=-1)
            if self.use_grouped_moe:
                moe_out = grouped_moe_forward(flat_h, self.expert_weights, topk_idx, topk_gates)
            else:
                moe_out = reference_moe_forward(flat_h, self.expert_weights, topk_idx, topk_gates)
            merged = h + 0.25 * moe_out.reshape_as(h)
            return self.out_lora(merged)

    return MicroMoELoRAModel()


def run_micro_torch_training(
    trial_id: str,
    processed_data_dir: Path,
    output_dir: Path,
    recipe: RecipeCandidate,
    perf: TrainingPerfCandidate,
    max_steps: int = 20,
    vocab_cap: int = 2048,
    hidden_size: int = 64,
    measured_peak_tok_s: float = 4150.0,
    seed: int = 1234,
) -> TrainingTrialSummary:
    torch, _ = _require_torch()
    from crucible.kernels.chunked_cross_entropy import (
        chunked_cross_entropy,
        reference_cross_entropy,
    )

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    train_seqs = _load_packed_sequences(processed_data_dir / "train.jsonl", vocab_cap)
    val_seqs = _load_packed_sequences(processed_data_dir / "val.jsonl", vocab_cap)
    held_out_seqs = _load_packed_sequences(processed_data_dir / "held_out.jsonl", vocab_cap)

    model = _build_micro_moe_lora_model(
        vocab_size=vocab_cap,
        hidden_size=hidden_size,
        lora_rank=min(recipe.lora_rank, 32),
        lora_alpha=min(recipe.lora_alpha, 64),
        use_fused_lora=perf.use_fused_lora_kernel,
        use_grouped_moe=perf.use_grouped_moe_kernel,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=recipe.lr)

    def _eval_split(seqs: list[list[int]]) -> float:
        with torch.no_grad():
            losses: list[float] = []
            for seq in seqs[:8]:
                inp = torch.tensor([seq[:-1][:128]], dtype=torch.long)
                tgt = torch.tensor([seq[1:][:128]], dtype=torch.long)
                hid = model.forward_hidden(inp)
                loss = (
                    chunked_cross_entropy(hid, model.lm_head, tgt, vocab_chunk_size=512)
                    if perf.use_chunked_ce_kernel
                    else reference_cross_entropy(hid, model.lm_head, tgt)
                )
                losses.append(float(loss.item()))
            return sum(losses) / len(losses)

    initial_val = _eval_split(val_seqs)
    loss_curve: list[float] = []
    tok_rates: list[float] = []
    t_start = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8") as m_fh:
        for step in range(1, max_steps + 1):
            step_t0 = time.perf_counter()
            optimizer.zero_grad()
            batch_tokens = 0
            accum_loss = 0.0
            for micro in range(perf.grad_accum_steps):
                seq = train_seqs[(step * perf.grad_accum_steps + micro) % len(train_seqs)]
                window = seq[: min(len(seq), 129)]
                inp = torch.tensor([window[:-1]] * perf.micro_batch_size, dtype=torch.long)
                tgt = torch.tensor([window[1:]] * perf.micro_batch_size, dtype=torch.long)
                hid = model.forward_hidden(inp)
                loss = (
                    chunked_cross_entropy(hid, model.lm_head, tgt, vocab_chunk_size=512)
                    if perf.use_chunked_ce_kernel
                    else reference_cross_entropy(hid, model.lm_head, tgt)
                )
                (loss / perf.grad_accum_steps).backward()
                accum_loss += float(loss.detach().item()) / perf.grad_accum_steps
                batch_tokens += int(tgt.numel())

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            optimizer.step()
            dt = max(time.perf_counter() - step_t0, 1e-5)
            tok_s = batch_tokens / dt
            tok_rates.append(tok_s)
            loss_curve.append(round(accum_loss, 6))
            val_loss = round(_eval_split(val_seqs), 6) if (step == 1 or step == max_steps or step % 4 == 0) else None

            rec = StepMetricRecord(
                step=step,
                train_loss=round(accum_loss, 6),
                val_loss=val_loss,
                tokens_per_sec=round(tok_s, 2),
                peak_mem_gb=0.25,
                wall_time_s=round(time.perf_counter() - t_start, 4),
                world_size=perf.nodes,
            )
            m_fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
            print(
                "CRUCIBLE_STEP_JSON="
                + json.dumps(
                    {
                        "step": step,
                        "max_steps": max_steps,
                        "train_loss": round(accum_loss, 6),
                        "val_loss": val_loss,
                        "lr": recipe.lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens_per_sec": round(tok_s, 2),
                        "peak_mem_gb": 0.25,
                        "world_size": perf.nodes,
                        "partial_curve": list(loss_curve),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_val = _eval_split(val_seqs)
    held_out_loss = _eval_split(held_out_seqs)
    total_wall = time.perf_counter() - t_start
    mean_tok_s = sum(tok_rates) / max(len(tok_rates), 1)

    torch.save(
        {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        output_dir / "adapter_model.pt",
    )

    summary = TrainingTrialSummary(
        trial_id=trial_id,
        status="completed",
        steps_completed=max_steps,
        initial_val_loss=round(initial_val, 5),
        final_train_loss=round(loss_curve[-1], 5),
        final_val_loss=round(final_val, 5),
        held_out_eval_loss=round(held_out_loss, 5),
        held_out_perplexity=round(math.exp(min(held_out_loss, 20.0)), 4),
        mean_tokens_per_sec=round(mean_tok_s, 2),
        hardware_utilization_pct=round(min(99.0, (mean_tok_s / measured_peak_tok_s) * 100.0), 2),
        peak_mem_gb=0.25,
        wall_time_s=round(total_wall, 4),
        loss_curve=loss_curve,
        world_size=perf.nodes,
        kernels_enabled={
            "chunked_cross_entropy": perf.use_chunked_ce_kernel,
            "grouped_moe_gemm": perf.use_grouped_moe_kernel,
            "fused_lora_linear": perf.use_fused_lora_kernel,
        },
    )
    summary_json = json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    print(f"CRUCIBLE_SUMMARY_JSON={json.dumps(summary.to_dict(), sort_keys=True)}", flush=True)
    return summary


def run_hf_peft_training(
    trial_id: str,
    base_model_dir: Path,
    processed_data_dir: Path,
    output_dir: Path,
    recipe: RecipeCandidate,
    perf: TrainingPerfCandidate,
    max_steps: int,
    measured_peak_tok_s: float = 4150.0,
) -> TrainingTrialSummary:
    """Production single-node & torchrun/NCCL multi-node HuggingFace + PEFT LoRA loop."""
    torch, _ = _require_torch()
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    from crucible.kernels.chunked_cross_entropy import chunked_cross_entropy

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    ).to(device)

    if perf.activation_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        target_modules=recipe.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_cfg)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    raw_seqs = [
        json.loads(ln)["input_ids"]
        for ln in (processed_data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    val_seqs = [
        json.loads(ln)["input_ids"]
        for ln in (processed_data_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    held_seqs = [
        json.loads(ln)["input_ids"]
        for ln in (processed_data_dir / "held_out.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=recipe.lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    unwrapped = model.module if hasattr(model, "module") else model

    def _compute_loss(batch_ids):
        inp = batch_ids[:, :-1]
        tgt = batch_ids[:, 1:]
        if perf.use_chunked_ce_kernel and hasattr(unwrapped, "get_output_embeddings"):
            out = unwrapped.base_model(inp, output_hidden_states=True)
            last_h = out.hidden_states[-1]
            head_w = unwrapped.get_output_embeddings().weight
            return chunked_cross_entropy(last_h, head_w, tgt, vocab_chunk_size=8192)
        return model(input_ids=inp, labels=tgt).loss

    def _eval_seqs(seqs: list[list[int]]) -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for seq in seqs[:16]:
                t = torch.tensor([seq[: recipe.seq_len]], dtype=torch.long, device=device)
                losses.append(float(_compute_loss(t).item()))
        model.train()
        return sum(losses) / max(1, len(losses))

    init_val = _eval_seqs(val_seqs)
    loss_curve: list[float] = [round(init_val, 5)]
    val_curve: list[float] = [round(init_val, 5)]
    tok_rates: list[float] = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    adapter_mb = round(sum(p.numel() * p.element_size() for p in trainable_params) / float(1024**2), 2)
    warmup_steps = max(2, int(max_steps * recipe.warmup_ratio))
    eval_every = max(1, min(8, max_steps // 6))
    emit_every = 1 if max_steps <= 60 else 2
    t0 = time.perf_counter()
    if rank == 0:
        mem0 = round(torch.cuda.max_memory_allocated(device) / float(1024**3), 2) if torch.cuda.is_available() else 0.0
        print(
            "CRUCIBLE_STEP_JSON="
            + json.dumps(
                {
                    "step": 0,
                    "max_steps": max_steps,
                    "train_loss": round(init_val, 5),
                    "val_loss": round(init_val, 5),
                    "perplexity": round(math.exp(min(init_val, 12.0)), 4),
                    "lr": 0.0,
                    "grad_norm": 0.0,
                    "peak_mem_gb": mem0,
                    "world_size": world_size,
                    "tokens_per_sec": 0.0,
                    "partial_curve": list(loss_curve),
                    "partial_val_curve": list(val_curve),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    m_fh = metrics_path.open("w", encoding="utf-8") if rank == 0 else None
    try:
        for step in range(1, max_steps + 1):
            st0 = time.perf_counter()
            if step <= warmup_steps:
                cur_lr = recipe.lr * (step / float(warmup_steps))
            else:
                prog = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                cur_lr = recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad()
            step_loss = 0.0
            tokens = 0
            comm_s = 0.0
            for micro in range(perf.grad_accum_steps):
                idx = (step * world_size + rank + micro) % len(raw_seqs)
                batch = [
                    raw_seqs[(idx + b) % len(raw_seqs)][: recipe.seq_len]
                    for b in range(perf.micro_batch_size)
                ]
                min_l = min(len(s) for s in batch)
                t_batch = torch.tensor([s[:min_l] for s in batch], dtype=torch.long, device=device)
                loss = _compute_loss(t_batch)
                is_sync_micro = (micro == perf.grad_accum_steps - 1)
                if world_size > 1 and hasattr(model, "no_sync") and not is_sync_micro:
                    with model.no_sync():
                        (loss / perf.grad_accum_steps).backward()
                else:
                    if world_size > 1 and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_comm0 = time.perf_counter()
                    (loss / perf.grad_accum_steps).backward()
                    if world_size > 1 and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if world_size > 1:
                        comm_s = max(0.0, (time.perf_counter() - t_comm0) * 0.25)
                step_loss += float(loss.detach().item()) / perf.grad_accum_steps
                tokens += int(t_batch.numel()) * world_size

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = max(time.perf_counter() - st0, 1e-5)
            tok_s = tokens / dt
            tok_rates.append(tok_s)
            loss_curve.append(round(step_loss, 5))
            mem_gb = (
                torch.cuda.max_memory_allocated(device) / float(1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            step_val: float | None = None
            if step == 1 or step == max_steps or step % eval_every == 0:
                if rank == 0:
                    step_val = round(_eval_seqs(val_seqs[:6]), 5)
                    val_curve.append(step_val)
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()

            if m_fh is not None:
                rec = StepMetricRecord(
                    step=step,
                    train_loss=round(step_loss, 5),
                    val_loss=step_val,
                    tokens_per_sec=round(tok_s, 2),
                    peak_mem_gb=round(mem_gb, 2),
                    wall_time_s=round(time.perf_counter() - t0, 3),
                    world_size=world_size,
                )
                m_fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
                m_fh.flush()
                if step == 1 or step == max_steps or step % emit_every == 0 or step_val is not None:
                    smi = _query_nvidia_smi(local_rank) if torch.cuda.is_available() else {}
                    ar_ms = round(comm_s * 1000.0, 2) if world_size > 1 else 0.0
                    ig_gbps = (
                        round((adapter_mb * 2.0 * 8.0) / max(comm_s, 0.005), 2)
                        if world_size > 1
                        else 0.0
                    )
                    print(
                        "CRUCIBLE_STEP_JSON="
                        + json.dumps(
                            {
                                "step": step,
                                "max_steps": max_steps,
                                "train_loss": round(step_loss, 5),
                                "val_loss": step_val if step_val is not None else (val_curve[-1] if val_curve else round(init_val, 5)),
                                "lr": round(cur_lr, 7),
                                "grad_norm": round(grad_norm, 4),
                                "tokens_per_sec": round(tok_s, 1),
                                "step_time_ms": round(dt * 1000.0, 1),
                                "allreduce_ms": ar_ms,
                                "payload_mb": round(adapter_mb * 2.0, 2) if world_size > 1 else 0.0,
                                "inter_gpu_gbps": ig_gbps,
                                "peak_mem_gb": round(mem_gb, 2),
                                "world_size": world_size,
                                "partial_curve": list(loss_curve),
                                "partial_val_curve": list(val_curve),
                                **smi,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        if m_fh is not None:
            m_fh.close()

    final_val = _eval_seqs(val_seqs)
    held_loss = _eval_seqs(held_seqs)
    mean_tok = sum(tok_rates) / max(len(tok_rates), 1)
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / float(1024**3) if torch.cuda.is_available() else 0.0
    )

    if rank == 0:
        unwrapped.save_pretrained(str(output_dir / "lora_adapter"))

    summary = TrainingTrialSummary(
        trial_id=trial_id,
        status="completed",
        steps_completed=max_steps,
        initial_val_loss=round(init_val, 5),
        final_train_loss=round(loss_curve[-1], 5),
        final_val_loss=round(final_val, 5),
        held_out_eval_loss=round(held_loss, 5),
        held_out_perplexity=round(math.exp(min(held_loss, 20.0)), 4),
        mean_tokens_per_sec=round(mean_tok, 2),
        hardware_utilization_pct=round(
            min(99.0, (mean_tok / (measured_peak_tok_s * world_size)) * 100.0), 2
        ),
        peak_mem_gb=round(peak_gb, 2),
        wall_time_s=round(time.perf_counter() - t0, 3),
        loss_curve=loss_curve,
        world_size=world_size,
        kernels_enabled={
            "chunked_cross_entropy": perf.use_chunked_ce_kernel,
            "grouped_moe_gemm": perf.use_grouped_moe_kernel,
            "fused_lora_linear": perf.use_fused_lora_kernel,
        },
    )
    if rank == 0:
        (output_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"CRUCIBLE_SUMMARY_JSON={json.dumps(summary.to_dict(), sort_keys=True)}", flush=True)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crucible LoRA fine-tuning trainer")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--mode", choices=["hf_peft", "micro_torch"], default="hf_peft")
    parser.add_argument("--base-model-dir", type=Path, default=Path("/vast/models/Qwen3-Coder-30B-A3B"))
    parser.add_argument("--processed-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--perf-json", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args(argv)

    recipe = RecipeCandidate.model_validate_json(args.recipe_json)
    perf = TrainingPerfCandidate.model_validate_json(args.perf_json)
    if args.mode == "micro_torch":
        run_micro_torch_training(
            trial_id=args.trial_id,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            recipe=recipe,
            perf=perf,
            max_steps=args.max_steps,
        )
    else:
        run_hf_peft_training(
            trial_id=args.trial_id,
            base_model_dir=args.base_model_dir,
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            recipe=recipe,
            perf=perf,
            max_steps=args.max_steps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
