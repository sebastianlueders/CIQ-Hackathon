"""Dynamic open-weight model catalog and architecture inference for ANY HuggingFace model ID."""

from __future__ import annotations

import re
from typing import Any

from crucible.config import (
    ClusterConfig,
    DatasetSplitConfig,
    ExperimentConfig,
    InferencePerfCandidate,
    ModelProfileConfig,
    RecipeCandidate,
    StageARecipeConfig,
    StageBTrainingPerfConfig,
    StageCInferencePerfConfig,
    TrainingPerfCandidate,
)

MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "label": "Qwen3-Coder-30B-A3B",
        "family": "MoE Code",
        "architecture": "qwen3_moe",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_layers": 48,
        "num_experts": 128,
        "top_k_experts": 8,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "label": "Qwen2.5-Coder-7B",
        "family": "Dense Code",
        "architecture": "qwen2_dense",
        "total_params_b": 7.6,
        "active_params_b": 7.6,
        "vocab_size": 152064,
        "hidden_size": 3584,
        "num_layers": 28,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "label": "Qwen2.5-Coder-14B",
        "family": "Dense Code",
        "architecture": "qwen2_dense",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "vocab_size": 152064,
        "hidden_size": 5120,
        "num_layers": 48,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "label": "Qwen2.5-Coder-32B",
        "family": "Dense Code",
        "architecture": "qwen2_dense",
        "total_params_b": 32.8,
        "active_params_b": 32.8,
        "vocab_size": 152064,
        "hidden_size": 5120,
        "num_layers": 64,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "label": "DeepSeek-Coder-V2-Lite",
        "family": "MoE Code",
        "architecture": "deepseek_v2_moe",
        "total_params_b": 15.7,
        "active_params_b": 2.4,
        "vocab_size": 102400,
        "hidden_size": 2048,
        "num_layers": 27,
        "num_experts": 64,
        "top_k_experts": 6,
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B",
        "family": "Dense General",
        "architecture": "llama3_dense",
        "total_params_b": 8.0,
        "active_params_b": 8.0,
        "vocab_size": 128256,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "mistralai/Codestral-22B-v0.1",
        "label": "Codestral-22B",
        "family": "Dense Code",
        "architecture": "mistral_dense",
        "total_params_b": 22.2,
        "active_params_b": 22.2,
        "vocab_size": 32768,
        "hidden_size": 6144,
        "num_layers": 56,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "label": "Llama-3.3-70B",
        "family": "2-Node Dense",
        "architecture": "llama3_dense",
        "total_params_b": 70.6,
        "active_params_b": 70.6,
        "vocab_size": 128256,
        "hidden_size": 8192,
        "num_layers": 80,
        "num_experts": 1,
        "top_k_experts": 1,
    },
    {
        "id": "Qwen/Qwen3-Coder-80B-A3B-Instruct",
        "label": "Qwen3-Coder-80B-A3B",
        "family": "2-Node MoE",
        "architecture": "qwen3_moe",
        "total_params_b": 80.0,
        "active_params_b": 8.0,
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_layers": 64,
        "num_experts": 128,
        "top_k_experts": 8,
    },
]

_MIXTRAL_RE = re.compile(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB]")
_ACTIVE_RE = re.compile(r"[-_][aA](\d+(?:\.\d+)?)\s*[bB]\b")
_PARAMS_B_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*[bB](?:$|[-_/])")


def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return clean[:48] or "custom-model"


def infer_model_profile(model_id: str, node_mem_gb: int = 104) -> ModelProfileConfig:
    """Resolve exact preset or infer architecture, parameter scale, MoE routing, and node sizing
    for ANY HuggingFace repo ID (`org/model`) or local `/models/...` path."""
    cleaned = model_id.strip() or "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    lower = cleaned.lower()

    for preset in MODEL_CATALOG:
        if preset["id"].lower() == lower or preset["label"].lower() == lower:
            bf16_gb = round(float(preset["total_params_b"]) * 1.92, 1)
            min_nodes = 1 if bf16_gb <= (node_mem_gb - 18) else 2
            subpath = cleaned.lstrip("/").removeprefix("models/")
            return ModelProfileConfig(
                name=cleaned,
                weights_subpath=subpath,
                architecture=preset["architecture"],
                total_params_b=float(preset["total_params_b"]),
                active_params_b=float(preset["active_params_b"]),
                vocab_size=int(preset["vocab_size"]),
                hidden_size=int(preset["hidden_size"]),
                num_layers=int(preset["num_layers"]),
                num_experts=int(preset["num_experts"]),
                top_k_experts=int(preset["top_k_experts"]),
                bf16_footprint_gb=bf16_gb,
                min_nodes_required=min_nodes,
            )

    # Dynamic inference for arbitrary custom model IDs
    mix_match = _MIXTRAL_RE.search(cleaned)
    act_match = _ACTIVE_RE.search(cleaned)
    param_matches = _PARAMS_B_RE.findall(cleaned)

    is_moe = bool(
        mix_match
        or act_match
        or any(tok in lower for tok in ("moe", "mixtral", "deepseek-v2", "deepseek-v3", "deepseek-coder-v2", "qwen3-coder-30b", "qwen3-coder-80b", "dbrx", "arctic", "jamba"))
    )

    if mix_match:
        n_exp = int(mix_match.group(1))
        per_exp = float(mix_match.group(2))
        total_b = round(n_exp * per_exp * 0.84, 1)
        active_b = round(2.0 * per_exp, 1)
        top_k = 2
    elif param_matches:
        total_b = float(param_matches[0])
        if act_match:
            active_b = float(act_match.group(1))
        elif is_moe:
            active_b = round(max(1.5, total_b * 0.14), 1)
        else:
            active_b = total_b
        n_exp = 64 if is_moe else 1
        top_k = 6 if is_moe else 1
    else:
        total_b = 16.0 if is_moe else 8.0
        active_b = 2.5 if is_moe else 8.0
        n_exp = 64 if is_moe else 1
        top_k = 6 if is_moe else 1

    if "qwen" in lower:
        vocab = 151936
        arch = "qwen_moe" if is_moe else "qwen_dense"
    elif "llama" in lower:
        vocab = 128256
        arch = "llama_moe" if is_moe else "llama_dense"
    elif "deepseek" in lower:
        vocab = 102400
        arch = "deepseek_moe" if is_moe else "deepseek_dense"
    elif "gemma" in lower:
        vocab = 256000
        arch = "gemma_dense"
    else:
        vocab = 65536
        arch = "transformer_moe" if is_moe else "transformer_dense"

    hidden = 2048 if active_b <= 4.0 else (4096 if active_b <= 16.0 else (5120 if active_b <= 36.0 else 8192))
    layers = 28 if total_b <= 9.0 else (40 if total_b <= 20.0 else (48 if total_b <= 36.0 else 64))
    bf16_gb = round(total_b * 1.95, 1)
    min_nodes = 1 if bf16_gb <= (node_mem_gb - 16) else 2
    subpath = cleaned.lstrip("/").removeprefix("models/")

    return ModelProfileConfig(
        name=cleaned,
        weights_subpath=subpath,
        architecture=arch,
        total_params_b=total_b,
        active_params_b=active_b,
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=layers,
        num_experts=n_exp,
        top_k_experts=top_k,
        bf16_footprint_gb=bf16_gb,
        min_nodes_required=min_nodes,
    )


def build_dynamic_experiment_config(
    model_id: str,
    dataset_name: str,
    raw_subpath: str,
    cluster: ClusterConfig,
) -> ExperimentConfig:
    """Construct a complete 4-stage `ExperimentConfig` tailored to `model_id` and `dataset_name`."""
    profile = infer_model_profile(model_id, node_mem_gb=cluster.nodes.memory_gb_per_node)
    is_moe = profile.num_experts > 1
    two_node_only = profile.min_nodes_required >= 2
    exp_slug = f"{_slugify(dataset_name)}-{_slugify(profile.name.split('/')[-1])}"

    base_lr = 0.0002 if profile.active_params_b <= 10.0 else 0.00012
    stage_a = StageARecipeConfig(
        parallel_nodes=cluster.nodes.count,
        reduction_factor=2,
        rung_steps=[40, 120, 320] if two_node_only else [60, 180, 480],
        candidates=[
            RecipeCandidate(
                lr=round(base_lr, 6),
                lora_rank=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                warmup_ratio=0.05,
                seq_len=2048,
            ),
            RecipeCandidate(
                lr=round(base_lr * 0.75, 6),
                lora_rank=32,
                lora_alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                warmup_ratio=0.05,
                seq_len=2048,
            ),
            RecipeCandidate(
                lr=round(base_lr * 1.5, 6),
                lora_rank=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                warmup_ratio=0.03,
                seq_len=2048,
            ),
            RecipeCandidate(
                lr=round(base_lr * 0.45, 6),
                lora_rank=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                warmup_ratio=0.08,
                seq_len=2048,
            ),
            RecipeCandidate(
                lr=round(base_lr * 3.8, 6),
                lora_rank=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                warmup_ratio=0.02,
                seq_len=1024,
            ),
        ],
    )

    if two_node_only:
        b_cands = [
            TrainingPerfCandidate(
                label="pytorch-fsdp-2node",
                nodes=2,
                micro_batch_size=2,
                grad_accum_steps=4,
                activation_checkpointing=True,
                use_chunked_ce_kernel=False,
                use_grouped_moe_kernel=False,
                use_fused_lora_kernel=False,
            ),
            TrainingPerfCandidate(
                label="sm121-chunked-ce-fsdp-2node",
                nodes=2,
                micro_batch_size=4,
                grad_accum_steps=2,
                activation_checkpointing=True,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=False,
                use_fused_lora_kernel=False,
            ),
            TrainingPerfCandidate(
                label="sm121-all-kernels-fsdp-2node",
                nodes=2,
                micro_batch_size=4,
                grad_accum_steps=2,
                activation_checkpointing=True,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            ),
            TrainingPerfCandidate(
                label="oversized-global-batch-2node",
                nodes=2,
                micro_batch_size=6,
                grad_accum_steps=8,
                activation_checkpointing=True,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            ),
        ]
    else:
        max_mb = 8 if profile.bf16_footprint_gb <= 24.0 else (6 if profile.bf16_footprint_gb <= 60.0 else 4)
        b_cands = [
            TrainingPerfCandidate(
                label="pytorch-baseline-1node",
                nodes=1,
                micro_batch_size=2,
                grad_accum_steps=8,
                activation_checkpointing=True,
                use_chunked_ce_kernel=False,
                use_grouped_moe_kernel=False,
                use_fused_lora_kernel=False,
            ),
            TrainingPerfCandidate(
                label="tuned-batch-ckpt-1node",
                nodes=1,
                micro_batch_size=4,
                grad_accum_steps=4,
                activation_checkpointing=True,
                use_chunked_ce_kernel=False,
                use_grouped_moe_kernel=False,
                use_fused_lora_kernel=False,
            ),
            TrainingPerfCandidate(
                label="sm121-chunked-ce-1node",
                nodes=1,
                micro_batch_size=4,
                grad_accum_steps=4,
                activation_checkpointing=False,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=False,
                use_fused_lora_kernel=False,
            ),
            TrainingPerfCandidate(
                label="sm121-all-kernels-1node",
                nodes=1,
                micro_batch_size=max_mb,
                grad_accum_steps=max(1, 16 // max_mb),
                activation_checkpointing=False,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            ),
            TrainingPerfCandidate(
                label="sm121-all-kernels-2node-ddp",
                nodes=2,
                micro_batch_size=max_mb,
                grad_accum_steps=max(1, 8 // max_mb),
                activation_checkpointing=False,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            ),
            TrainingPerfCandidate(
                label="oversized-global-batch-2node",
                nodes=2,
                micro_batch_size=max_mb,
                grad_accum_steps=8,
                activation_checkpointing=False,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            ),
        ]

    stage_b = StageBTrainingPerfConfig(
        loss_curve_tolerance_abs=0.025,
        verify_steps=120 if two_node_only else 180,
        candidates=b_cands,
    )

    if two_node_only:
        c_cands = [
            InferencePerfCandidate(
                label="bf16-1replica-tp2",
                precision="BF16",
                tensor_parallel=2,
                replicas=1,
                max_num_seqs=32,
                use_low_prec_decode_kernel=False,
            ),
            InferencePerfCandidate(
                label="fp8-1replica-tp2",
                precision="FP8",
                tensor_parallel=2,
                replicas=1,
                max_num_seqs=64,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="nvfp4-2replicas-tp1",
                precision="NVFP4",
                tensor_parallel=1,
                replicas=2,
                max_num_seqs=64,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="nvfp4-aggressive-batch256",
                precision="NVFP4",
                tensor_parallel=1,
                replicas=2,
                max_num_seqs=256,
                use_low_prec_decode_kernel=True,
            ),
        ]
    else:
        c_cands = [
            InferencePerfCandidate(
                label="bf16-1replica-tp1",
                precision="BF16",
                tensor_parallel=1,
                replicas=1,
                max_num_seqs=32,
                use_low_prec_decode_kernel=False,
            ),
            InferencePerfCandidate(
                label="fp8-1replica-tp1",
                precision="FP8",
                tensor_parallel=1,
                replicas=1,
                max_num_seqs=64,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="fp8-2replicas-tp1",
                precision="FP8",
                tensor_parallel=1,
                replicas=2,
                max_num_seqs=64,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="nvfp4-2replicas-tp1",
                precision="NVFP4",
                tensor_parallel=1,
                replicas=2,
                max_num_seqs=96,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="nvfp4-1replica-tp2",
                precision="NVFP4",
                tensor_parallel=2,
                replicas=1,
                max_num_seqs=96,
                use_low_prec_decode_kernel=True,
            ),
            InferencePerfCandidate(
                label="nvfp4-aggressive-batch256",
                precision="NVFP4",
                tensor_parallel=1,
                replicas=2,
                max_num_seqs=256,
                use_low_prec_decode_kernel=True,
            ),
        ]

    stage_c = StageCInferencePerfConfig(
        max_perplexity_regression_pct=2.5,
        min_completion_pass_rate=0.84,
        p99_latency_ceiling_ms=550.0 if two_node_only else 450.0,
        candidates=c_cands,
    )

    return ExperimentConfig(
        experiment_id=exp_slug,
        description=f"Crucible 4-stage optimization of {profile.name} on {dataset_name}",
        target_definition_id=cluster.definition_id,
        model=profile,
        dataset=DatasetSplitConfig(
            name=_slugify(dataset_name),
            raw_subpath=raw_subpath,
            processed_subpath=f"{_slugify(dataset_name)}-packed",
            chunk_seq_len=2048,
            chunk_overlap=128,
            split_salt=f"crucible-{_slugify(dataset_name)}-v1",
        ),
        stage_a_recipe=stage_a,
        stage_b_training_perf=stage_b,
        stage_c_inference_perf=stage_c,
    )
