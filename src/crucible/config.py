"""Typed configuration models for cluster topology, search stages, and trial specs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeTopologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=2, ge=1)
    gpu_model: str = "NVIDIA GB10"
    sm_arch: str = "sm_121"
    gpus_per_node: int = Field(default=1, ge=1)
    cpu_cores_per_node: int = Field(default=20, ge=1)
    memory_gb_per_node: int = Field(default=112, ge=8)
    measured_peak_bf16_tflops: float = Field(default=198.0, gt=0)
    measured_peak_train_tok_s_per_node: float = Field(default=4150.0, gt=0)
    interconnect: str = "NCCL-RoCE-200GbE"


class VastStorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_key: str = "vast-shared"
    persistent_volume_name: str = "crucible-vast"
    provisioner: str = "vast"
    mount_path: str = "/vast"
    models_dir: str = "/vast/models"
    datasets_dir: str = "/vast/data"
    runs_dir: str = "/vast/runs"


class ContainerImagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fetch: str = "docker://nvcr.io/nvidia/pytorch:25.09-py3"
    train: str = "docker://nvcr.io/nvidia/pytorch:25.09-py3"
    serve: str = "docker://nvcr.io/nvidia/vllm:25.09-py3"


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_name: str = "ciq-gb10-hackathon"
    fuzzfile_version: Literal["v4", "v1"] = "v4"
    nodes: NodeTopologyConfig = Field(default_factory=NodeTopologyConfig)
    vast: VastStorageConfig = Field(default_factory=VastStorageConfig)
    images: ContainerImagesConfig = Field(default_factory=ContainerImagesConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ClusterConfig":
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg = cls.model_validate(raw)
        else:
            cfg = cls()
        return cfg.with_env_overrides()

    def with_env_overrides(self) -> "ClusterConfig":
        data = self.model_dump()
        if os.getenv("CRUCIBLE_FUZZFILE_VERSION") in {"v4", "v1"}:
            data["fuzzfile_version"] = os.environ["CRUCIBLE_FUZZFILE_VERSION"]
        if os.getenv("CRUCIBLE_VAST_VOLUME"):
            data["vast"]["persistent_volume_name"] = os.environ["CRUCIBLE_VAST_VOLUME"]
        if os.getenv("CRUCIBLE_VAST_PROVISIONER"):
            data["vast"]["provisioner"] = os.environ["CRUCIBLE_VAST_PROVISIONER"]
        if os.getenv("CRUCIBLE_TRAIN_IMAGE"):
            data["images"]["train"] = os.environ["CRUCIBLE_TRAIN_IMAGE"]
        if os.getenv("CRUCIBLE_SERVE_IMAGE"):
            data["images"]["serve"] = os.environ["CRUCIBLE_SERVE_IMAGE"]
        return ClusterConfig.model_validate(data)


class ModelProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Qwen/Qwen3-Coder-30B-A3B"
    weights_subpath: str = "Qwen3-Coder-30B-A3B"
    architecture: str = "qwen3_moe"
    total_params_b: float = 30.5
    active_params_b: float = 3.3
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_layers: int = 48
    num_experts: int = 128
    top_k_experts: int = 8
    bf16_footprint_gb: float = 57.0
    min_nodes_required: int = Field(default=1, ge=1)


class DatasetSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "lammps-cpp"
    raw_subpath: str = "lammps"
    processed_subpath: str = "lammps-packed"
    chunk_seq_len: int = Field(default=2048, ge=64)
    chunk_overlap: int = Field(default=128, ge=0)
    train_ratio: float = Field(default=0.80, gt=0, lt=1)
    val_ratio: float = Field(default=0.10, gt=0, lt=1)
    held_out_ratio: float = Field(default=0.10, gt=0, lt=1)
    split_salt: str = "crucible-lammps-v1"

    @model_validator(mode="after")
    def _validate_ratios(self) -> "DatasetSplitConfig":
        total = self.train_ratio + self.val_ratio + self.held_out_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        if self.chunk_overlap >= self.chunk_seq_len:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_seq_len")
        return self


ALLOWED_LORA_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


class RecipeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0, le=0.05)
    lora_rank: int = Field(ge=2, le=256)
    lora_alpha: int = Field(ge=2, le=512)
    target_modules: list[str]
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    seq_len: int = Field(default=2048, ge=128, le=32768)

    @model_validator(mode="after")
    def _check_modules(self) -> "RecipeCandidate":
        unknown = set(self.target_modules) - ALLOWED_LORA_MODULES
        if unknown:
            raise ValueError(f"Unsupported LoRA target_modules: {sorted(unknown)}")
        return self

    def short_id(self) -> str:
        mods = f"{len(self.target_modules)}m"
        return f"r{self.lora_rank}-a{self.lora_alpha}-lr{self.lr:.0e}-{mods}"


class StageARecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parallel_nodes: int = Field(default=2, ge=1)
    reduction_factor: int = Field(default=2, ge=2)
    rung_steps: list[int] = Field(default_factory=lambda: [60, 180, 480])
    candidates: list[RecipeCandidate]


class TrainingPerfCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    nodes: int = Field(default=1, ge=1)
    micro_batch_size: int = Field(default=2, ge=1)
    grad_accum_steps: int = Field(default=8, ge=1)
    activation_checkpointing: bool = True
    use_chunked_ce_kernel: bool = False
    use_grouped_moe_kernel: bool = False
    use_fused_lora_kernel: bool = False

    @property
    def global_batch_sequences(self) -> int:
        return self.nodes * self.micro_batch_size * self.grad_accum_steps


class StageBTrainingPerfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss_curve_tolerance_abs: float = Field(default=0.025, gt=0)
    verify_steps: int = Field(default=180, ge=10)
    candidates: list[TrainingPerfCandidate]


class InferencePerfCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    precision: Literal["BF16", "FP8", "NVFP4"] = "BF16"
    tensor_parallel: int = Field(default=1, ge=1)
    replicas: int = Field(default=1, ge=1)
    max_num_seqs: int = Field(default=32, ge=1)
    use_low_prec_decode_kernel: bool = False

    @property
    def total_gpus_required(self) -> int:
        return self.tensor_parallel * self.replicas


class StageCInferencePerfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_perplexity_regression_pct: float = Field(default=2.5, ge=0.0)
    min_completion_pass_rate: float = Field(default=0.84, ge=0.0, le=1.0)
    p99_latency_ceiling_ms: float = Field(default=450.0, gt=0.0)
    candidates: list[InferencePerfCandidate]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    description: str = ""
    model: ModelProfileConfig
    dataset: DatasetSplitConfig
    stage_a_recipe: StageARecipeConfig
    stage_b_training_perf: StageBTrainingPerfConfig
    stage_c_inference_perf: StageCInferencePerfConfig

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
