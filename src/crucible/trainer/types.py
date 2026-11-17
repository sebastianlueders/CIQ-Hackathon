"""Pure-Python telemetry and summary types for LoRA training trials (zero torch dependency)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class StepMetricRecord:
    step: int
    train_loss: float
    val_loss: float | None
    tokens_per_sec: float
    peak_mem_gb: float
    wall_time_s: float
    world_size: int


@dataclass(frozen=True)
class TrainingTrialSummary:
    trial_id: str
    status: Literal["completed", "early_stopped", "oom"]
    steps_completed: int
    initial_val_loss: float
    final_train_loss: float
    final_val_loss: float
    held_out_eval_loss: float
    held_out_perplexity: float
    mean_tokens_per_sec: float
    hardware_utilization_pct: float
    peak_mem_gb: float
    wall_time_s: float
    loss_curve: list[float]
    world_size: int
    kernels_enabled: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
