"""Stage B (Training Performance Search) & Stage C (Inference Performance Search)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from crucible.config import (
    RecipeCandidate,
    StageBTrainingPerfConfig,
    StageCInferencePerfConfig,
    TrainingPerfCandidate,
)
from crucible.fuzzball.runner import ExecutionBackend
from crucible.serving.vllm_bench import ServingTrialSummary
from crucible.trainer.types import TrainingTrialSummary


@dataclass(frozen=True)
class TrainingPerfTrialOutcome:
    trial_id: str
    label: str
    candidate: dict[str, object]
    summary: dict[str, object]
    max_loss_curve_delta: float
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageBResult:
    baseline_outcome: TrainingPerfTrialOutcome
    winning_candidate: TrainingPerfCandidate
    winning_outcome: TrainingPerfTrialOutcome
    overall_speedup_x: float
    single_node_kernel_speedup_x: float
    outcomes: list[TrainingPerfTrialOutcome]


@dataclass(frozen=True)
class StageCResult:
    baseline_summary: ServingTrialSummary
    winning_candidate_label: str
    winning_summary: ServingTrialSummary
    serving_speedup_x: float
    outcomes: list[ServingTrialSummary]


def _max_curve_deviation(ref_curve: list[float], cand_curve: list[float]) -> float:
    if not ref_curve or not cand_curve:
        return 999.0
    n = min(len(ref_curve), len(cand_curve))
    # Compare smoothed checkpoints along the trajectory (last half + endpoints) to avoid benign step noise
    indices = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return round(max(abs(ref_curve[i] - cand_curve[i]) for i in indices), 5)


def run_stage_b_training_perf_search(
    cfg: StageBTrainingPerfConfig,
    winning_recipe: RecipeCandidate,
    backend: ExecutionBackend,
) -> StageBResult:
    if not cfg.candidates:
        raise ValueError("Stage B requires at least one training performance candidate.")

    outcomes: list[TrainingPerfTrialOutcome] = []
    ref_curve: list[float] | None = None

    for idx, cand in enumerate(cfg.candidates):
        stop_ev = getattr(backend, "stop_event", None)
        if idx >= 1 and stop_ev is not None and stop_ev.is_set() and getattr(backend, "backend_name", "") in {"fuzzball_mcp", "fuzzball"}:
            break
        trial_id = f"stageb-{idx:02d}-{cand.label}"
        summary: TrainingTrialSummary = backend.run_training_trial(
            trial_id=trial_id,
            recipe=winning_recipe,
            perf=cand,
            max_steps=cfg.verify_steps,
            reference_recipe=winning_recipe,
        )
        if ref_curve is None:
            ref_curve = summary.loss_curve

        if summary.status == "oom":
            outcomes.append(
                TrainingPerfTrialOutcome(
                    trial_id=trial_id,
                    label=cand.label,
                    candidate=cand.model_dump(),
                    summary=summary.to_dict(),
                    max_loss_curve_delta=999.0,
                    accepted=False,
                    rejection_reason=f"OOM ({summary.peak_mem_gb:.1f} GB exceeded node capacity)",
                )
            )
            continue

        delta = _max_curve_deviation(ref_curve, summary.loss_curve)
        accepted = delta <= cfg.loss_curve_tolerance_abs
        reason = (
            None
            if accepted
            else f"loss_curve_delta={delta:.4f} exceeded tolerance {cfg.loss_curve_tolerance_abs:.4f}"
        )
        outcomes.append(
            TrainingPerfTrialOutcome(
                trial_id=trial_id,
                label=cand.label,
                candidate=cand.model_dump(),
                summary=summary.to_dict(),
                max_loss_curve_delta=delta,
                accepted=accepted,
                rejection_reason=reason,
            )
        )

    baseline = outcomes[0]
    accepted_pairs = [
        (cand, out) for cand, out in zip(cfg.candidates, outcomes) if out.accepted
    ]
    if not accepted_pairs:
        accepted_pairs = [(cfg.candidates[0], baseline)]

    win_cand, win_out = max(
        accepted_pairs,
        key=lambda pair: float(pair[1].summary["mean_tokens_per_sec"]),
    )

    base_tok_s = max(float(baseline.summary["mean_tokens_per_sec"]), 1e-6)
    best_tok_s = float(win_out.summary["mean_tokens_per_sec"])

    single_node_kernel_tok_s = base_tok_s
    for cand, out in accepted_pairs:
        if cand.nodes == 1 and (
            cand.use_chunked_ce_kernel or cand.use_grouped_moe_kernel or cand.use_fused_lora_kernel
        ):
            single_node_kernel_tok_s = max(
                single_node_kernel_tok_s, float(out.summary["mean_tokens_per_sec"])
            )

    return StageBResult(
        baseline_outcome=baseline,
        winning_candidate=win_cand,
        winning_outcome=win_out,
        overall_speedup_x=round(best_tok_s / base_tok_s, 2),
        single_node_kernel_speedup_x=round(single_node_kernel_tok_s / base_tok_s, 2),
        outcomes=outcomes,
    )


def run_stage_c_inference_perf_search(
    cfg: StageCInferencePerfConfig,
    baseline_finetuned_loss: float,
    backend: ExecutionBackend,
) -> StageCResult:
    if not cfg.candidates:
        raise ValueError("Stage C requires at least one inference performance candidate.")

    outcomes: list[ServingTrialSummary] = []
    for idx, cand in enumerate(cfg.candidates):
        stop_ev = getattr(backend, "stop_event", None)
        if idx >= 1 and stop_ev is not None and stop_ev.is_set() and getattr(backend, "backend_name", "") in {"fuzzball_mcp", "fuzzball"}:
            break
        trial_id = f"stagec-{idx:02d}-{cand.label}"
        summ = backend.run_serving_trial(
            trial_id=trial_id,
            cand=cand,
            baseline_finetuned_loss=baseline_finetuned_loss,
        )
        outcomes.append(summ)

    baseline = outcomes[0]
    accepted = [o for o in outcomes if o.accepted]
    winner = max(accepted, key=lambda o: o.throughput_tokens_per_sec) if accepted else baseline

    return StageCResult(
        baseline_summary=baseline,
        winning_candidate_label=winner.label,
        winning_summary=winner,
        serving_speedup_x=round(
            winner.throughput_tokens_per_sec / max(baseline.throughput_tokens_per_sec, 1e-6), 2
        ),
        outcomes=outcomes,
    )
