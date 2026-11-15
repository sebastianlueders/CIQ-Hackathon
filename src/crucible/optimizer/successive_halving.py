"""Stage A: Deterministic Successive Halving optimizer across parallel cluster nodes."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from crucible.config import RecipeCandidate, StageARecipeConfig, TrainingPerfCandidate
from crucible.fuzzball.runner import ExecutionBackend
from crucible.trainer.types import TrainingTrialSummary


@dataclass(frozen=True)
class RecipeRungEntry:
    rung_index: int
    steps: int
    trial_id: str
    candidate_index: int
    candidate_label: str
    recipe: dict[str, object]
    val_loss: float
    held_out_loss: float
    held_out_perplexity: float
    tokens_per_sec: float
    wall_time_s: float
    promoted: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageAResult:
    winning_candidate_index: int
    winning_label: str
    winning_recipe: RecipeCandidate
    winning_summary: TrainingTrialSummary
    base_held_out_loss: float
    base_held_out_perplexity: float
    total_trials_launched: int
    early_stopped_count: int
    serial_wall_time_s: float
    parallel_wall_time_s: float
    two_node_wall_speedup: float
    rung_history: list[RecipeRungEntry]


def run_stage_a_recipe_search(
    cfg: StageARecipeConfig,
    baseline_perf: TrainingPerfCandidate,
    backend: ExecutionBackend,
) -> StageAResult:
    survivor_indices = list(range(len(cfg.candidates)))
    rung_history: list[RecipeRungEntry] = []
    summaries_by_trial: dict[str, TrainingTrialSummary] = {}

    serial_wall = 0.0
    parallel_wall = 0.0
    total_trials = 0
    early_stopped = 0

    for rung_idx, steps in enumerate(cfg.rung_steps):
        is_final_rung = rung_idx == len(cfg.rung_steps) - 1
        rung_results: list[tuple[int, str, TrainingTrialSummary]] = []
        batch_durations: list[float] = []

        def _launch_one(idx: int) -> tuple[int, str, TrainingTrialSummary]:
            cand = cfg.candidates[idx]
            trial_id = f"stagea-r{rung_idx}-c{idx}-{cand.short_id()}"
            summary = backend.run_training_trial(
                trial_id=trial_id,
                recipe=cand,
                perf=baseline_perf,
                max_steps=steps,
            )
            return idx, trial_id, summary

        # On live Fuzzball backends, submit the rung's candidate workflows concurrently across
        # `cfg.parallel_nodes` (2 GPUs) so GPU 0 and GPU 1 run in parallel without false "QUEUED" flicker!
        if getattr(backend, "backend_name", "") in {"fuzzball_mcp", "fuzzball"} and len(survivor_indices) > 1:
            workers = min(len(survivor_indices), max(1, cfg.parallel_nodes))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                launched = list(pool.map(_launch_one, survivor_indices))
        else:
            launched = [_launch_one(idx) for idx in survivor_indices]

        for idx, trial_id, summary in launched:
            total_trials += 1
            summaries_by_trial[trial_id] = summary
            rung_results.append((idx, trial_id, summary))
            serial_wall += summary.wall_time_s
            batch_durations.append(summary.wall_time_s)

        # Parallel scheduling across `cfg.parallel_nodes` nodes (1 trial per node per wave)
        for wave_start in range(0, len(batch_durations), cfg.parallel_nodes):
            wave = batch_durations[wave_start : wave_start + cfg.parallel_nodes]
            parallel_wall += max(wave) if wave else 0.0

        # Rank by validation loss on intermediate rungs; final winner picked by held-out eval loss
        if not is_final_rung:
            rung_results.sort(key=lambda item: item[2].final_val_loss)
            keep_n = max(1, math.ceil(len(rung_results) / cfg.reduction_factor))
        else:
            rung_results.sort(key=lambda item: item[2].held_out_eval_loss)
            keep_n = 1

        promoted_set = {item[0] for item in rung_results[:keep_n]}
        if not is_final_rung:
            early_stopped += len(rung_results) - len(promoted_set)

        for cand_idx, trial_id, summ in rung_results:
            cand = cfg.candidates[cand_idx]
            rung_history.append(
                RecipeRungEntry(
                    rung_index=rung_idx,
                    steps=steps,
                    trial_id=trial_id,
                    candidate_index=cand_idx,
                    candidate_label=cand.short_id(),
                    recipe=cand.model_dump(),
                    val_loss=summ.final_val_loss,
                    held_out_loss=summ.held_out_eval_loss,
                    held_out_perplexity=summ.held_out_perplexity,
                    tokens_per_sec=summ.mean_tokens_per_sec,
                    wall_time_s=summ.wall_time_s,
                    promoted=(cand_idx in promoted_set),
                )
            )

        survivor_indices = [item[0] for item in rung_results[:keep_n]]

    win_idx = survivor_indices[0]
    win_recipe = cfg.candidates[win_idx]
    final_entries = [e for e in rung_history if e.rung_index == len(cfg.rung_steps) - 1 and e.promoted]
    win_summary = summaries_by_trial[final_entries[0].trial_id]
    base_loss = win_summary.initial_val_loss

    return StageAResult(
        winning_candidate_index=win_idx,
        winning_label=win_recipe.short_id(),
        winning_recipe=win_recipe,
        winning_summary=win_summary,
        base_held_out_loss=round(base_loss, 5),
        base_held_out_perplexity=round(math.exp(min(base_loss, 20.0)), 4),
        total_trials_launched=total_trials,
        early_stopped_count=early_stopped,
        serial_wall_time_s=round(serial_wall, 2),
        parallel_wall_time_s=round(parallel_wall, 2),
        two_node_wall_speedup=round(serial_wall / max(parallel_wall, 1e-6), 2),
        rung_history=rung_history,
    )
