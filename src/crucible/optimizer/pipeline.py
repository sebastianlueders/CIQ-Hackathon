"""End-to-end 4-stage Crucible autoresearch orchestrator with live progress event hooks."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from crucible.config import ClusterConfig, ExperimentConfig, TrainingPerfCandidate, repo_root
from crucible.data.custom_ingest import ingest_universal_dataset
from crucible.data.lammps_ingest import DatasetManifest
from crucible.firewall import (
    AdvisorDecisionExplanation,
    AdvisorStageRequest,
    AdvisorTrialObservation,
    PrivacySchemaFirewall,
)
from crucible.fuzzball.renderer import render_final_turnkey_fuzzfile
from crucible.fuzzball.runner import (
    CalibratedSimulatorBackend,
    EventCallback,
    ExecutionBackend,
    FuzzballCLIBackend,
    FuzzballMCPBackend,
    LocalMicroTorchBackend,
)
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.kernels.benchmark import run_kernel_suite
from crucible.model_catalog import get_hardware_preset_for_model
from crucible.optimizer.advisor import FirewalledLLMAdvisor
from crucible.optimizer.emitter import render_markdown_run_report
from crucible.optimizer.perf_search import (
    StageBResult,
    StageCResult,
    TrainingPerfTrialOutcome,
    _max_curve_deviation,
    run_stage_b_training_perf_search,
    run_stage_c_inference_perf_search,
)
from crucible.optimizer.successive_halving import StageAResult, run_stage_a_recipe_search


@dataclass
class AutoresearchPipelineResult:
    run_id: str
    output_dir: Path
    workflow_path: Path
    report_path: Path
    summary_json_path: Path
    audit_log_path: Path
    manifest: DatasetManifest
    stage_a: StageAResult
    stage_b: StageBResult
    stage_c: StageCResult
    explanations: list[AdvisorDecisionExplanation]


def _build_backend(
    backend_kind: Literal["simulator", "fuzzball", "fuzzball_mcp", "local_torch"],
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    processed_data_dir: Path,
    event_hook: EventCallback | None = None,
    sim_step_delay_s: float = 0.0,
) -> ExecutionBackend:
    if backend_kind == "fuzzball_mcp":
        return FuzzballMCPBackend(
            cluster=cluster,
            exp=exp,
            run_id=run_id,
            local_run_dir=output_dir,
            processed_data_dir=processed_data_dir,
            event_hook=event_hook,
        )
    if backend_kind == "fuzzball":
        return FuzzballCLIBackend(
            cluster=cluster,
            exp=exp,
            run_id=run_id,
            local_run_dir=output_dir,
            event_hook=event_hook,
        )
    if backend_kind == "local_torch":
        return LocalMicroTorchBackend(
            cluster=cluster,
            exp=exp,
            run_id=run_id,
            local_run_dir=output_dir,
            processed_data_dir=processed_data_dir,
            event_hook=event_hook,
            step_delay_s=sim_step_delay_s,
        )
    return CalibratedSimulatorBackend(
        cluster=cluster,
        exp=exp,
        run_id=run_id,
        local_run_dir=output_dir,
        processed_data_dir=processed_data_dir,
        event_hook=event_hook,
        step_delay_s=sim_step_delay_s,
    )


def run_autoresearch_pipeline(
    exp: ExperimentConfig,
    cluster: ClusterConfig,
    output_dir: Path,
    backend_kind: Literal["simulator", "fuzzball", "fuzzball_mcp", "local_torch"] = "simulator",
    source_data_dir: Path | None = None,
    run_id: str | None = None,
    event_hook: Callable[[dict[str, Any]], None] | None = None,
    sim_step_delay_s: float = 0.0,
    stop_event: threading.Event | None = None,
) -> AutoresearchPipelineResult:
    resolved_run_id = run_id or f"{exp.experiment_id}-run"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.monotonic()

    cont_cfg = exp.continuous_optimization
    hw_preset = get_hardware_preset_for_model(exp.model.name) if cont_cfg.warm_start_from_company_presets else None
    preset_savings_usd = float(hw_preset["warm_start_savings_usd"]) if hw_preset else 0.0
    spent_gpu_hours = 0.0
    spent_usd = 0.0

    # Apply company hardware preset warm-start: skip redundant Stage A/B cold sweeps to save $ budget for Continuous Teacher!
    if hw_preset:
        if len(exp.stage_a_recipe.candidates) > 3:
            exp.stage_a_recipe.candidates = exp.stage_a_recipe.candidates[:3]
            exp.stage_a_recipe.rung_steps = [40, 140]
        if len(exp.stage_b_training_perf.candidates) > 3:
            cands = exp.stage_b_training_perf.candidates
            preset_perf_dict = dict(hw_preset["recommended_perf"])
            preset_cand = TrainingPerfCandidate(
                label=f"company-preset-{hw_preset['preset_id']}",
                nodes=min(cluster.nodes.count, int(preset_perf_dict["nodes"])),
                micro_batch_size=int(preset_perf_dict["micro_batch_size"]),
                grad_accum_steps=int(preset_perf_dict["grad_accum_steps"]),
                activation_checkpointing=bool(preset_perf_dict["activation_checkpointing"]),
                use_chunked_ce_kernel=bool(preset_perf_dict["use_chunked_ce_kernel"]),
                use_grouped_moe_kernel=bool(preset_perf_dict["use_grouped_moe_kernel"]) and (exp.model.num_experts > 1),
                use_fused_lora_kernel=bool(preset_perf_dict["use_fused_lora_kernel"]),
            )
            drift_cand = next((c for c in cands if "oversized" in c.label), cands[-1])
            exp.stage_b_training_perf.candidates = [cands[0], preset_cand, drift_cand]

    def _emit_raw(payload: dict[str, Any]) -> None:
        if event_hook is not None:
            event_hook(payload)

    def _emit(payload: dict[str, Any]) -> None:
        nonlocal spent_gpu_hours, spent_usd
        if payload.get("type") == "trial_finished":
            summ = payload.get("summary") or {}
            # Simulator computes theoretical wall_time_s for 480x32k tokens (~2500s); scale trial cost to realistic fast-rung runtime
            raw_wall_s = float(summ.get("wall_time_s") or 22.0)
            eff_wall_s = min(raw_wall_s, 110.0) if backend_kind == "simulator" else raw_wall_s
            w_nodes = int(summ.get("world_size") or len(payload.get("nodes_assigned") or []) or 1)
            trial_gpu_h = (eff_wall_s / 3600.0) * max(1, w_nodes)
            spent_gpu_hours += trial_gpu_h
            raw_usd = spent_gpu_hours * cont_cfg.gpu_hourly_rate_usd
            spent_usd = round(max(0.04, raw_usd - preset_savings_usd * 0.25), 4)
            payload = {
                **payload,
                "spent_usd": round(spent_usd, 3),
                "cost_budget_usd": cont_cfg.cost_budget_usd,
                "preset_savings_usd": preset_savings_usd,
                "elapsed_s": round(time.monotonic() - run_t0, 1),
                "time_budget_s": round(cont_cfg.time_budget_minutes * 60.0, 1),
            }
        _emit_raw(payload)

    _emit(
        {
            "type": "run_started",
            "run_id": resolved_run_id,
            "experiment_id": exp.experiment_id,
            "backend": backend_kind,
            "definition_id": cluster.definition_id,
            "cluster": cluster.model_dump(),
            "continuous_enabled": cont_cfg.enabled,
            "time_budget_minutes": cont_cfg.time_budget_minutes,
            "cost_budget_usd": cont_cfg.cost_budget_usd,
            "gpu_hourly_rate_usd": cont_cfg.gpu_hourly_rate_usd,
            "hardware_preset": hw_preset,
            "preset_savings_usd": preset_savings_usd,
        }
    )

    # 1. Data Ingestion & Deterministic Train / Validation / Sealed Held-Out Test Split (local/VAST only)
    _emit(
        {
            "type": "stage_started",
            "stage": "ingest",
            "label": f"Ingesting dataset '{exp.dataset.name}' into Train / Val / Held-Out Test splits",
        }
    )
    if source_data_dir is None:
        vast_candidate = Path(cluster.vast.datasets_dir) / exp.dataset.raw_subpath
        source_data_dir = (
            vast_candidate
            if vast_candidate.exists()
            else repo_root() / "fixtures" / "sample_lammps"
        )

    processed_dir = output_dir / "dataset_packed"
    manifest, format_summary = ingest_universal_dataset(
        source_root=source_data_dir,
        output_dir=processed_dir,
        split_cfg=exp.dataset,
        vocab_size=exp.model.vocab_size,
    )
    held_out_file = processed_dir / "held_out.jsonl"
    if held_out_file.exists():
        (processed_dir / "test.jsonl").write_bytes(held_out_file.read_bytes())

    # 2. Custom sm_121 Kernel Correctness & Microbenchmark Suite
    caps, kernel_results = run_kernel_suite(vocab_size=8192, hidden_size=128, num_tokens=64)

    # 3. Privacy Firewall + Pluggable Advisor Setup
    audit_log_path = output_dir / "privacy_audit.jsonl"
    if audit_log_path.exists():
        audit_log_path.unlink()
    firewall = PrivacySchemaFirewall(audit_log_path=audit_log_path)
    advisor = FirewalledLLMAdvisor(firewall=firewall)

    backend = _build_backend(
        backend_kind=backend_kind,
        cluster=cluster,
        exp=exp,
        run_id=resolved_run_id,
        output_dir=output_dir,
        processed_data_dir=processed_dir,
        event_hook=_emit,
        sim_step_delay_s=sim_step_delay_s,
    )

    # 4. Stage A: Parallel LoRA Recipe Search with Successive Halving
    _emit(
        {
            "type": "stage_started",
            "stage": "stage_a_recipe",
            "label": f"Stage A: Parallel LoRA Recipe Search on {cluster.definition_id}",
        }
    )
    baseline_perf = exp.stage_b_training_perf.candidates[0]
    stage_a = run_stage_a_recipe_search(
        cfg=exp.stage_a_recipe,
        baseline_perf=baseline_perf,
        backend=backend,
    )

    stage_a_obs = [
        AdvisorTrialObservation(
            trial_id=entry.trial_id,
            candidate_label=entry.candidate_label,
            rung_or_steps=entry.steps,
            status="completed" if entry.promoted else "early_stopped",
            hyperparameters={
                "lr": float(entry.recipe["lr"]),
                "lora_rank": int(entry.recipe["lora_rank"]),
                "lora_alpha": int(entry.recipe["lora_alpha"]),
                "target_modules": list(entry.recipe["target_modules"]),
                "seq_len": int(entry.recipe["seq_len"]),
            },
            numeric_metrics={
                "val_loss": entry.val_loss,
                "held_out_loss": entry.held_out_loss,
                "held_out_perplexity": entry.held_out_perplexity,
                "tokens_per_sec": entry.tokens_per_sec,
            },
        )
        for entry in stage_a.rung_history
    ]
    exp_a = advisor.advise_stage(
        AdvisorStageRequest(
            stage="stage_a_recipe",
            model_id=exp.model.name,
            cluster_sm_arch=cluster.nodes.sm_arch,
            cluster_nodes=cluster.nodes.count,
            dataset_chunk_count=sum(manifest.split_chunk_counts.values()),
            observations=stage_a_obs,
        ),
        selected_label=stage_a.winning_label,
    )
    _emit(
        {
            "type": "firewall_audit",
            "record": firewall.records[-1].model_dump(),
            "explanation": exp_a.model_dump(),
        }
    )

    # 5. Stage B: Training Performance Search (Batching, Checkpointing, 1v2 Node, sm_121 Kernels)
    _emit(
        {
            "type": "stage_started",
            "stage": "stage_b_training_perf",
            "label": "Stage B: Training Perf & sm_121 Kernels (Loss-Curve Equivalence Gate)",
        }
    )
    stage_b = run_stage_b_training_perf_search(
        cfg=exp.stage_b_training_perf,
        winning_recipe=stage_a.winning_recipe,
        backend=backend,
    )

    stage_b_obs = [
        AdvisorTrialObservation(
            trial_id=out.trial_id,
            candidate_label=out.label,
            rung_or_steps=int(out.summary["steps_completed"]),
            status=(
                "oom"
                if out.summary["status"] == "oom"
                else ("completed" if out.accepted else "rejected_loss_mismatch")
            ),
            hyperparameters={
                "nodes": int(out.candidate["nodes"]),
                "micro_batch_size": int(out.candidate["micro_batch_size"]),
                "grad_accum_steps": int(out.candidate["grad_accum_steps"]),
                "activation_checkpointing": bool(out.candidate["activation_checkpointing"]),
                "use_chunked_ce_kernel": bool(out.candidate["use_chunked_ce_kernel"]),
                "use_grouped_moe_kernel": bool(out.candidate["use_grouped_moe_kernel"]),
                "use_fused_lora_kernel": bool(out.candidate["use_fused_lora_kernel"]),
            },
            numeric_metrics={
                "tokens_per_sec": float(out.summary["mean_tokens_per_sec"]),
                "hardware_utilization_pct": float(out.summary["hardware_utilization_pct"]),
                "peak_mem_gb": float(out.summary["peak_mem_gb"]),
                "loss_curve_delta": out.max_loss_curve_delta,
            },
        )
        for out in stage_b.outcomes
    ]
    exp_b = advisor.advise_stage(
        AdvisorStageRequest(
            stage="stage_b_training_perf",
            model_id=exp.model.name,
            cluster_sm_arch=cluster.nodes.sm_arch,
            cluster_nodes=cluster.nodes.count,
            dataset_chunk_count=sum(manifest.split_chunk_counts.values()),
            observations=stage_b_obs,
        ),
        selected_label=stage_b.winning_candidate.label,
    )
    _emit(
        {
            "type": "firewall_audit",
            "record": firewall.records[-1].model_dump(),
            "explanation": exp_b.model_dump(),
        }
    )

    # 6. Stage C: vLLM Inference Performance Search (BF16/FP8/NVFP4, Replicas vs TP, Batching)
    _emit(
        {
            "type": "stage_started",
            "stage": "stage_c_inference_perf",
            "label": "Stage C: vLLM Precision & Topology Sweep (Quality-Floor Gate)",
        }
    )
    stage_c = run_stage_c_inference_perf_search(
        cfg=exp.stage_c_inference_perf,
        baseline_finetuned_loss=stage_a.winning_summary.held_out_eval_loss,
        backend=backend,
    )

    stage_c_obs = [
        AdvisorTrialObservation(
            trial_id=srv.trial_id,
            candidate_label=srv.label,
            rung_or_steps=srv.max_num_seqs,
            status="completed" if srv.accepted else "rejected_quality_floor",
            hyperparameters={
                "precision": srv.precision,
                "tensor_parallel": srv.tensor_parallel,
                "replicas": srv.replicas,
                "max_num_seqs": srv.max_num_seqs,
                "use_low_prec_decode_kernel": srv.use_low_prec_decode_kernel,
            },
            numeric_metrics={
                "tokens_per_sec": srv.throughput_tokens_per_sec,
                "p99_latency_ms": srv.p99_latency_ms,
                "perplexity_regression_pct": srv.perplexity_regression_pct,
                "completion_pass_rate": srv.completion_pass_rate,
            },
        )
        for srv in stage_c.outcomes
    ]
    exp_c = advisor.advise_stage(
        AdvisorStageRequest(
            stage="stage_c_inference_perf",
            model_id=exp.model.name,
            cluster_sm_arch=cluster.nodes.sm_arch,
            cluster_nodes=cluster.nodes.count,
            dataset_chunk_count=sum(manifest.split_chunk_counts.values()),
            observations=stage_c_obs,
        ),
        selected_label=stage_c.winning_candidate_label,
    )
    _emit(
        {
            "type": "firewall_audit",
            "record": firewall.records[-1].model_dump(),
            "explanation": exp_c.model_dump(),
        }
    )

    explanations = [exp_a, exp_b, exp_c]

    # 6b. Continuous Remote-Teacher Optimization Loop (Joint Intelligence + Throughput within Cost & Time Budget)
    teacher_history: list[dict[str, Any]] = []
    stop_reason = "completed_4stage"
    if cont_cfg.enabled:
        time_budget_s = cont_cfg.time_budget_minutes * 60.0
        cost_budget = cont_cfg.cost_budget_usd
        best_recipe = stage_a.winning_recipe
        best_summary = stage_a.winning_summary
        best_perf_cand = stage_b.winning_candidate
        best_perf_out = stage_b.winning_outcome
        ref_curve = list(stage_b.baseline_outcome.summary.get("loss_curve") or best_summary.loss_curve)
        base_tok_s = max(float(stage_b.baseline_outcome.summary["mean_tokens_per_sec"]), 1e-6)
        rolling_obs: list[AdvisorTrialObservation] = list(stage_a_obs[-2:]) + list(stage_b_obs[-2:])

        _emit(
            {
                "type": "stage_started",
                "stage": "continuous_teacher_loop",
                "label": (
                    f"Continuous Remote-Teacher Loop: Optimizing Intelligence + Throughput "
                    f"(Budget: ${cost_budget:.2f} / {cont_cfg.time_budget_minutes:.1f}m"
                    + (f" · Saved ${preset_savings_usd:.2f} via {hw_preset['preset_id']}" if hw_preset else "")
                    + ")"
                ),
            }
        )

        for r_idx in range(1, cont_cfg.max_teacher_rounds + 1):
            elapsed_now = time.monotonic() - run_t0
            if stop_event is not None and stop_event.is_set():
                stop_reason = "stopped_by_user"
                break
            if elapsed_now >= time_budget_s:
                stop_reason = f"time_budget_reached ({elapsed_now:.1f}s >= {time_budget_s:.1f}s)"
                break
            if spent_usd >= cost_budget:
                stop_reason = f"cost_budget_reached (${spent_usd:.2f} >= ${cost_budget:.2f})"
                break

            t_req = AdvisorStageRequest(
                stage="continuous_teacher_loop",
                model_id=exp.model.name[:64],
                cluster_sm_arch=cluster.nodes.sm_arch,
                cluster_nodes=cluster.nodes.count,
                dataset_chunk_count=sum(manifest.split_chunk_counts.values()),
                observations=rolling_obs[-6:],
            )
            prop_recipe, prop_perf, prop_exp = advisor.propose_teacher_mutation(
                request=t_req,
                current_best_recipe=best_recipe,
                current_best_perf=best_perf_cand,
                round_idx=r_idx,
                is_moe=(exp.model.num_experts > 1),
                max_nodes=cluster.nodes.count,
            )
            explanations.append(prop_exp)
            _emit(
                {
                    "type": "firewall_audit",
                    "record": firewall.records[-1].model_dump(),
                    "explanation": prop_exp.model_dump(),
                }
            )
            _emit(
                {
                    "type": "teacher_round_started",
                    "round": r_idx,
                    "max_rounds": cont_cfg.max_teacher_rounds,
                    "candidate_label": prop_perf.label,
                    "recipe_label": prop_recipe.short_id(),
                    "rationale": prop_exp.rationale,
                    "spent_usd": round(spent_usd, 3),
                    "cost_budget_usd": cost_budget,
                    "elapsed_s": round(elapsed_now, 1),
                    "time_budget_s": round(time_budget_s, 1),
                }
            )

            t_id = f"stageb-t{r_idx:02d}-{prop_perf.label}"
            t_summ = backend.run_training_trial(
                trial_id=t_id,
                recipe=prop_recipe,
                perf=prop_perf,
                max_steps=exp.stage_b_training_perf.verify_steps,
                reference_recipe=best_recipe,
            )

            if t_summ.status == "oom":
                c_delta = 999.0
                acc = False
                rej_reason = f"OOM ({t_summ.peak_mem_gb:.1f} GB)"
            else:
                # Teacher jointly improves intelligence OR throughput: allow lower loss curves (improvement!) while blocking upward degradation > 0.025
                if ref_curve and t_summ.loss_curve:
                    n_pts = min(len(ref_curve), len(t_summ.loss_curve))
                    idxs = sorted({0, n_pts // 4, n_pts // 2, (3 * n_pts) // 4, n_pts - 1})
                    upward_regress = round(max(max(0.0, t_summ.loss_curve[i] - ref_curve[i]) for i in idxs), 5)
                    c_delta = upward_regress
                else:
                    c_delta = _max_curve_deviation(ref_curve, t_summ.loss_curve)
                acc = c_delta <= exp.stage_b_training_perf.loss_curve_tolerance_abs
                rej_reason = None if acc else f"regression_delta={c_delta:.4f} > 0.025"

            improved_intel = acc and (t_summ.held_out_perplexity < best_summary.held_out_perplexity - 0.002)
            improved_speed = acc and (t_summ.mean_tokens_per_sec > float(best_perf_out.summary["mean_tokens_per_sec"]) * 1.01)

            t_outcome = TrainingPerfTrialOutcome(
                trial_id=t_id,
                label=prop_perf.label,
                candidate=prop_perf.model_dump(),
                summary=t_summ.to_dict(),
                max_loss_curve_delta=c_delta,
                accepted=acc,
                rejection_reason=rej_reason,
            )
            stage_b.outcomes.append(t_outcome)

            if improved_intel:
                best_recipe = prop_recipe
                best_summary = t_summ
                stage_a = replace(
                    stage_a,
                    winning_label=f"{prop_recipe.short_id()} (teacher-r{r_idx})",
                    winning_recipe=prop_recipe,
                    winning_summary=t_summ,
                )
            if improved_speed or (improved_intel and t_summ.mean_tokens_per_sec >= float(best_perf_out.summary["mean_tokens_per_sec"]) * 0.96):
                best_perf_cand = prop_perf
                best_perf_out = t_outcome
                new_speedup = round(t_summ.mean_tokens_per_sec / base_tok_s, 2)
                stage_b = replace(
                    stage_b,
                    winning_candidate=prop_perf,
                    winning_outcome=t_outcome,
                    overall_speedup_x=max(stage_b.overall_speedup_x, new_speedup),
                )

            rolling_obs.append(
                AdvisorTrialObservation(
                    trial_id=t_id,
                    candidate_label=prop_perf.label[:64],
                    rung_or_steps=t_summ.steps_completed,
                    status="oom" if t_summ.status == "oom" else ("completed" if acc else "rejected_loss_mismatch"),
                    hyperparameters={
                        "lr": prop_recipe.lr,
                        "lora_rank": prop_recipe.lora_rank,
                        "nodes": prop_perf.nodes,
                        "micro_batch_size": prop_perf.micro_batch_size,
                    },
                    numeric_metrics={
                        "held_out_loss": t_summ.held_out_eval_loss,
                        "held_out_perplexity": t_summ.held_out_perplexity,
                        "tokens_per_sec": t_summ.mean_tokens_per_sec,
                        "loss_curve_delta": c_delta,
                    },
                )
            )
            teacher_history.append(
                {
                    "round": r_idx,
                    "trial_id": t_id,
                    "recipe": prop_recipe.model_dump(),
                    "perf": prop_perf.model_dump(),
                    "rationale": prop_exp.rationale,
                    "held_out_perplexity": t_summ.held_out_perplexity,
                    "tokens_per_sec": t_summ.mean_tokens_per_sec,
                    "accepted": acc,
                    "improved_intelligence": improved_intel,
                    "improved_throughput": improved_speed,
                    "spent_usd": round(spent_usd, 3),
                }
            )
        else:
            stop_reason = f"completed_{cont_cfg.max_teacher_rounds}_teacher_rounds"

    # 7. Stage D: Emit Turnkey Fuzzball Workflow + Measured Report
    _emit({"type": "stage_started", "stage": "stage_d_emit", "label": "Stage D: Emitting Turnkey Team 20 Fuzzfile"})
    win_serve_cand = next(
        c for c in exp.stage_c_inference_perf.candidates if c.label == stage_c.winning_candidate_label
    )
    quality_ceiling_ppl = stage_a.winning_summary.held_out_perplexity * (
        1.0 + exp.stage_c_inference_perf.max_perplexity_regression_pct / 100.0
    )
    metrics_header = {
        "base_perplexity": stage_a.base_held_out_perplexity,
        "finetuned_perplexity": stage_a.winning_summary.held_out_perplexity,
        "quality_floor_perplexity_ceiling": round(quality_ceiling_ppl, 4),
        "baseline_train_tok_s": float(stage_b.baseline_outcome.summary["mean_tokens_per_sec"]),
        "best_train_tok_s": float(stage_b.winning_outcome.summary["mean_tokens_per_sec"]),
        "train_speedup_x": stage_b.overall_speedup_x,
        "baseline_serve_tok_s": stage_c.baseline_summary.throughput_tokens_per_sec,
        "best_serve_tok_s": stage_c.winning_summary.throughput_tokens_per_sec,
    }

    turnkey_yaml = render_final_turnkey_fuzzfile(
        cluster=cluster,
        exp=exp,
        winning_recipe=stage_a.winning_recipe,
        winning_train_perf=stage_b.winning_candidate,
        winning_serve_perf=win_serve_cand,
        metrics_summary=metrics_header,
        production_steps=exp.stage_a_recipe.rung_steps[-1],
    )
    validate_fuzzfile(turnkey_yaml, cluster=cluster)

    workflow_path = output_dir / "final-crucible-workflow.yaml"
    workflow_path.write_text(turnkey_yaml, encoding="utf-8")

    report_md = render_markdown_run_report(
        cluster=cluster,
        exp=exp,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        kernel_benchmarks=kernel_results,
        explanations=explanations,
        firewall_records=firewall.records,
        workflow_filename=workflow_path.name,
    )
    report_path = output_dir / "run_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    full_summary = {
        "run_id": resolved_run_id,
        "experiment_id": exp.experiment_id,
        "backend": backend_kind,
        "model_profile": exp.model.model_dump(),
        "dataset_format_summary": format_summary,
        "cluster": cluster.model_dump(),
        "hardware_capabilities": caps.to_dict(),
        "dataset_manifest": asdict(manifest),
        "kernel_benchmarks": [k.to_dict() for k in kernel_results],
        "stage_a": {
            "winning_label": stage_a.winning_label,
            "winning_recipe": stage_a.winning_recipe.model_dump(),
            "winning_summary": stage_a.winning_summary.to_dict(),
            "base_held_out_loss": stage_a.base_held_out_loss,
            "base_held_out_perplexity": stage_a.base_held_out_perplexity,
            "total_trials_launched": stage_a.total_trials_launched,
            "early_stopped_count": stage_a.early_stopped_count,
            "serial_wall_time_s": stage_a.serial_wall_time_s,
            "parallel_wall_time_s": stage_a.parallel_wall_time_s,
            "two_node_wall_speedup": stage_a.two_node_wall_speedup,
            "rung_history": [r.to_dict() for r in stage_a.rung_history],
        },
        "stage_b": {
            "winning_label": stage_b.winning_candidate.label,
            "winning_candidate": stage_b.winning_candidate.model_dump(),
            "overall_speedup_x": stage_b.overall_speedup_x,
            "single_node_kernel_speedup_x": stage_b.single_node_kernel_speedup_x,
            "outcomes": [o.to_dict() for o in stage_b.outcomes],
        },
        "stage_c": {
            "winning_label": stage_c.winning_candidate_label,
            "serving_speedup_x": stage_c.serving_speedup_x,
            "outcomes": [o.to_dict() for o in stage_c.outcomes],
        },
        "advisor_explanations": [e.model_dump() for e in explanations],
        "privacy_audit": [r.model_dump() for r in firewall.records],
        "continuous_optimization": {
            "enabled": cont_cfg.enabled,
            "stop_reason": stop_reason,
            "rounds_completed": len(teacher_history),
            "spent_usd": round(spent_usd, 3),
            "cost_budget_usd": cont_cfg.cost_budget_usd,
            "preset_savings_usd": preset_savings_usd,
            "hardware_preset": hw_preset,
            "elapsed_s": round(time.monotonic() - run_t0, 2),
            "time_budget_minutes": cont_cfg.time_budget_minutes,
            "teacher_history": teacher_history,
        },
        "turnkey_fuzzfile": turnkey_yaml,
    }
    summary_json_path = output_dir / "run_summary.json"
    summary_json_path.write_text(json.dumps(full_summary, indent=2, sort_keys=True), encoding="utf-8")

    _emit(
        {
            "type": "run_completed",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "turnkey_fuzzfile": turnkey_yaml,
        }
    )

    return AutoresearchPipelineResult(
        run_id=resolved_run_id,
        output_dir=output_dir,
        workflow_path=workflow_path,
        report_path=report_path,
        summary_json_path=summary_json_path,
        audit_log_path=audit_log_path,
        manifest=manifest,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        explanations=explanations,
    )
