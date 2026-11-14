"""End-to-end 4-stage Crucible autoresearch orchestrator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from crucible.config import ClusterConfig, ExperimentConfig, repo_root
from crucible.data.lammps_ingest import DatasetManifest, ingest_lammps_corpus
from crucible.firewall import (
    AdvisorDecisionExplanation,
    AdvisorStageRequest,
    AdvisorTrialObservation,
    PrivacySchemaFirewall,
)
from crucible.fuzzball.renderer import render_final_turnkey_fuzzfile
from crucible.fuzzball.runner import (
    CalibratedSimulatorBackend,
    ExecutionBackend,
    FuzzballCLIBackend,
    LocalMicroTorchBackend,
)
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.kernels.benchmark import run_kernel_suite
from crucible.optimizer.advisor import FirewalledLLMAdvisor
from crucible.optimizer.emitter import render_markdown_run_report
from crucible.optimizer.perf_search import (
    StageBResult,
    StageCResult,
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
    backend_kind: Literal["simulator", "fuzzball", "local_torch"],
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    processed_data_dir: Path,
) -> ExecutionBackend:
    if backend_kind == "fuzzball":
        return FuzzballCLIBackend(
            cluster=cluster,
            exp=exp,
            run_id=run_id,
            local_run_dir=output_dir,
        )
    if backend_kind == "local_torch":
        return LocalMicroTorchBackend(
            cluster=cluster,
            exp=exp,
            run_id=run_id,
            local_run_dir=output_dir,
            processed_data_dir=processed_data_dir,
        )
    return CalibratedSimulatorBackend(
        cluster=cluster,
        exp=exp,
        run_id=run_id,
        local_run_dir=output_dir,
        processed_data_dir=processed_data_dir,
    )


def run_autoresearch_pipeline(
    exp: ExperimentConfig,
    cluster: ClusterConfig,
    output_dir: Path,
    backend_kind: Literal["simulator", "fuzzball", "local_torch"] = "simulator",
    source_data_dir: Path | None = None,
    run_id: str | None = None,
) -> AutoresearchPipelineResult:
    resolved_run_id = run_id or f"{exp.experiment_id}-run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Ingestion & Whole-File Sealed Split (local/VAST only)
    if source_data_dir is None:
        vast_candidate = Path(cluster.vast.datasets_dir) / exp.dataset.raw_subpath
        source_data_dir = (
            vast_candidate
            if vast_candidate.exists()
            else repo_root() / "fixtures" / "sample_lammps"
        )

    processed_dir = output_dir / "dataset_packed"
    manifest = ingest_lammps_corpus(
        source_root=source_data_dir,
        output_dir=processed_dir,
        split_cfg=exp.dataset,
        vocab_size=exp.model.vocab_size,
    )

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
    )

    # 4. Stage A: Parallel LoRA Recipe Search with Successive Halving
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

    # 5. Stage B: Training Performance Search (Batching, Checkpointing, 1v2 Node, sm_121 Kernels)
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

    # 6. Stage C: vLLM Inference Performance Search (BF16/FP8/NVFP4, Replicas vs TP, Batching)
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

    explanations = [exp_a, exp_b, exp_c]

    # 7. Stage D: Emit Turnkey Fuzzball Workflow + Measured Report
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
    validate_fuzzfile(turnkey_yaml)

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
        "turnkey_fuzzfile": turnkey_yaml,
    }
    summary_json_path = output_dir / "run_summary.json"
    summary_json_path.write_text(json.dumps(full_summary, indent=2, sort_keys=True), encoding="utf-8")

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
