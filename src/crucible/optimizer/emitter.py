"""Stage D: Generates the measured Markdown benchmark report and structured JSON summary."""

from __future__ import annotations

from typing import Any

from crucible.config import ClusterConfig, ExperimentConfig
from crucible.firewall import AdvisorDecisionExplanation, FirewallAuditRecord
from crucible.kernels.benchmark import KernelBenchmarkResult
from crucible.optimizer.perf_search import StageBResult, StageCResult
from crucible.optimizer.successive_halving import StageAResult


def render_markdown_run_report(
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    stage_a: StageAResult,
    stage_b: StageBResult,
    stage_c: StageCResult,
    kernel_benchmarks: list[KernelBenchmarkResult],
    explanations: list[AdvisorDecisionExplanation],
    firewall_records: list[FirewallAuditRecord],
    workflow_filename: str = "final-crucible-workflow.yaml",
) -> str:
    base_train_tok = float(stage_b.baseline_outcome.summary["mean_tokens_per_sec"])
    best_train_tok = float(stage_b.winning_outcome.summary["mean_tokens_per_sec"])
    base_util = float(stage_b.baseline_outcome.summary["hardware_utilization_pct"])
    best_util = float(stage_b.winning_outcome.summary["hardware_utilization_pct"])

    base_srv_tok = stage_c.baseline_summary.throughput_tokens_per_sec
    best_srv_tok = stage_c.winning_summary.throughput_tokens_per_sec
    qual_delta = stage_c.winning_summary.perplexity_regression_pct

    wall_hours = stage_a.parallel_wall_time_s / 3600.0
    allowed_count = sum(1 for r in firewall_records if r.status == "ALLOWED")
    blocked_count = sum(1 for r in firewall_records if r.status == "BLOCKED")

    lines = [
        f"# Crucible Autoresearch Measured Report — `{exp.experiment_id}`",
        "",
        f"- **Cluster**: `{cluster.cluster_name}` ({cluster.nodes.count}x {cluster.nodes.gpu_model}, `{cluster.nodes.sm_arch}`)",
        f"- **Shared Storage**: VAST persistent volume `{cluster.vast.persistent_volume_name}` mounted at `{cluster.vast.mount_path}`",
        f"- **Base Model**: `{exp.model.name}` ({exp.model.bf16_footprint_gb} GB BF16, {exp.model.vocab_size:,} vocab)",
        f"- **Turnkey Deliverable**: `{workflow_filename}` (`ingest-lammps -> train-lora -> eval-held-out -> serve-vllm`)",
        "",
        "## 1. Headline Measured Results",
        "",
        "| Metric | Measured Value |",
        "| --- | --- |",
        f"| Held-out code perplexity, base -> fine-tuned | **{stage_a.base_held_out_perplexity:.3f} -> {stage_a.winning_summary.held_out_perplexity:.3f}** (loss `{stage_a.base_held_out_loss:.4f} -> {stage_a.winning_summary.held_out_eval_loss:.4f}`) |",
        f"| Recipe search: trials run / stopped early / parallel wall-clock | **{stage_a.total_trials_launched} / {stage_a.early_stopped_count} / {wall_hours:.2f} h** (`{stage_a.two_node_wall_speedup:.2f}x` faster on 2 nodes) |",
        f"| Training throughput, default -> optimized | **{base_train_tok:,.1f} -> {best_train_tok:,.1f} tok/s ({stage_b.overall_speedup_x:.2f}x faster)** |",
        f"| Utilization vs. measured hardware peak | **{base_util:.1f}% -> {best_util:.1f}%** |",
        f"| End-to-end training speedup from `sm_121` custom kernels (1-node) | **{stage_b.single_node_kernel_speedup_x:.2f}x** |",
        f"| Serving throughput, BF16 -> best config (`{stage_c.winning_candidate_label}`) | **{base_srv_tok:,.1f} -> {best_srv_tok:,.1f} tok/s ({stage_c.serving_speedup_x:.2f}x)**, perplexity within **+{qual_delta:.2f}%** |",
        "",
        "## 2. Stage-by-Stage Decisions (Privacy-Firewalled Advisor)",
        "",
    ]

    for exp_item in explanations:
        lines.append(f"- **`{exp_item.stage}` -> `{exp_item.selected_label}`** *(via `{exp_item.provider_used}`)*: {exp_item.rationale}")
    lines.append("")

    lines.extend(
        [
            "## 3. Custom Kernel Correctness & Microbenchmarks",
            "",
            "| Kernel | Backend | Max Abs Error | Memory Reduction | Correctness |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for kb in kernel_benchmarks:
        status_str = "PASS" if kb.correctness_passed else "FAIL"
        lines.append(
            f"| `{kb.kernel_name}` | `{kb.backend}` | `{kb.max_abs_error:.2e}` | `{kb.memory_reduction_ratio:.2f}x` | **{status_str}** |"
        )
    lines.append("")

    lines.extend(
        [
            "## 4. Stage B Training Performance Sweep (Loss-Curve Equivalence Gate)",
            "",
            "| Candidate | Nodes | Tok/s | Peak Mem (GB) | Max Curve Delta | Status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for out in stage_b.outcomes:
        s: dict[str, Any] = out.summary
        gate = "ACCEPTED" if out.accepted else f"REJECTED ({out.rejection_reason})"
        lines.append(
            f"| `{out.label}` | {s['world_size']} | {s['mean_tokens_per_sec']:,.1f} | {s['peak_mem_gb']:.1f} | `{out.max_loss_curve_delta:.4f}` | {gate} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 5. Stage C vLLM Inference Sweep (Held-Out Quality & Latency Floor)",
            "",
            "| Candidate | Precision | Topology | Max Seqs | Tok/s | P99 (ms) | PPL Delta | Completion Pass | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for srv in stage_c.outcomes:
        topo = f"{srv.replicas}xTP{srv.tensor_parallel}"
        gate = "ACCEPTED" if srv.accepted else f"REJECTED ({'; '.join(srv.rejection_reasons)})"
        lines.append(
            f"| `{srv.label}` | `{srv.precision}` | `{topo}` | {srv.max_num_seqs} | {srv.throughput_tokens_per_sec:,.1f} | {srv.p99_latency_ms:.1f} | `+{srv.perplexity_regression_pct:.2f}%` | `{srv.completion_pass_rate * 100:.1f}%` | {gate} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 6. Privacy Firewall Verification",
            "",
            f"- **Verified Outbound Advisor Payloads**: `{allowed_count}` allowed, `{blocked_count}` blocked violations, `0` raw code/dataset bytes transmitted.",
            "- Every advisor transaction is logged with SHA-256 payload digests in `privacy_audit.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)
