"""Execution backends for CIQ Fuzzball workflows: live CLI backend + calibrated GB10 simulator."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from crucible.config import (
    ClusterConfig,
    ExperimentConfig,
    InferencePerfCandidate,
    RecipeCandidate,
    TrainingPerfCandidate,
)
from crucible.eval.evaluator import evaluate_quality_floor
from crucible.fuzzball.renderer import (
    render_serving_trial_fuzzfile,
    render_training_trial_fuzzfile,
)
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.serving.vllm_bench import ServingTrialSummary
from crucible.trainer.lora_trainer import (
    StepMetricRecord,
    TrainingTrialSummary,
    run_micro_torch_training,
)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


class ExecutionBackend(Protocol):
    backend_name: str

    def run_training_trial(
        self,
        trial_id: str,
        recipe: RecipeCandidate,
        perf: TrainingPerfCandidate,
        max_steps: int,
        reference_recipe: RecipeCandidate | None = None,
    ) -> TrainingTrialSummary: ...

    def run_serving_trial(
        self,
        trial_id: str,
        cand: InferencePerfCandidate,
        baseline_finetuned_loss: float,
    ) -> ServingTrialSummary: ...


class FuzzballCLIBackend:
    """Submits rendered Fuzzfiles to a live CIQ Fuzzball cluster via the `fuzzball` CLI."""

    backend_name = "fuzzball"

    def __init__(
        self,
        cluster: ClusterConfig,
        exp: ExperimentConfig,
        run_id: str,
        local_run_dir: Path,
        poll_interval_s: float = 5.0,
        fuzzball_bin: str = "fuzzball",
    ) -> None:
        if shutil.which(fuzzball_bin) is None:
            raise RuntimeError(
                f"'{fuzzball_bin}' CLI not found on PATH. Authenticate on the CIQ Fuzzball workstation "
                "or pass '--backend simulator' / '--backend local_torch' for local runs."
            )
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.poll_interval_s = poll_interval_s
        self.fuzzball_bin = fuzzball_bin

    def _submit_and_wait(self, fuzzfile_path: Path, workflow_name: str) -> str:
        validate_fuzzfile(fuzzfile_path)
        subprocess.run(
            [self.fuzzball_bin, "workflow", "validate", str(fuzzfile_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        started = subprocess.run(
            [self.fuzzball_bin, "workflow", "start", str(fuzzfile_path), "-n", workflow_name],
            check=True,
            capture_output=True,
            text=True,
        )
        match = UUID_RE.search(started.stdout)
        if not match:
            raise RuntimeError(f"Could not parse workflow UUID from fuzzball output: {started.stdout!r}")
        wf_uuid = match.group(0)

        while True:
            desc = subprocess.run(
                [self.fuzzball_bin, "workflow", "describe", wf_uuid],
                check=True,
                capture_output=True,
                text=True,
            )
            if "STAGE_STATUS_FINISHED" in desc.stdout:
                return wf_uuid
            if "STAGE_STATUS_FAILED" in desc.stdout or "STAGE_STATUS_CANCELED" in desc.stdout:
                raise RuntimeError(f"Fuzzball workflow {wf_uuid} ({workflow_name}) failed:\n{desc.stdout}")
            time.sleep(self.poll_interval_s)

    def run_training_trial(
        self,
        trial_id: str,
        recipe: RecipeCandidate,
        perf: TrainingPerfCandidate,
        max_steps: int,
        reference_recipe: RecipeCandidate | None = None,
    ) -> TrainingTrialSummary:
        trial_dir = self.local_run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = render_training_trial_fuzzfile(
            self.cluster, self.exp, trial_id, recipe, perf, max_steps, self.run_id
        )
        fz_path = trial_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        self._submit_and_wait(fz_path, f"crucible-{trial_id}")

        vast_summary = (
            Path(self.cluster.vast.runs_dir) / self.run_id / "trials" / trial_id / "summary.json"
        )
        raw = json.loads(vast_summary.read_text(encoding="utf-8"))
        return TrainingTrialSummary(**raw)

    def run_serving_trial(
        self,
        trial_id: str,
        cand: InferencePerfCandidate,
        baseline_finetuned_loss: float,
    ) -> ServingTrialSummary:
        srv_dir = self.local_run_dir / "serving" / trial_id
        srv_dir.mkdir(parents=True, exist_ok=True)
        adapter_vast = f"{self.cluster.vast.runs_dir}/{self.run_id}/best_adapter"
        yaml_text = render_serving_trial_fuzzfile(
            self.cluster,
            self.exp,
            trial_id,
            adapter_vast,
            cand,
            baseline_finetuned_loss,
            self.run_id,
        )
        fz_path = srv_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        self._submit_and_wait(fz_path, f"crucible-srv-{trial_id}")

        vast_summary = (
            Path(self.cluster.vast.runs_dir) / self.run_id / "serving" / trial_id / "summary.json"
        )
        raw = json.loads(vast_summary.read_text(encoding="utf-8"))
        return ServingTrialSummary(**raw)


class CalibratedSimulatorBackend:
    """Deterministic GB10 (sm_121) simulator that renders & validates real CIQ Fuzzfiles for every
    trial and models MoE LoRA convergence, memory footprint, kernel speedups, and quantization quality."""

    backend_name = "simulator"

    BASE_MODEL_HELD_OUT_LOSS = 1.9240

    def __init__(
        self,
        cluster: ClusterConfig,
        exp: ExperimentConfig,
        run_id: str,
        local_run_dir: Path,
        processed_data_dir: Path | None = None,
    ) -> None:
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.processed_data_dir = processed_data_dir

    @staticmethod
    def _jitter(key: str, scale: float = 0.003) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit = (int.from_bytes(digest[:4], "big") / float(1 << 32)) * 2.0 - 1.0
        return unit * scale

    def run_training_trial(
        self,
        trial_id: str,
        recipe: RecipeCandidate,
        perf: TrainingPerfCandidate,
        max_steps: int,
        reference_recipe: RecipeCandidate | None = None,
    ) -> TrainingTrialSummary:
        trial_dir = self.local_run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        yaml_text = render_training_trial_fuzzfile(
            self.cluster, self.exp, trial_id, recipe, perf, max_steps, self.run_id
        )
        fz_path = trial_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(fz_path)

        # Memory budget model on GB10 (112 GB container-allocatable unified memory per node)
        weights_per_node = self.exp.model.bf16_footprint_gb / (
            perf.nodes if self.exp.model.min_nodes_required > 1 else 1.0
        )
        logit_buffer_gb = (
            0.32 * perf.micro_batch_size
            if perf.use_chunked_ce_kernel
            else 1.85 * perf.micro_batch_size * (recipe.seq_len / 2048.0)
        )
        act_gb = (2.4 if perf.activation_checkpointing else 5.2) * perf.micro_batch_size * (
            recipe.seq_len / 2048.0
        )
        adapter_opt_gb = 0.04 * recipe.lora_rank * (len(recipe.target_modules) / 4.0)
        peak_mem_gb = round(weights_per_node + logit_buffer_gb + act_gb + adapter_opt_gb, 2)

        if peak_mem_gb > self.cluster.nodes.memory_gb_per_node:
            oom_summary = TrainingTrialSummary(
                trial_id=trial_id,
                status="oom",
                steps_completed=0,
                initial_val_loss=self.BASE_MODEL_HELD_OUT_LOSS,
                final_train_loss=99.0,
                final_val_loss=99.0,
                held_out_eval_loss=99.0,
                held_out_perplexity=9999.0,
                mean_tokens_per_sec=0.0,
                hardware_utilization_pct=0.0,
                peak_mem_gb=peak_mem_gb,
                wall_time_s=1.0,
                loss_curve=[],
                world_size=perf.nodes,
                kernels_enabled={
                    "chunked_cross_entropy": perf.use_chunked_ce_kernel,
                    "grouped_moe_gemm": perf.use_grouped_moe_kernel,
                    "fused_lora_linear": perf.use_fused_lora_kernel,
                },
            )
            (trial_dir / "summary.json").write_text(
                json.dumps(oom_summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            return oom_summary

        # Recipe quality dynamics anchored to Qwen3-Coder-30B-A3B on LAMMPS C++
        optimal_log_lr = math.log10(1.6e-4)
        lr_dist = abs(math.log10(recipe.lr) - optimal_log_lr)
        lr_penalty = 0.22 * (lr_dist**1.6)
        module_coverage = len(recipe.target_modules) / 7.0
        rank_gain = 0.14 * min(1.0, math.log2(max(recipe.lora_rank, 2)) / 5.0)
        seq_bonus = 0.04 if recipe.seq_len >= 2048 else -0.05
        asymptote = max(
            1.18,
            self.BASE_MODEL_HELD_OUT_LOSS - 0.38 * module_coverage - rank_gain - seq_bonus + lr_penalty,
        )

        # Kernel equivalence property: custom sm_121 kernels preserve exact mathematical trajectory
        # up to BF16 accumulation jitter (< 0.003); changing global batch way off alters curve slightly.
        global_batch_delta = abs(perf.global_batch_sequences - 16) * 0.0012
        kernel_eps = (0.0011 if perf.use_chunked_ce_kernel else 0.0) + (
            0.0008 if perf.use_grouped_moe_kernel else 0.0
        )

        # Throughput model on GB10 sm_121 (baseline ~1,243 tok/s -> ~3,822 tok/s 1-node, ~6,956 tok/s 2-node)
        base_tok_s = 1120.0
        batch_mult = 1.0 + 0.11 * min(perf.micro_batch_size - 1, 5)
        ckpt_mult = 1.0 if perf.activation_checkpointing else 1.21
        ce_mult = 1.24 if perf.use_chunked_ce_kernel else 1.0
        moe_mult = 1.31 if perf.use_grouped_moe_kernel else 1.0
        lora_mult = 1.12 if perf.use_fused_lora_kernel else 1.0
        node_mult = 1.0 if perf.nodes == 1 else (perf.nodes * 0.91)
        tok_s = round(base_tok_s * batch_mult * ckpt_mult * ce_mult * moe_mult * lora_mult * node_mult, 2)

        peak_cluster_tok_s = self.cluster.nodes.measured_peak_train_tok_s_per_node * perf.nodes
        hw_util_pct = round(min(96.5, (tok_s / peak_cluster_tok_s) * 100.0), 2)

        rate = 0.011 * min(2.2, (recipe.lr / 1.5e-4) ** 0.45)
        loss_curve: list[float] = []
        metrics_path = trial_dir / "metrics.jsonl"
        tokens_per_step = perf.global_batch_sequences * recipe.seq_len
        cum_wall = 0.0

        with metrics_path.open("w", encoding="utf-8") as fh:
            for step in range(1, max_steps + 1):
                decay = math.exp(-rate * step)
                osc = 0.008 * math.sin(step * 0.37) * math.exp(-0.004 * step)
                instability = (
                    0.09 * math.sin(step * 0.9) if recipe.lr >= 7e-4 and step > 25 else 0.0
                )
                step_loss = (
                    asymptote
                    + (self.BASE_MODEL_HELD_OUT_LOSS - asymptote) * decay
                    + osc
                    + instability
                    + global_batch_delta
                    + kernel_eps
                )
                step_loss = round(step_loss, 5)
                loss_curve.append(step_loss)
                cum_wall += tokens_per_step / max(tok_s, 1.0)
                if step == 1 or step == max_steps or step % 15 == 0:
                    rec = StepMetricRecord(
                        step=step,
                        train_loss=step_loss,
                        val_loss=round(step_loss + 0.018, 5),
                        tokens_per_sec=tok_s,
                        peak_mem_gb=peak_mem_gb,
                        wall_time_s=round(cum_wall, 2),
                        world_size=perf.nodes,
                    )
                    fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")

        final_val = round(loss_curve[-1] + 0.018 + self._jitter(recipe.short_id(), 0.002), 5)
        held_out = round(final_val + 0.011, 5)
        summary = TrainingTrialSummary(
            trial_id=trial_id,
            status="completed",
            steps_completed=max_steps,
            initial_val_loss=self.BASE_MODEL_HELD_OUT_LOSS,
            final_train_loss=loss_curve[-1],
            final_val_loss=final_val,
            held_out_eval_loss=held_out,
            held_out_perplexity=round(math.exp(held_out), 4),
            mean_tokens_per_sec=tok_s,
            hardware_utilization_pct=hw_util_pct,
            peak_mem_gb=peak_mem_gb,
            wall_time_s=round(cum_wall, 2),
            loss_curve=loss_curve,
            world_size=perf.nodes,
            kernels_enabled={
                "chunked_cross_entropy": perf.use_chunked_ce_kernel,
                "grouped_moe_gemm": perf.use_grouped_moe_kernel,
                "fused_lora_linear": perf.use_fused_lora_kernel,
            },
        )
        (trial_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def run_serving_trial(
        self,
        trial_id: str,
        cand: InferencePerfCandidate,
        baseline_finetuned_loss: float,
    ) -> ServingTrialSummary:
        srv_dir = self.local_run_dir / "serving" / trial_id
        srv_dir.mkdir(parents=True, exist_ok=True)

        yaml_text = render_serving_trial_fuzzfile(
            self.cluster,
            self.exp,
            trial_id,
            f"{self.cluster.vast.runs_dir}/{self.run_id}/best_adapter",
            cand,
            baseline_finetuned_loss,
            self.run_id,
        )
        fz_path = srv_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(fz_path)

        # Serving throughput & latency physics on 2x GB10 (273 GB/s unified LPDDR5X per node)
        prec_speed = {"BF16": 1.0, "FP8": 1.78, "NVFP4": 2.68}[cand.precision]
        kernel_mult = 1.22 if cand.use_low_prec_decode_kernel else 1.0
        batch_efficiency = math.log2(max(cand.max_num_seqs, 2)) / math.log2(32.0)
        # On 200GbE RoCE, 2 independent replicas scale at ~1.96x with zero per-token all-reduce;
        # TP=2 across nodes pays per-layer all-reduce latency overhead (~1.52x net).
        topo_mult = (cand.replicas * 0.98) * (1.0 if cand.tensor_parallel == 1 else 1.52)
        tok_s = round(410.0 * prec_speed * kernel_mult * batch_efficiency * topo_mult, 2)

        base_p50 = 92.0 / (prec_speed**0.55)
        tp_comm_overhead = 24.0 if cand.tensor_parallel > 1 else 0.0
        queueing_ms = (cand.max_num_seqs / 32.0) ** 1.48 * 38.0
        p50_ms = round(base_p50 + tp_comm_overhead + queueing_ms * 0.55, 2)
        p99_ms = round(p50_ms * 1.55 + queueing_ms * 0.92, 2)
        ttft_ms = round(26.0 + queueing_ms * 0.25, 2)

        # Precision impact on held-out LAMMPS C++ NLL and completion accuracy
        loss_shift = {"BF16": 0.0, "FP8": 0.0055, "NVFP4": 0.0155}[cand.precision]
        # Extreme dynamic batching with FP4 KV quantization pressure degrades long-context C++ completion
        kv_pressure_shift = 0.022 if cand.max_num_seqs >= 200 and cand.precision == "NVFP4" else 0.0
        srv_loss = baseline_finetuned_loss + loss_shift + kv_pressure_shift

        completion_pass = {
            "BF16": 0.9167,
            "FP8": 0.9083,
            "NVFP4": 0.8750 if cand.max_num_seqs < 200 else 0.8083,
        }[cand.precision]

        q_report = evaluate_quality_floor(
            held_out_loss=srv_loss,
            baseline_finetuned_loss=baseline_finetuned_loss,
            completion_pass_rate=completion_pass,
            mean_identifier_recall=round(completion_pass * 0.95, 4),
            max_perplexity_regression_pct=self.exp.stage_c_inference_perf.max_perplexity_regression_pct,
            min_completion_pass_rate=self.exp.stage_c_inference_perf.min_completion_pass_rate,
        )
        passed_lat = p99_ms <= self.exp.stage_c_inference_perf.p99_latency_ceiling_ms
        reasons = list(q_report.rejection_reasons)
        if not passed_lat:
            reasons.append(
                f"p99_latency_ms={p99_ms:.1f} exceeds ceiling "
                f"{self.exp.stage_c_inference_perf.p99_latency_ceiling_ms:.1f}"
            )

        summary = ServingTrialSummary(
            trial_id=trial_id,
            label=cand.label,
            precision=cand.precision,
            tensor_parallel=cand.tensor_parallel,
            replicas=cand.replicas,
            max_num_seqs=cand.max_num_seqs,
            use_low_prec_decode_kernel=cand.use_low_prec_decode_kernel,
            throughput_tokens_per_sec=tok_s,
            ttft_ms=ttft_ms,
            p50_latency_ms=p50_ms,
            p99_latency_ms=p99_ms,
            held_out_loss=q_report.held_out_loss,
            held_out_perplexity=q_report.held_out_perplexity,
            perplexity_regression_pct=q_report.perplexity_regression_pct,
            completion_pass_rate=q_report.completion_pass_rate,
            passed_quality_floor=q_report.passed_quality_floor,
            passed_latency_ceiling=passed_lat,
            accepted=(q_report.passed_quality_floor and passed_lat),
            rejection_reasons=reasons,
        )
        (srv_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary


class LocalMicroTorchBackend(CalibratedSimulatorBackend):
    """Executes real in-process PyTorch MoE + LoRA training (`run_micro_torch_training`) on CPU/GPU."""

    backend_name = "local_torch"

    def run_training_trial(
        self,
        trial_id: str,
        recipe: RecipeCandidate,
        perf: TrainingPerfCandidate,
        max_steps: int,
        reference_recipe: RecipeCandidate | None = None,
    ) -> TrainingTrialSummary:
        if self.processed_data_dir is None:
            raise ValueError("LocalMicroTorchBackend requires processed_data_dir.")
        trial_dir = self.local_run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = render_training_trial_fuzzfile(
            self.cluster, self.exp, trial_id, recipe, perf, max_steps, self.run_id
        )
        fz_path = trial_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(fz_path)
        return run_micro_torch_training(
            trial_id=trial_id,
            processed_data_dir=self.processed_data_dir,
            output_dir=trial_dir,
            recipe=recipe,
            perf=perf,
            max_steps=min(max_steps, 16),
            measured_peak_tok_s=self.cluster.nodes.measured_peak_train_tok_s_per_node,
        )
