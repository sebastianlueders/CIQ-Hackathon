"""Execution backends for CIQ Fuzzball workflows: MCP 4.2, CLI, local PyTorch, and GB10 simulator."""

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
from typing import Any, Callable, Protocol

from crucible.config import (
    ClusterConfig,
    ExperimentConfig,
    InferencePerfCandidate,
    RecipeCandidate,
    TrainingPerfCandidate,
)
from crucible.eval.evaluator import evaluate_quality_floor
from crucible.fuzzball.mcp_client import FuzzballMCPError, FuzzballMCPStdioClient
from crucible.fuzzball.renderer import (
    render_serving_trial_fuzzfile,
    render_training_trial_fuzzfile,
    trial_metrics_object_uri,
)
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.serving.vllm_bench import ServingTrialSummary
from crucible.trainer.types import StepMetricRecord, TrainingTrialSummary

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
EventCallback = Callable[[dict[str, Any]], None]


def _parse_summary_from_logs(log_text: str) -> dict[str, Any] | None:
    for line in reversed(log_text.splitlines()):
        if "CRUCIBLE_SUMMARY_JSON=" in line:
            _, rhs = line.split("CRUCIBLE_SUMMARY_JSON=", 1)
            return json.loads(rhs.strip())
    return None


def _extract_placed_nodes(raw_describe: Any, cluster: ClusterConfig, requested_nodes: int, seq_idx: int) -> list[str]:
    """Extract live Fuzzball node hostnames/IDs from `workflow_describe` output, falling back to
    round-robin placement over `cluster.nodes.inventory` (`CIQ-CHI-SPARK39`, `CIQ-CHI-SPARK40`)."""
    text_blob = json.dumps(raw_describe) if not isinstance(raw_describe, str) else raw_describe
    matched: list[str] = []
    for inv in cluster.nodes.inventory:
        if (
            inv.hostname in text_blob
            or inv.node_id in text_blob
            or inv.address in text_blob
        ) and inv.hostname not in matched:
            matched.append(inv.hostname)
    if matched:
        return matched

    inv_names = [n.hostname for n in cluster.nodes.inventory] or [
        f"gb10-node-{i + 1:02d}" for i in range(cluster.nodes.count)
    ]
    n_req = max(1, min(requested_nodes, len(inv_names)))
    if n_req >= len(inv_names):
        return list(inv_names)
    return [inv_names[seq_idx % len(inv_names)]]


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
        event_hook: EventCallback | None = None,
    ) -> None:
        if shutil.which(fuzzball_bin) is None:
            raise RuntimeError(
                f"'{fuzzball_bin}' CLI not found on PATH. Authenticate on the Team 20 workstation "
                "or select 'simulator' / 'local_torch' backend."
            )
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.poll_interval_s = poll_interval_s
        self.fuzzball_bin = fuzzball_bin
        self.event_hook = event_hook
        self._trial_counter = 0

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_hook is not None:
            self.event_hook(payload)

    def _submit_and_wait(
        self,
        fuzzfile_path: Path,
        workflow_name: str,
        trial_id: str,
        kind: str,
        requested_nodes: int,
        object_uri: str,
    ) -> tuple[str, list[str]]:
        seq_idx = self._trial_counter
        self._trial_counter += 1
        yaml_text = fuzzfile_path.read_text(encoding="utf-8")
        validate_fuzzfile(yaml_text, cluster=self.cluster)
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
        initial_nodes = _extract_placed_nodes("", self.cluster, requested_nodes, seq_idx)
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": kind,
                "workflow_id": wf_uuid,
                "workflow_name": workflow_name,
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": initial_nodes,
                "object_uri": object_uri,
                "fuzzfile": yaml_text,
            }
        )

        placed_nodes = initial_nodes
        while True:
            desc = subprocess.run(
                [self.fuzzball_bin, "workflow", "describe", wf_uuid],
                check=True,
                capture_output=True,
                text=True,
            )
            placed_nodes = _extract_placed_nodes(desc.stdout, self.cluster, requested_nodes, seq_idx)
            if "STAGE_STATUS_FINISHED" in desc.stdout:
                return wf_uuid, placed_nodes
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
        obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, trial_id)
        wf_uuid, nodes = self._submit_and_wait(
            fz_path, f"crucible-{trial_id}", trial_id, "train", perf.nodes, obj_uri
        )

        vast_summary = (
            Path(self.cluster.vast.runs_dir) / self.run_id / "trials" / trial_id / "summary.json"
        )
        if vast_summary.exists():
            raw = json.loads(vast_summary.read_text(encoding="utf-8"))
        else:
            log_proc = subprocess.run(
                [self.fuzzball_bin, "workflow", "log", wf_uuid, f"train-{trial_id}"],
                check=True,
                capture_output=True,
                text=True,
            )
            parsed = _parse_summary_from_logs(log_proc.stdout)
            if parsed is None:
                raise RuntimeError(
                    f"Trial {trial_id} finished ({wf_uuid}) but no summary found on VAST or job logs."
                )
            raw = parsed
        (trial_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_uuid, "nodes": nodes, "object_uri": obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_uuid,
                "nodes_assigned": nodes,
                "summary": raw,
            }
        )
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
        srv_obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, f"serving-{trial_id}")
        wf_uuid, nodes = self._submit_and_wait(
            fz_path,
            f"crucible-srv-{trial_id}",
            trial_id,
            "serve",
            cand.total_gpus_required,
            srv_obj_uri,
        )

        vast_summary = (
            Path(self.cluster.vast.runs_dir) / self.run_id / "serving" / trial_id / "summary.json"
        )
        if vast_summary.exists():
            raw = json.loads(vast_summary.read_text(encoding="utf-8"))
        else:
            log_proc = subprocess.run(
                [self.fuzzball_bin, "workflow", "log", wf_uuid, f"bench-{trial_id}"],
                check=True,
                capture_output=True,
                text=True,
            )
            parsed = _parse_summary_from_logs(log_proc.stdout)
            if parsed is None:
                raise RuntimeError(
                    f"Serving trial {trial_id} finished ({wf_uuid}) but no summary found in job logs."
                )
            raw = parsed
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "serve",
                "workflow_id": wf_uuid,
                "nodes_assigned": nodes,
                "summary": raw,
            }
        )
        return ServingTrialSummary(**raw)


class FuzzballMCPBackend:
    """Executes trial Fuzzfiles over CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve --allow-write --allow-exec`).

    Follows CIQ's Fuzzball 4.2 MCP agent contract:
    1. `workflow_validate` -> checks rendered Fuzzfile server-side
    2. `workflow_start`    -> launches trial workflow on `team20` and obtains `workflowId`
    3. `workflow_wait`     -> waits until terminal workflow state (`finished` / `STAGE_STATUS_FINISHED`)
    4. `object_get`        -> fetches compact numeric `metrics.json` egressed to the Fuzzball object cache
                              (never reading or egressing raw training data off VAST)
    """

    backend_name = "fuzzball_mcp"

    def __init__(
        self,
        cluster: ClusterConfig,
        exp: ExperimentConfig,
        run_id: str,
        local_run_dir: Path,
        mcp_command: list[str] | None = None,
        timeout_seconds: int = 7200,
        event_hook: EventCallback | None = None,
    ) -> None:
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.mcp_command = mcp_command
        self.timeout_seconds = timeout_seconds
        self.event_hook = event_hook
        self._trial_counter = 0

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_hook is not None:
            self.event_hook(payload)

    def _execute_and_fetch_metrics(
        self,
        render_fn: Callable[[ClusterConfig], str],
        fz_path: Path,
        workflow_name: str,
        trial_id: str,
        kind: str,
        requested_nodes: int,
        object_uri: str,
    ) -> tuple[dict[str, object], str, list[str]]:
        seq_idx = self._trial_counter
        self._trial_counter += 1
        yaml_text = render_fn(self.cluster)
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(yaml_text, cluster=self.cluster)

        with FuzzballMCPStdioClient(
            command=self.mcp_command,
            allow_write=True,
            allow_exec=True,
        ) as mcp:
            val_res = mcp.workflow_validate(yaml_text, fuzzfile_path=fz_path)
            if not val_res.get("valid", True):
                err_blob = json.dumps(val_res.get("errors") or []).lower()
                adapted = False
                if self.cluster.nodes.arch_device and "cpu/arm64" in err_blob:
                    self.cluster.nodes.arch_device = ""
                    adapted = True
                if self.cluster.pin_definition_annotation and (
                    "annotation" in err_blob or "provisioner" in err_blob
                ):
                    self.cluster.pin_definition_annotation = False
                    adapted = True
                if adapted:
                    yaml_text = render_fn(self.cluster)
                    fz_path.write_text(yaml_text, encoding="utf-8")
                    validate_fuzzfile(yaml_text, cluster=self.cluster)
                    val_res = mcp.workflow_validate(yaml_text, fuzzfile_path=fz_path)

            if not val_res.get("valid", True):
                raise FuzzballMCPError(
                    f"Fuzzball MCP workflow_validate rejected '{workflow_name}': {val_res.get('errors')}"
                )
            wf_id = mcp.workflow_start(yaml_text, name=workflow_name, fuzzfile_path=fz_path)
            desc_initial = (
                mcp.workflow_describe(wf_id) if "workflow_describe" in mcp.advertised_tools else {}
            )
            nodes = _extract_placed_nodes(desc_initial, self.cluster, requested_nodes, seq_idx)
            self._emit(
                {
                    "type": "trial_started",
                    "trial_id": trial_id,
                    "kind": kind,
                    "workflow_id": wf_id,
                    "workflow_name": workflow_name,
                    "definition_id": self.cluster.definition_id,
                    "nodes_assigned": nodes,
                    "object_uri": object_uri,
                    "fuzzfile": yaml_text,
                }
            )

            wait_res = mcp.workflow_wait(wf_id, timeout_seconds=self.timeout_seconds)
            status_str = str(wait_res.get("status", "")).upper()
            if status_str and ("FAIL" in status_str or "ERROR" in status_str or "CANCEL" in status_str):
                raise FuzzballMCPError(
                    f"Fuzzball MCP workflow {wf_id} ({workflow_name}) ended with status={status_str}"
                )
            if "workflow_describe" in mcp.advertised_tools:
                nodes = _extract_placed_nodes(
                    mcp.workflow_describe(wf_id), self.cluster, requested_nodes, seq_idx
                )
            job_name = f"train-{trial_id}" if kind == "train" else f"bench-{trial_id}"
            if not self.cluster.vast.egress_metrics_to_object_cache and "workflow_log" in mcp.advertised_tools:
                log_txt = mcp.workflow_log(wf_id, job_name)
                from_log = _parse_summary_from_logs(log_txt)
                if from_log is not None:
                    return from_log, wf_id, nodes

            try:
                obj_payload = mcp.object_get(object_uri, max_inline_bytes=65536)
                if isinstance(obj_payload, str):
                    parsed = json.loads(obj_payload)
                elif isinstance(obj_payload, dict) and "content" in obj_payload and isinstance(
                    obj_payload["content"], str
                ):
                    parsed = json.loads(obj_payload["content"])
                elif isinstance(obj_payload, dict):
                    parsed = obj_payload
                else:
                    raise FuzzballMCPError(f"Unexpected object_get payload: {obj_payload!r}")
            except Exception as obj_exc:
                if "workflow_log" in mcp.advertised_tools:
                    log_txt = mcp.workflow_log(wf_id, job_name)
                    from_log = _parse_summary_from_logs(log_txt)
                    if from_log is not None:
                        return from_log, wf_id, nodes
                raise FuzzballMCPError(
                    f"Could not fetch trial summary from workflow_log or object_get ({object_uri}): {obj_exc}"
                ) from obj_exc
            return parsed, wf_id, nodes

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
        fz_path = trial_dir / "workflow.yaml"
        obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, trial_id)
        raw, wf_id, nodes = self._execute_and_fetch_metrics(
            lambda c: render_training_trial_fuzzfile(
                c, self.exp, trial_id, recipe, perf, max_steps, self.run_id
            ),
            fz_path,
            f"crucible-{trial_id}",
            trial_id,
            "train",
            perf.nodes,
            obj_uri,
        )
        (trial_dir / "summary.json").write_text(
            json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8"
        )
        (trial_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "nodes_assigned": nodes,
                "summary": raw,
            }
        )
        return TrainingTrialSummary(**raw)

    def run_serving_trial(
        self,
        trial_id: str,
        cand: InferencePerfCandidate,
        baseline_finetuned_loss: float,
    ) -> ServingTrialSummary:
        srv_dir = self.local_run_dir / "serving" / trial_id
        srv_dir.mkdir(parents=True, exist_ok=True)
        fz_path = srv_dir / "workflow.yaml"
        adapter_vast = f"{self.cluster.vast.runs_dir}/{self.run_id}/best_adapter"
        srv_obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, f"serving-{trial_id}")
        raw, wf_id, nodes = self._execute_and_fetch_metrics(
            lambda c: render_serving_trial_fuzzfile(
                c,
                self.exp,
                trial_id,
                adapter_vast,
                cand,
                baseline_finetuned_loss,
                self.run_id,
            ),
            fz_path,
            f"crucible-srv-{trial_id}",
            trial_id,
            "serve",
            cand.total_gpus_required,
            srv_obj_uri,
        )
        (srv_dir / "summary.json").write_text(
            json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8"
        )
        (srv_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": srv_obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "serve",
                "workflow_id": wf_id,
                "nodes_assigned": nodes,
                "summary": raw,
            }
        )
        return ServingTrialSummary(**raw)


class CalibratedSimulatorBackend:
    """Deterministic Team 20 GB10 (`sm_121`) simulator that renders & validates real CIQ Fuzzfiles
    for every trial across `CIQ-CHI-SPARK39` and `CIQ-CHI-SPARK40` and models MoE LoRA convergence,
    memory footprint, kernel speedups, and quantization quality."""

    backend_name = "simulator"

    BASE_MODEL_HELD_OUT_LOSS = 1.9240

    def __init__(
        self,
        cluster: ClusterConfig,
        exp: ExperimentConfig,
        run_id: str,
        local_run_dir: Path,
        processed_data_dir: Path | None = None,
        event_hook: EventCallback | None = None,
        step_delay_s: float = 0.0,
    ) -> None:
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.processed_data_dir = processed_data_dir
        self.event_hook = event_hook
        self.step_delay_s = step_delay_s
        self._trial_counter = 0

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_hook is not None:
            self.event_hook(payload)

    @staticmethod
    def _jitter(key: str, scale: float = 0.003) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit = (int.from_bytes(digest[:4], "big") / float(1 << 32)) * 2.0 - 1.0
        return unit * scale

    def _sim_workflow_id(self, trial_id: str) -> str:
        h = hashlib.sha256(f"{self.run_id}:{trial_id}".encode("utf-8")).hexdigest()
        return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"

    def run_training_trial(
        self,
        trial_id: str,
        recipe: RecipeCandidate,
        perf: TrainingPerfCandidate,
        max_steps: int,
        reference_recipe: RecipeCandidate | None = None,
    ) -> TrainingTrialSummary:
        seq_idx = self._trial_counter
        self._trial_counter += 1
        trial_dir = self.local_run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        yaml_text = render_training_trial_fuzzfile(
            self.cluster, self.exp, trial_id, recipe, perf, max_steps, self.run_id
        )
        fz_path = trial_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(yaml_text, cluster=self.cluster)

        wf_id = self._sim_workflow_id(trial_id)
        nodes = _extract_placed_nodes("", self.cluster, perf.nodes, seq_idx)
        obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, trial_id)
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "workflow_name": f"crucible-{trial_id}",
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": nodes,
                "object_uri": obj_uri,
                "fuzzfile": yaml_text,
            }
        )
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)

        # Memory budget model on GB10 (112 GiB container-allocatable unified memory per node)
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
            raw_oom = oom_summary.to_dict()
            (trial_dir / "summary.json").write_text(
                json.dumps(raw_oom, indent=2, sort_keys=True), encoding="utf-8"
            )
            (trial_dir / "placement.json").write_text(
                json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": obj_uri}, indent=2),
                encoding="utf-8",
            )
            self._emit(
                {
                    "type": "trial_finished",
                    "trial_id": trial_id,
                    "kind": "train",
                    "workflow_id": wf_id,
                    "nodes_assigned": nodes,
                    "summary": raw_oom,
                }
            )
            return oom_summary

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

        global_batch_delta = abs(perf.global_batch_sequences - 16) * 0.0012
        kernel_eps = (0.0011 if perf.use_chunked_ce_kernel else 0.0) + (
            0.0008 if perf.use_grouped_moe_kernel else 0.0
        )

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
        raw_dict = summary.to_dict()
        (trial_dir / "summary.json").write_text(
            json.dumps(raw_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (trial_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "nodes_assigned": nodes,
                "summary": raw_dict,
            }
        )
        return summary

    def run_serving_trial(
        self,
        trial_id: str,
        cand: InferencePerfCandidate,
        baseline_finetuned_loss: float,
    ) -> ServingTrialSummary:
        seq_idx = self._trial_counter
        self._trial_counter += 1
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
        validate_fuzzfile(yaml_text, cluster=self.cluster)

        wf_id = self._sim_workflow_id(f"srv-{trial_id}")
        nodes = _extract_placed_nodes("", self.cluster, cand.total_gpus_required, seq_idx)
        srv_obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, f"serving-{trial_id}")
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": "serve",
                "workflow_id": wf_id,
                "workflow_name": f"crucible-srv-{trial_id}",
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": nodes,
                "object_uri": srv_obj_uri,
                "fuzzfile": yaml_text,
            }
        )
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)

        prec_speed = {"BF16": 1.0, "FP8": 1.78, "NVFP4": 2.68}[cand.precision]
        kernel_mult = 1.22 if cand.use_low_prec_decode_kernel else 1.0
        batch_efficiency = math.log2(max(cand.max_num_seqs, 2)) / math.log2(32.0)
        topo_mult = (cand.replicas * 0.98) * (1.0 if cand.tensor_parallel == 1 else 1.52)
        tok_s = round(410.0 * prec_speed * kernel_mult * batch_efficiency * topo_mult, 2)

        base_p50 = 92.0 / (prec_speed**0.55)
        tp_comm_overhead = 24.0 if cand.tensor_parallel > 1 else 0.0
        queueing_ms = (cand.max_num_seqs / 32.0) ** 1.48 * 38.0
        p50_ms = round(base_p50 + tp_comm_overhead + queueing_ms * 0.55, 2)
        p99_ms = round(p50_ms * 1.55 + queueing_ms * 0.92, 2)
        ttft_ms = round(26.0 + queueing_ms * 0.25, 2)

        loss_shift = {"BF16": 0.0, "FP8": 0.0055, "NVFP4": 0.0155}[cand.precision]
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
        raw_srv = summary.to_dict()
        (srv_dir / "summary.json").write_text(
            json.dumps(raw_srv, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (srv_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": srv_obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "serve",
                "workflow_id": wf_id,
                "nodes_assigned": nodes,
                "summary": raw_srv,
            }
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
        seq_idx = self._trial_counter
        self._trial_counter += 1
        trial_dir = self.local_run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = render_training_trial_fuzzfile(
            self.cluster, self.exp, trial_id, recipe, perf, max_steps, self.run_id
        )
        fz_path = trial_dir / "workflow.yaml"
        fz_path.write_text(yaml_text, encoding="utf-8")
        validate_fuzzfile(yaml_text, cluster=self.cluster)

        wf_id = self._sim_workflow_id(f"torch-{trial_id}")
        nodes = _extract_placed_nodes("", self.cluster, perf.nodes, seq_idx)
        obj_uri = trial_metrics_object_uri(self.cluster, self.run_id, trial_id)
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "workflow_name": f"crucible-{trial_id}",
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": nodes,
                "object_uri": obj_uri,
                "fuzzfile": yaml_text,
            }
        )
        from crucible.trainer.lora_trainer import run_micro_torch_training

        summary = run_micro_torch_training(
            trial_id=trial_id,
            processed_data_dir=self.processed_data_dir,
            output_dir=trial_dir,
            recipe=recipe,
            perf=perf,
            max_steps=min(max_steps, 16),
            measured_peak_tok_s=self.cluster.nodes.measured_peak_train_tok_s_per_node,
        )
        (trial_dir / "placement.json").write_text(
            json.dumps({"workflow_id": wf_id, "nodes": nodes, "object_uri": obj_uri}, indent=2),
            encoding="utf-8",
        )
        self._emit(
            {
                "type": "trial_finished",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "nodes_assigned": nodes,
                "summary": summary.to_dict(),
            }
        )
        return summary
