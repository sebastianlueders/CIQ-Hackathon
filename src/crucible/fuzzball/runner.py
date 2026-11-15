"""Execution backends for CIQ Fuzzball workflows: MCP 4.2, CLI, local PyTorch, and GB10 simulator."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from crucible.fuzzball.discovery import (
    _extract_volume_provisioner,
    ensure_provisioner_volume,
    extract_team_from_text,
    resolve_fuzzball_executable,
)
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


def _parse_steps_from_logs(log_text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    last_eval_loss: float | None = None
    for line in log_text.splitlines():
        if "CRUCIBLE_STEP_JSON=" in line:
            try:
                _, rhs = line.split("CRUCIBLE_STEP_JSON=", 1)
                steps.append(json.loads(rhs.strip()))
            except Exception:
                pass
        elif "RESULT {" in line:
            try:
                _, rhs = line.split("RESULT ", 1)
                rec = json.loads(rhs.strip())
                kind = rec.get("kind")
                if kind == "eval" and rec.get("eval_loss") is not None:
                    last_eval_loss = float(rec["eval_loss"])
                elif kind == "train":
                    ar_s = float(rec.get("allreduce_s") or 0.0)
                    steps.append(
                        {
                            "step": int(rec.get("step", len(steps) + 1)),
                            "train_loss": float(rec.get("loss", 0.0)),
                            "val_loss": last_eval_loss,
                            "tokens_per_sec": float(rec.get("tok_s", 0.0)),
                            "tokens_seen": int(rec.get("tokens", 0)),
                            "lr": float(rec.get("lr", 0.0)),
                            "grad_norm": float(rec.get("grad_norm", 0.0)),
                            "step_time_ms": round(float(rec.get("step_s", 0.0)) * 1000.0, 1),
                            "allreduce_ms": round(ar_s * 1000.0, 2),
                            "peak_mem_gb": float(rec.get("peak_mem_gib", 0.0)),
                            "world_size": int(rec.get("world_size", 1)),
                            "router_entropy": rec.get("router_entropy_mean"),
                        }
                    )
            except Exception:
                pass
    return steps


def inspect_workflow_stages(desc: Any) -> dict[str, Any]:
    """Inspect Fuzzball `workflow_describe` JSON/dict (`status.stages[]`) to distinguish
    `QUEUED` (waiting for GPU) vs `RUNNING` (on GPU) vs terminal `FINISHED`/`ERRORED`/`CANCELED`."""
    if isinstance(desc, str):
        stripped = desc.strip()
        if stripped.startswith("{"):
            try:
                desc = json.loads(stripped)
            except Exception:
                pass

    if isinstance(desc, dict):
        status_block = desc.get("status")
        stages = status_block.get("stages") if isinstance(status_block, dict) else None
        if isinstance(stages, list) and stages:
            wf_stage_status = ""
            job_stage_status = ""
            job_name = ""
            errors: list[str] = []
            for st in stages:
                if not isinstance(st, dict):
                    continue
                kind = str(st.get("kind", "")).upper()
                st_val = str(st.get("status", "")).upper()
                if st.get("error"):
                    errors.append(f"{st.get('name') or kind}: {st['error']}")
                if "WORKFLOW" in kind:
                    wf_stage_status = st_val
                elif "JOB" in kind or "SERVICE" in kind:
                    job_stage_status = st_val
                    job_name = str(st.get("name") or job_name)

            err_joined = "; ".join(errors)
            if "FAILED" in wf_stage_status or "ERROR" in wf_stage_status or "FAILED" in job_stage_status:
                return {"has_status": True, "terminal": True, "phase": "ERRORED", "job_name": job_name, "error": err_joined}
            if "CANCEL" in wf_stage_status or "CANCEL" in job_stage_status:
                return {"has_status": True, "terminal": True, "phase": "CANCELED", "job_name": job_name, "error": err_joined}
            if "FINISHED" in wf_stage_status or ("FINISHED" in job_stage_status and job_stage_status):
                return {"has_status": True, "terminal": True, "phase": "FINISHED", "job_name": job_name, "error": ""}
            if "STARTED" in job_stage_status or "RUNNING" in job_stage_status:
                return {"has_status": True, "terminal": False, "phase": "RUNNING", "job_name": job_name, "error": ""}
            # Workflow started (volume/image done), but STAGE_KIND_JOB is still waiting for a free GPU!
            return {"has_status": True, "terminal": False, "phase": "QUEUED", "job_name": job_name, "error": ""}

        # Flat status dict fallback (`{"status": "finished"}` or `{"state": "..."}`)
        for k in ("status", "state", "workflowStatus"):
            val = desc.get(k)
            if isinstance(val, str) and val.strip():
                up = val.strip().upper()
                if any(t in up for t in ("FINISH", "SUCCEED", "COMPLETE")):
                    return {"has_status": True, "terminal": True, "phase": "FINISHED", "job_name": "", "error": ""}
                if any(t in up for t in ("FAIL", "ERROR")):
                    return {"has_status": True, "terminal": True, "phase": "ERRORED", "job_name": "", "error": str(desc.get("error") or "")}
                if "CANCEL" in up:
                    return {"has_status": True, "terminal": True, "phase": "CANCELED", "job_name": "", "error": ""}
                if any(t in up for t in ("PENDING", "QUEUE", "UNSPECIFIED", "WAIT", "CREATING")):
                    return {"has_status": True, "terminal": False, "phase": "QUEUED", "job_name": "", "error": ""}
                if any(t in up for t in ("START", "RUN")):
                    return {"has_status": True, "terminal": False, "phase": "RUNNING", "job_name": "", "error": ""}
        return {"has_status": False, "terminal": False, "phase": "UNKNOWN", "job_name": "", "error": ""}

    raw_str = str(desc or "")
    if "STAGE_STATUS_FINISHED" in raw_str:
        return {"has_status": True, "terminal": True, "phase": "FINISHED", "job_name": "", "error": ""}
    if "STAGE_STATUS_FAILED" in raw_str or "Errored" in raw_str:
        return {"has_status": True, "terminal": True, "phase": "ERRORED", "job_name": "", "error": raw_str[-300:]}
    if "STAGE_STATUS_CANCELED" in raw_str:
        return {"has_status": True, "terminal": True, "phase": "CANCELED", "job_name": "", "error": ""}
    if "Job" in raw_str and ("Pending" in raw_str or "UNSPECIFIED" in raw_str):
        return {"has_status": True, "terminal": False, "phase": "QUEUED", "job_name": "", "error": ""}
    if "STAGE_STATUS_STARTED" in raw_str:
        return {"has_status": True, "terminal": False, "phase": "RUNNING", "job_name": "", "error": ""}
    return {"has_status": False, "terminal": False, "phase": "UNKNOWN", "job_name": "", "error": ""}


def _compact_splits_json(processed_data_dir: Path | None, max_seqs_per_split: int = 6, max_toks: int = 128) -> str | None:
    if processed_data_dir is None or not processed_data_dir.exists():
        return None
    payload: dict[str, list[list[int]]] = {}
    for split_name in ("train", "val", "held_out"):
        fpath = processed_data_dir / f"{split_name}.jsonl"
        if not fpath.exists():
            continue
        seqs: list[list[int]] = []
        for ln in fpath.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            ids = json.loads(ln).get("input_ids") or []
            if len(ids) >= 8:
                seqs.append([int(x) for x in ids[:max_toks]])
            if len(seqs) >= max_seqs_per_split:
                break
        if seqs:
            payload[split_name] = seqs
    return json.dumps(payload, separators=(",", ":")) if payload else None


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
        resolved_bin = resolve_fuzzball_executable(fuzzball_bin)
        if resolved_bin is None:
            raise RuntimeError(
                f"'{fuzzball_bin}' CLI not found on PATH or ./fuzzball. Authenticate on the workstation "
                "or select 'simulator' / 'local_torch' backend."
            )
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.poll_interval_s = poll_interval_s
        self.fuzzball_bin = resolved_bin
        self.event_hook = event_hook
        self._trial_counter = 0
        if not self.cluster.vast.ephemeral and self.cluster.vast.provisioner:
            self.cluster.vast.persistent_volume_name = ensure_provisioner_volume(
                self.cluster.vast.provisioner,
                preferred_name=self.cluster.vast.persistent_volume_name,
                fuzzball_bin=self.fuzzball_bin,
            )

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
        job_name = f"train-{trial_id}" if kind == "train" else f"bench-{trial_id}"
        queue_timeout_s = int(os.getenv("CRUCIBLE_QUEUE_TIMEOUT", "14400"))
        t_wait0 = time.monotonic()
        last_phase = ""
        seen_steps = 0

        while True:
            elapsed_s = int(time.monotonic() - t_wait0)
            if elapsed_s > queue_timeout_s:
                raise RuntimeError(
                    f"Fuzzball workflow {wf_uuid} ({workflow_name}) exceeded queue/execution timeout ({queue_timeout_s}s)."
                )
            desc = subprocess.run(
                [self.fuzzball_bin, "workflow", "describe", wf_uuid, "-o", "json"],
                check=False,
                capture_output=True,
                text=True,
            )
            desc_out = desc.stdout if desc.returncode == 0 and desc.stdout.strip() else subprocess.run(
                [self.fuzzball_bin, "workflow", "describe", wf_uuid],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            placed_nodes = _extract_placed_nodes(desc_out, self.cluster, requested_nodes, seq_idx)
            st_info = inspect_workflow_stages(desc_out)

            if st_info["terminal"]:
                if st_info["phase"] == "FINISHED":
                    return wf_uuid, placed_nodes
                log_tail = ""
                log_p = subprocess.run(
                    [self.fuzzball_bin, "workflow", "log", wf_uuid, st_info["job_name"] or job_name, "-n", "25"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if log_p.stdout.strip():
                    log_tail = "\nLast job logs:\n" + "\n".join(log_p.stdout.strip().splitlines()[-25:])
                raise RuntimeError(
                    f"Fuzzball workflow {wf_uuid} ({workflow_name}) {st_info['phase']}: "
                    f"{st_info['error']}{log_tail}"
                )

            if st_info["phase"] == "QUEUED":
                mins, secs = divmod(elapsed_s, 60)
                self._emit(
                    {
                        "type": "trial_queued",
                        "trial_id": trial_id,
                        "workflow_id": wf_uuid,
                        "elapsed_s": elapsed_s,
                        "message": f"QUEUED — Waiting for free {self.cluster.definition_id} GPU ({mins}m {secs:02d}s elapsed)",
                    }
                )
            elif st_info["phase"] == "RUNNING":
                if last_phase != "RUNNING":
                    self._emit(
                        {
                            "type": "trial_running",
                            "trial_id": trial_id,
                            "workflow_id": wf_uuid,
                            "nodes_assigned": placed_nodes,
                            "message": f"RUNNING ON GPU — {trial_id} executing on {', '.join(placed_nodes)}",
                        }
                    )
                log_p = subprocess.run(
                    [self.fuzzball_bin, "workflow", "log", wf_uuid, st_info["job_name"] or job_name, "-n", "30"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if log_p.returncode == 0 and log_p.stdout:
                    steps = _parse_steps_from_logs(log_p.stdout)
                    for step_rec in steps[seen_steps:]:
                        self._emit({"type": "trial_step", "trial_id": trial_id, **step_rec})
                    seen_steps = len(steps)
            last_phase = st_info["phase"]
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
        timeout_seconds: int | None = None,
        poll_interval_s: float = 2.5,
        processed_data_dir: Path | None = None,
        event_hook: EventCallback | None = None,
    ) -> None:
        self.cluster = cluster
        self.exp = exp
        self.run_id = run_id
        self.local_run_dir = local_run_dir
        self.mcp_command = mcp_command
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else int(os.getenv("CRUCIBLE_QUEUE_TIMEOUT", "14400"))
        )
        self.poll_interval_s = poll_interval_s
        self.processed_data_dir = processed_data_dir
        self.event_hook = event_hook
        self._trial_counter = 0
        self._counter_lock = os.getpid()

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
            # Ensure target persistent volume actually exists on `self.cluster.vast.provisioner`
            if (
                self._trial_counter == 1
                and self.mcp_command is None
                and not self.cluster.vast.ephemeral
                and self.cluster.vast.provisioner
            ):
                ensured = ensure_provisioner_volume(
                    self.cluster.vast.provisioner,
                    preferred_name=self.cluster.vast.persistent_volume_name,
                    mcp=mcp,
                )
                if ensured != self.cluster.vast.persistent_volume_name:
                    self.cluster.vast.persistent_volume_name = ensured
                    yaml_text = render_fn(self.cluster)
                    fz_path.write_text(yaml_text, encoding="utf-8")

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
                err_txt = json.dumps(val_res.get("errors") or [])
                inferred_team = extract_team_from_text(err_txt)
                if inferred_team and inferred_team != self.cluster.vast.provisioner:
                    self.cluster.definition_id = inferred_team
                    self.cluster.vast.provisioner = inferred_team
                    yaml_text = render_fn(self.cluster)
                    fz_path.write_text(yaml_text, encoding="utf-8")
                    val_res = mcp.workflow_validate(yaml_text, fuzzfile_path=fz_path)
            if not val_res.get("valid", True):
                raise FuzzballMCPError(
                    f"Fuzzball MCP workflow_validate rejected '{workflow_name}': {val_res.get('errors')}"
                )
            try:
                wf_id = mcp.workflow_start(yaml_text, name=workflow_name, fuzzfile_path=fz_path)
            except FuzzballMCPError as start_exc:
                msg_low = str(start_exc).lower()
                retried = False
                inferred_team = extract_team_from_text(str(start_exc))
                if "cannot access provisioner" in msg_low and inferred_team:
                    self.cluster.definition_id = inferred_team
                    self.cluster.vast.provisioner = inferred_team
                    self.cluster.vast.persistent_volume_name = ensure_provisioner_volume(
                        inferred_team,
                        preferred_name="runs",
                        mcp=mcp,
                    )
                    retried = True
                elif "volume" in msg_low and ("not exist" in msg_low or "not found" in msg_low):
                    prov = self.cluster.vast.provisioner or self.cluster.definition_id or "team3"
                    failed_vol = self.cluster.vast.persistent_volume_name
                    matched_on_prov: list[str] = []
                    if "volume_list" in mcp.advertised_tools:
                        try:
                            vraw = mcp.volume_list(provisioner=prov)
                            vlist = vraw.get("volumes", []) if isinstance(vraw, dict) else []
                            if isinstance(vlist, list):
                                for item in vlist:
                                    if isinstance(item, dict) and item.get("name"):
                                        iprov = _extract_volume_provisioner(item)
                                        if iprov and iprov.lower() == prov.lower():
                                            matched_on_prov.append(str(item["name"]))
                        except Exception:
                            pass
                    if matched_on_prov:
                        pick = next((m for m in matched_on_prov if m != failed_vol), matched_on_prov[0])
                        self.cluster.vast.persistent_volume_name = pick
                        retried = True
                    else:
                        target_new_vol = "crucible-vast" if failed_vol == "runs" else "runs"
                        created = ensure_provisioner_volume(prov, preferred_name=target_new_vol, mcp=mcp)
                        self.cluster.vast.persistent_volume_name = created
                        retried = True
                if not retried:
                    raise
                yaml_text = render_fn(self.cluster)
                fz_path.write_text(yaml_text, encoding="utf-8")
                validate_fuzzfile(yaml_text, cluster=self.cluster)
                try:
                    wf_id = mcp.workflow_start(yaml_text, name=workflow_name, fuzzfile_path=fz_path)
                except FuzzballMCPError as second_exc:
                    if "volume" in str(second_exc).lower() and not self.cluster.vast.ephemeral:
                        self.cluster.vast.ephemeral = True
                        yaml_text = render_fn(self.cluster)
                        fz_path.write_text(yaml_text, encoding="utf-8")
                        validate_fuzzfile(yaml_text, cluster=self.cluster)
                        wf_id = mcp.workflow_start(yaml_text, name=workflow_name, fuzzfile_path=fz_path)
                    else:
                        raise
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

            job_name = f"train-{trial_id}" if kind == "train" else f"bench-{trial_id}"
            init_info = inspect_workflow_stages(desc_initial)
            if init_info["has_status"] and "workflow_describe" in mcp.advertised_tools:
                # Non-blocking live queue & GPU execution polling (supports multi-hour GPU queues cleanly!)
                t_wait0 = time.monotonic()
                last_phase = ""
                seen_steps = 0
                while True:
                    elapsed_s = int(time.monotonic() - t_wait0)
                    if elapsed_s > self.timeout_seconds:
                        raise FuzzballMCPError(
                            f"Workflow {wf_id} ({workflow_name}) exceeded queue/run timeout ({self.timeout_seconds}s)."
                        )
                    cur_desc = mcp.workflow_describe(wf_id)
                    nodes = _extract_placed_nodes(cur_desc, self.cluster, requested_nodes, seq_idx)
                    st_info = inspect_workflow_stages(cur_desc)
                    resolved_job = st_info["job_name"] or job_name

                    if st_info["terminal"]:
                        if st_info["phase"] == "FINISHED":
                            break
                        log_tail = ""
                        if "workflow_log" in mcp.advertised_tools:
                            try:
                                raw_log = mcp.workflow_log(wf_id, resolved_job)
                                if raw_log.strip():
                                    log_tail = "\nLast job logs:\n" + "\n".join(raw_log.strip().splitlines()[-25:])
                            except Exception:
                                pass
                        raise FuzzballMCPError(
                            f"Fuzzball workflow {wf_id} ({workflow_name}) {st_info['phase']}: "
                            f"{st_info['error']}{log_tail}"
                        )

                    if st_info["phase"] == "QUEUED":
                        mins, secs = divmod(elapsed_s, 60)
                        self._emit(
                            {
                                "type": "trial_queued",
                                "trial_id": trial_id,
                                "workflow_id": wf_id,
                                "elapsed_s": elapsed_s,
                                "message": f"QUEUED — Waiting for free {self.cluster.definition_id} GPU ({mins}m {secs:02d}s elapsed)",
                            }
                        )
                    elif st_info["phase"] == "RUNNING":
                        if last_phase != "RUNNING":
                            self._emit(
                                {
                                    "type": "trial_running",
                                    "trial_id": trial_id,
                                    "workflow_id": wf_id,
                                    "nodes_assigned": nodes,
                                    "message": f"RUNNING ON GPU — {trial_id} executing on {', '.join(nodes)}",
                                }
                            )
                        if "workflow_log" in mcp.advertised_tools:
                            try:
                                cur_log = mcp.workflow_log(wf_id, resolved_job)
                                steps = _parse_steps_from_logs(cur_log)
                                for step_rec in steps[seen_steps:]:
                                    self._emit({"type": "trial_step", "trial_id": trial_id, **step_rec})
                                seen_steps = len(steps)
                            except Exception:
                                pass
                    last_phase = st_info["phase"]
                    time.sleep(self.poll_interval_s)
            else:
                wait_res = mcp.workflow_wait(wf_id, timeout_seconds=self.timeout_seconds)
                wait_info = inspect_workflow_stages(wait_res)
                if wait_info["terminal"] and wait_info["phase"] != "FINISHED":
                    log_tail = ""
                    if "workflow_log" in mcp.advertised_tools:
                        try:
                            raw_log = mcp.workflow_log(wf_id, job_name)
                            if raw_log.strip():
                                log_tail = "\nLast job logs:\n" + "\n".join(raw_log.strip().splitlines()[-25:])
                        except Exception:
                            pass
                    raise FuzzballMCPError(
                        f"Fuzzball MCP workflow {wf_id} ({workflow_name}) ended with {wait_info['phase']}: "
                        f"{wait_info['error']}{log_tail}"
                    )
                if "workflow_describe" in mcp.advertised_tools:
                    nodes = _extract_placed_nodes(
                        mcp.workflow_describe(wf_id), self.cluster, requested_nodes, seq_idx
                    )
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
        inline_splits = _compact_splits_json(self.processed_data_dir)
        raw, wf_id, nodes = self._execute_and_fetch_metrics(
            lambda c: render_training_trial_fuzzfile(
                c,
                self.exp,
                trial_id,
                recipe,
                perf,
                max_steps,
                self.run_id,
                inline_splits_json=inline_splits,
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

    def _empirical_split_calibration(self) -> tuple[float, float, float]:
        """Measure actual token/bigram overlap across `train.jsonl`, `val.jsonl`, and `held_out.jsonl`
        so validation and held-out test losses directly reflect the active preset or uploaded dataset."""
        if self.processed_data_dir is None or not (self.processed_data_dir / "train.jsonl").exists():
            return self.BASE_MODEL_HELD_OUT_LOSS, 0.018, 0.011

        def _read_bigrams(fname: str) -> set[tuple[int, int]]:
            p = self.processed_data_dir / fname
            if not p.exists():
                return set()
            bgs: set[tuple[int, int]] = set()
            for ln in p.read_text(encoding="utf-8").splitlines()[:24]:
                if not ln.strip():
                    continue
                ids = json.loads(ln).get("input_ids") or []
                for i in range(min(len(ids) - 1, 256)):
                    bgs.add((int(ids[i]), int(ids[i + 1])))
            return bgs

        train_bg = _read_bigrams("train.jsonl")
        val_bg = _read_bigrams("val.jsonl")
        test_bg = _read_bigrams("held_out.jsonl")
        if not train_bg or not val_bg or not test_bg:
            return self.BASE_MODEL_HELD_OUT_LOSS, 0.018, 0.011

        val_overlap = len(train_bg & val_bg) / float(max(len(val_bg), 1))
        test_overlap = len(train_bg & test_bg) / float(max(len(test_bg), 1))
        # Preserve exact 1.9240 anchor on sample_lammps while reflecting custom dataset difficulty
        manifest_path = self.processed_data_dir / "dataset_manifest.json"
        ds_name = ""
        if manifest_path.exists():
            ds_name = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_name", ""))
        if ds_name in {"lammps-cpp", "builtin-lammps-cpp-bench"}:
            return self.BASE_MODEL_HELD_OUT_LOSS, 0.018, 0.011

        base_loss = round(2.04 - 0.32 * min(0.85, val_overlap), 4)
        val_gap = round(max(0.010, 0.032 * (1.0 - min(0.8, val_overlap))), 5)
        test_gap = round(max(0.008, 0.024 * (1.0 - min(0.8, test_overlap))), 5)
        return base_loss, val_gap, test_gap

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
        comm_mode = (
            "NCCL_FSDP_SHARDED"
            if (perf.nodes > 1 and self.exp.model.min_nodes_required > 1)
            else ("NCCL_DDP_ALLREDUCE" if perf.nodes > 1 else "INDEPENDENT_1GPU")
        )
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": "train",
                "workflow_id": wf_id,
                "workflow_name": f"crucible-{trial_id}",
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": nodes,
                "world_size": perf.nodes,
                "comm_mode": comm_mode,
                "lora_rank": recipe.lora_rank,
                "lr": recipe.lr,
                "micro_batch_size": perf.micro_batch_size,
                "grad_accum_steps": perf.grad_accum_steps,
                "kernels_enabled": {
                    "chunked_cross_entropy": perf.use_chunked_ce_kernel,
                    "grouped_moe_gemm": perf.use_grouped_moe_kernel and (self.exp.model.num_experts > 1),
                    "fused_lora_linear": perf.use_fused_lora_kernel,
                },
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

        base_eval_loss, val_gap, test_gap = self._empirical_split_calibration()
        model_scale_bonus = min(0.11, 0.025 * math.log2(max(self.exp.model.total_params_b, 2.0) / 7.0))
        base_eval_loss = round(max(1.45, base_eval_loss - model_scale_bonus + (0.049 if abs(self.exp.model.total_params_b - 30.5) < 0.2 else 0.0)), 4)
        if abs(self.exp.model.total_params_b - 30.5) < 0.2 and val_gap == 0.018:
            base_eval_loss = self.BASE_MODEL_HELD_OUT_LOSS

        optimal_log_lr = math.log10(1.6e-4)
        lr_dist = abs(math.log10(recipe.lr) - optimal_log_lr)
        lr_penalty = 0.22 * (lr_dist**1.6)
        module_coverage = len(recipe.target_modules) / 7.0
        rank_gain = 0.14 * min(1.0, math.log2(max(recipe.lora_rank, 2)) / 5.0)
        seq_bonus = 0.04 if recipe.seq_len >= 2048 else -0.05
        asymptote = max(
            1.08,
            base_eval_loss - 0.38 * module_coverage - rank_gain - seq_bonus + lr_penalty,
        )

        global_batch_delta = abs(perf.global_batch_sequences - 16) * 0.0012
        is_moe = self.exp.model.num_experts > 1
        effective_moe_kernel = perf.use_grouped_moe_kernel and is_moe
        kernel_eps = (0.0011 if perf.use_chunked_ce_kernel else 0.0) + (
            0.0008 if effective_moe_kernel else 0.0
        )

        # Scale baseline throughput inversely with active parameter FLOPs (anchored at 3.3B active = 1120 tok/s)
        active_ratio = max(0.35, self.exp.model.active_params_b / 3.3)
        base_tok_s = 1120.0 / (active_ratio**0.62)
        batch_mult = 1.0 + 0.11 * min(perf.micro_batch_size - 1, 5)
        ckpt_mult = 1.0 if perf.activation_checkpointing else 1.21
        ce_mult = 1.24 if perf.use_chunked_ce_kernel else 1.0
        moe_mult = 1.31 if effective_moe_kernel else 1.0
        lora_mult = 1.12 if perf.use_fused_lora_kernel else 1.0
        node_mult = 1.0 if perf.nodes == 1 else (perf.nodes * 0.91)
        tok_s = round(base_tok_s * batch_mult * ckpt_mult * ce_mult * moe_mult * lora_mult * node_mult, 2)

        peak_cluster_tok_s = (self.cluster.nodes.measured_peak_train_tok_s_per_node / (active_ratio**0.62)) * perf.nodes
        hw_util_pct = round(min(96.5, (tok_s / max(peak_cluster_tok_s, 1.0)) * 100.0), 2)

        rate = 0.011 * min(2.2, (recipe.lr / 1.5e-4) ** 0.45)
        loss_curve: list[float] = []
        val_curve: list[float] = []
        metrics_path = trial_dir / "metrics.jsonl"
        tokens_per_step = perf.global_batch_sequences * recipe.seq_len
        cum_wall = 0.0
        emit_every = max(1, max_steps // (10 if self.step_delay_s > 0 else 4))
        warmup_steps = max(2, int(max_steps * recipe.warmup_ratio))
        raw_logit_gb = round(1.85 * perf.micro_batch_size * (recipe.seq_len / 2048.0), 2)
        saved_logit_gb = round(max(0.0, raw_logit_gb - logit_buffer_gb), 2)

        with metrics_path.open("w", encoding="utf-8") as fh:
            for step in range(1, max_steps + 1):
                decay = math.exp(-rate * step)
                osc = 0.008 * math.sin(step * 0.37) * math.exp(-0.004 * step)
                instability = (
                    0.09 * math.sin(step * 0.9) if recipe.lr >= 7e-4 and step > 25 else 0.0
                )
                step_loss = (
                    asymptote
                    + (base_eval_loss - asymptote) * decay
                    + osc
                    + instability
                    + global_batch_delta
                    + kernel_eps
                )
                step_loss = round(step_loss, 5)
                step_val = round(step_loss + val_gap, 5)
                loss_curve.append(step_loss)
                val_curve.append(step_val)
                step_dt_s = tokens_per_step / max(tok_s, 1.0)
                cum_wall += step_dt_s

                # Learning rate schedule (linear warmup + cosine decay, matching train_lora.py)
                if step <= warmup_steps:
                    cur_lr = recipe.lr * (step / float(warmup_steps))
                else:
                    prog = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                    cur_lr = recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

                # Gradient L2 norm & MoE router load-balancing entropy (matching RouterStats)
                grad_spike = abs(0.35 * math.sin(step * 0.9)) if recipe.lr >= 7e-4 and step > 20 else 0.0
                grad_norm = round(
                    min(1.0, max(0.09, 0.36 * math.exp(-0.005 * step) + 0.14 + 0.03 * abs(math.sin(step * 0.5)) + grad_spike)),
                    4,
                )
                router_ent = (
                    round(min(0.994, 0.935 + 0.045 * (1.0 - math.exp(-0.02 * step)) - 0.015 * grad_spike), 4)
                    if is_moe
                    else 1.0
                )

                # Inter-GPU collective sync metrics calibrated to GB10 2-node RoCE / c10d probes
                if perf.nodes > 1:
                    adapter_mb = round(0.85 * recipe.lora_rank * len(recipe.target_modules) * (self.exp.model.num_layers / 48.0), 2)
                    fsdp_shard_mb = round(self.exp.model.bf16_footprint_gb * 120.0, 1) if self.exp.model.min_nodes_required > 1 else 0.0
                    payload_mb = round(adapter_mb * 2.0 + fsdp_shard_mb, 2)
                    base_ar_ms = 53.2 + 0.38 * adapter_mb + (115.0 if fsdp_shard_mb > 0 else 0.0)
                    allreduce_ms = round(base_ar_ms * (1.0 + 0.14 * abs(math.sin(step * 0.7))), 2)
                    inter_gpu_gbps = round(min(192.0, max(12.4, (payload_mb * 8.0) / max(allreduce_ms / 1000.0, 0.004))), 2)
                else:
                    payload_mb = 0.0
                    allreduce_ms = 0.0
                    inter_gpu_gbps = 0.0

                bf16_tflops = round(min(99.1 * perf.nodes, (tok_s * 4.0 * self.exp.model.active_params_b) / 1000.0 * 3.8), 2)

                if step == 1 or step == max_steps or step % 15 == 0:
                    rec = StepMetricRecord(
                        step=step,
                        train_loss=step_loss,
                        val_loss=step_val,
                        tokens_per_sec=tok_s,
                        peak_mem_gb=peak_mem_gb,
                        wall_time_s=round(cum_wall, 2),
                        world_size=perf.nodes,
                    )
                    fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
                if step == 1 or step == max_steps or step % emit_every == 0:
                    self._emit(
                        {
                            "type": "trial_step",
                            "trial_id": trial_id,
                            "step": step,
                            "max_steps": max_steps,
                            "train_loss": step_loss,
                            "val_loss": step_val,
                            "perplexity": round(math.exp(min(step_val, 12.0)), 4),
                            "lr": round(cur_lr, 7),
                            "grad_norm": grad_norm,
                            "router_entropy": router_ent,
                            "tokens_per_sec": tok_s,
                            "tokens_seen": step * tokens_per_step,
                            "step_time_ms": round(step_dt_s * 1000.0, 1),
                            "allreduce_ms": allreduce_ms,
                            "payload_mb": payload_mb,
                            "inter_gpu_gbps": inter_gpu_gbps,
                            "comm_mode": comm_mode,
                            "peak_mem_gb": peak_mem_gb,
                            "logit_mem_saved_gb": saved_logit_gb,
                            "hw_util_pct": hw_util_pct,
                            "bf16_tflops": bf16_tflops,
                            "world_size": perf.nodes,
                            "nodes_assigned": nodes,
                            "partial_curve": list(loss_curve),
                            "partial_val_curve": list(val_curve),
                        }
                    )
                    if self.step_delay_s > 0:
                        time.sleep(min(0.035, self.step_delay_s * 0.45))

        final_val = round(loss_curve[-1] + val_gap + self._jitter(recipe.short_id(), 0.002), 5)
        held_out = round(final_val + test_gap, 5)
        summary = TrainingTrialSummary(
            trial_id=trial_id,
            status="completed",
            steps_completed=max_steps,
            initial_val_loss=base_eval_loss,
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
                "grouped_moe_gemm": effective_moe_kernel,
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
        srv_comm_mode = (
            "VLLM_TP2_ALLREDUCE"
            if cand.tensor_parallel > 1
            else ("VLLM_DUAL_REPLICA" if cand.replicas > 1 else "VLLM_SINGLE_GPU")
        )
        self._emit(
            {
                "type": "trial_started",
                "trial_id": trial_id,
                "kind": "serve",
                "workflow_id": wf_id,
                "workflow_name": f"crucible-srv-{trial_id}",
                "definition_id": self.cluster.definition_id,
                "nodes_assigned": nodes,
                "world_size": cand.total_gpus_required,
                "comm_mode": srv_comm_mode,
                "precision": cand.precision,
                "tensor_parallel": cand.tensor_parallel,
                "replicas": cand.replicas,
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
        tp_bw_gbps = round(min(184.0, (tok_s * 0.14)), 2) if cand.tensor_parallel > 1 else 0.0
        self._emit(
            {
                "type": "trial_step",
                "trial_id": trial_id,
                "kind": "serve",
                "step": 1,
                "max_steps": 1,
                "train_loss": round(baseline_finetuned_loss, 5),
                "val_loss": round(srv_loss, 5),
                "perplexity": round(math.exp(min(srv_loss, 12.0)), 4),
                "tokens_per_sec": tok_s,
                "ttft_ms": ttft_ms,
                "p50_latency_ms": p50_ms,
                "p99_latency_ms": p99_ms,
                "allreduce_ms": tp_comm_overhead,
                "inter_gpu_gbps": tp_bw_gbps,
                "comm_mode": srv_comm_mode,
                "precision": cand.precision,
                "tensor_parallel": cand.tensor_parallel,
                "replicas": cand.replicas,
                "world_size": cand.total_gpus_required,
                "nodes_assigned": nodes,
            }
        )

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
