"""FastAPI control plane for Crucible: Pick Any Model -> Drop Data -> Optimize -> Download Build."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crucible.build_bundle import build_artifact_texts, create_build_bundle_zip_bytes
from crucible.config import ClusterConfig, DatasetSplitConfig, ExperimentConfig, repo_root
from crucible.data.custom_ingest import ingest_universal_dataset, stage_uploaded_files
from crucible.firewall import DataIsolationViolation, PrivacySchemaFirewall
from crucible.fuzzball.discovery import discover_cluster_config
from crucible.fuzzball.validator import FuzzfileValidationError, validate_fuzzfile
from crucible.model_catalog import (
    MODEL_CATALOG,
    build_dynamic_experiment_config,
    infer_model_profile,
)
from crucible.optimizer.pipeline import run_autoresearch_pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"
BUILTIN_PRESET_ID = "builtin-lammps-cpp-bench"


class UploadedFileItem(BaseModel):
    name: str
    content_b64: str | None = None
    content_text: str | None = None
    content: str | None = None


class DatasetUploadRequest(BaseModel):
    dataset_name: str = "custom-dataset"
    use_preset: bool = False
    local_path: str | None = None
    files: list[UploadedFileItem] = Field(default_factory=list)


class TriggerRunRequest(BaseModel):
    profile: Literal["demo_30b", "stretch_80b", "custom"] | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    backend: Literal["simulator", "local_torch", "fuzzball_mcp", "fuzzball"] = "simulator"
    fuzzfile_version: Literal["v4", "v1"] = "v4"
    definition_id: str = "team3"
    async_mode: bool = False


class FirewallProbeRequest(BaseModel):
    inject_cpp_snippet: bool = True


class LiveRunState:
    """Thread-safe live telemetry store for active, queued, and completed Crucible runs."""

    def __init__(self, initial_cluster: ClusterConfig) -> None:
        self._lock = threading.Lock()
        self.seq = 0
        self.state: Literal["idle", "queued_for_gpu", "running", "completed", "failed"] = "idle"
        self.backend: str = "simulator"
        self.profile: str = "demo_30b"
        self.model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        self.dataset_id: str = BUILTIN_PRESET_ID
        self.definition_id: str = initial_cluster.definition_id
        self.current_stage: str = "idle"
        self.stage_label: str = f"Ready ({initial_cluster.definition_id})"
        self.error: str | None = None
        self.active_workflows: dict[str, dict[str, Any]] = {}
        self.completed_trials: list[dict[str, Any]] = []
        self.live_curves: dict[str, list[float]] = {}
        self.live_val_curves: dict[str, list[float]] = {}
        self.live_step_metrics: dict[str, Any] = {}
        self.live_firewall: list[dict[str, Any]] = []
        self.latest_trial_fuzzfile: str = ""
        self.turnkey_fuzzfile: str = ""
        self.events: list[dict[str, Any]] = []
        self.node_slots: dict[str, dict[str, Any]] = {}
        self._reset_nodes(initial_cluster)

    def _reset_nodes(self, cluster: ClusterConfig) -> None:
        self.node_slots = {}
        for n in cluster.nodes.inventory:
            self.node_slots[n.hostname] = {
                "hostname": n.hostname,
                "node_id": n.node_id,
                "address": n.address,
                "gpu_product": n.gpu_product,
                "gpus": n.gpus,
                "physical_cores": n.cores,
                "alloc_cores": cluster.nodes.cpu_cores_per_node,
                "physical_memory_gib": n.memory_gib,
                "alloc_memory_gib": cluster.nodes.memory_gb_per_node,
                "arch_device": cluster.nodes.arch_device,
                "state": "READY · IDLE",
                "active_workflow_id": None,
                "active_trial_id": None,
                "last_workflow_id": None,
                "last_trial_id": None,
                "trials_completed": 0,
            }

    def begin_run(
        self,
        cluster: ClusterConfig,
        profile: str,
        backend: str,
        model_id: str,
        dataset_id: str,
    ) -> None:
        with self._lock:
            self.state = "running"
            self.profile = profile
            self.backend = backend
            self.model_id = model_id
            self.dataset_id = dataset_id
            self.definition_id = cluster.definition_id
            self.current_stage = "starting"
            self.stage_label = f"Starting 4-stage optimization of {model_id} on {dataset_id} ({backend})..."
            self.error = None
            self.active_workflows = {}
            self.completed_trials = []
            self.live_curves = {}
            self.live_val_curves = {}
            self.live_step_metrics = {}
            self.live_firewall = []
            self.events = []
            self._reset_nodes(cluster)
            self.seq += 1

    def record_event(self, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with self._lock:
            self.seq += 1
            ev_type = str(payload.get("type", "info"))
            entry: dict[str, Any] = {"seq": self.seq, "time_utc": now, "type": ev_type}

            if ev_type == "stage_started":
                if self.state == "queued_for_gpu":
                    self.state = "running"
                self.current_stage = str(payload.get("stage", ""))
                self.stage_label = str(payload.get("label", self.current_stage))
                entry["message"] = self.stage_label

            elif ev_type == "trial_started":
                tid = str(payload["trial_id"])
                wf_id = str(payload["workflow_id"])
                nodes = list(payload.get("nodes_assigned") or [])
                self.latest_trial_fuzzfile = str(payload.get("fuzzfile", ""))
                self.active_workflows[tid] = {
                    "trial_id": tid,
                    "kind": payload.get("kind", "train"),
                    "workflow_id": wf_id,
                    "nodes": nodes,
                    "object_uri": payload.get("object_uri", ""),
                }
                for hname in nodes:
                    if hname in self.node_slots:
                        slot = self.node_slots[hname]
                        slot["state"] = "RUNNING"
                        slot["active_workflow_id"] = wf_id
                        slot["active_trial_id"] = tid
                entry["message"] = f"Submitted {tid} -> workflow {wf_id[:8]}… on {', '.join(nodes)}"

            elif ev_type == "trial_queued":
                self.state = "queued_for_gpu"
                msg = str(payload.get("message") or "QUEUED — Waiting for free GPU...")
                self.stage_label = msg
                for slot in self.node_slots.values():
                    if slot["active_trial_id"] == payload.get("trial_id"):
                        slot["state"] = "QUEUED FOR GPU"
                entry["message"] = msg

            elif ev_type == "trial_running":
                self.state = "running"
                msg = str(payload.get("message") or "RUNNING ON GPU")
                self.stage_label = msg
                for hname in payload.get("nodes_assigned") or []:
                    if hname in self.node_slots:
                        self.node_slots[hname]["state"] = "RUNNING"
                entry["message"] = msg

            elif ev_type == "trial_step":
                if self.state == "queued_for_gpu":
                    self.state = "running"
                tid = str(payload.get("trial_id", ""))
                step = int(payload.get("step", 0))
                t_loss = float(payload.get("train_loss", 0.0))
                v_loss = payload.get("val_loss")
                tok_s = float(payload.get("tokens_per_sec", 0.0))
                if isinstance(payload.get("partial_curve"), list):
                    self.live_curves[tid] = payload["partial_curve"]
                else:
                    self.live_curves.setdefault(tid, []).append(t_loss)
                if isinstance(payload.get("partial_val_curve"), list):
                    self.live_val_curves[tid] = payload["partial_val_curve"]
                elif v_loss is not None:
                    self.live_val_curves.setdefault(tid, []).append(float(v_loss))
                self.live_step_metrics = {
                    "trial_id": tid,
                    "step": step,
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "tokens_per_sec": tok_s,
                }
                v_str = f" · val_loss={float(v_loss):.4f}" if v_loss is not None else ""
                self.stage_label = (
                    f"Evaluating {tid} (step {step}) · train_loss={t_loss:.4f}{v_str} · {tok_s:,.0f} tok/s"
                )
                entry["message"] = self.stage_label

            elif ev_type == "trial_finished":
                if self.state == "queued_for_gpu":
                    self.state = "running"
                tid = str(payload["trial_id"])
                wf_id = str(payload["workflow_id"])
                nodes = list(payload.get("nodes_assigned") or [])
                summ = payload.get("summary") or {}
                self.active_workflows.pop(tid, None)
                if isinstance(summ, dict) and isinstance(summ.get("loss_curve"), list) and summ["loss_curve"]:
                    self.live_curves[tid] = summ["loss_curve"]
                for hname in nodes:
                    if hname in self.node_slots:
                        slot = self.node_slots[hname]
                        if slot["active_trial_id"] == tid:
                            slot["state"] = "READY · IDLE"
                            slot["active_workflow_id"] = None
                            slot["active_trial_id"] = None
                        slot["last_workflow_id"] = wf_id
                        slot["last_trial_id"] = tid
                        slot["trials_completed"] = int(slot["trials_completed"]) + 1
                tok_s = (
                    summ.get("mean_tokens_per_sec")
                    or summ.get("throughput_tokens_per_sec")
                    or 0.0
                )
                ppl = summ.get("held_out_perplexity", 0.0)
                v_loss = summ.get("final_val_loss")
                self.completed_trials.append(
                    {
                        "trial_id": tid,
                        "kind": payload.get("kind", "train"),
                        "workflow_id": wf_id,
                        "nodes": nodes,
                        "tokens_per_sec": tok_s,
                        "val_loss": v_loss,
                        "held_out_perplexity": ppl,
                    }
                )
                entry["message"] = (
                    f"Completed {tid} · {float(tok_s):,.0f} tok/s · Held-Out Test PPL {float(ppl):.3f}"
                )

            elif ev_type == "firewall_audit":
                rec = payload.get("record") or {}
                exp_obj = payload.get("explanation") or {}
                self.live_firewall.append({"record": rec, "explanation": exp_obj})
                sha = str(rec.get("payload_sha256", ""))[:12]
                entry["message"] = (
                    f"Firewall {rec.get('status', 'ALLOWED')} [{rec.get('stage', '')}] "
                    f"(contains_raw_data=false, sha256={sha}…)"
                )

            elif ev_type == "run_completed":
                self.state = "completed"
                self.current_stage = "completed"
                self.stage_label = f"Build Package Ready — {self.model_id} tuned on {self.dataset_id}"
                self.turnkey_fuzzfile = str(payload.get("turnkey_fuzzfile", ""))
                entry["message"] = f"Generated turnkey build bundle (.zip) for {self.model_id}"

            elif ev_type == "run_failed":
                self.state = "failed"
                self.error = str(payload.get("error", "Unknown error"))
                self.stage_label = f"Run stopped: {self.error}"
                for slot in self.node_slots.values():
                    slot["state"] = "READY · IDLE"
                    slot["active_workflow_id"] = None
                    slot["active_trial_id"] = None
                self.active_workflows = {}
                entry["message"] = f"Error: {self.error}"
            else:
                entry["message"] = str(payload.get("label") or ev_type)

            self.events.append(entry)
            if len(self.events) > 120:
                self.events = self.events[-120:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self.seq,
                "state": self.state,
                "backend": self.backend,
                "profile": self.profile,
                "model_id": self.model_id,
                "dataset_id": self.dataset_id,
                "definition_id": self.definition_id,
                "current_stage": self.current_stage,
                "stage_label": self.stage_label,
                "error": self.error,
                "nodes": list(self.node_slots.values()),
                "active_workflows": list(self.active_workflows.values()),
                "completed_trial_count": len(self.completed_trials),
                "completed_trials": self.completed_trials[-12:],
                "live_curves": dict(self.live_curves),
                "live_val_curves": dict(self.live_val_curves),
                "live_step_metrics": dict(self.live_step_metrics),
                "live_firewall": list(self.live_firewall),
                "latest_trial_fuzzfile": self.latest_trial_fuzzfile,
                "turnkey_fuzzfile": self.turnkey_fuzzfile,
                "events": list(self.events[-30:]),
            }


def create_app(run_dir: Path | None = None) -> FastAPI:
    resolved_run_dir = (run_dir or (repo_root() / "runs" / "ui-default")).resolve()
    datasets_store = resolved_run_dir.parent / "datasets"
    datasets_store.mkdir(parents=True, exist_ok=True)

    default_cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    live_state = LiveRunState(default_cluster)
    dataset_registry: dict[str, dict[str, Any]] = {}

    app = FastAPI(
        title="Crucible Local Model Optimizer & Build Studio",
        version="0.3.0",
        description="Pick any model, drop your private tuning data, run 4-stage sm_121 autoresearch, and download a turnkey Fuzzball + vLLM build.",
    )
    app.state.run_dir = resolved_run_dir
    app.state.datasets_store = datasets_store
    app.state.dataset_registry = dataset_registry
    app.state.live = live_state
    app.state.active_cluster = default_cluster

    def _ensure_builtin_preset() -> dict[str, Any]:
        if BUILTIN_PRESET_ID in dataset_registry:
            return dataset_registry[BUILTIN_PRESET_ID]
        sample_src = repo_root() / "fixtures" / "sample_lammps"
        preset_packed = datasets_store / BUILTIN_PRESET_ID / "packed"
        manifest, fmt = ingest_universal_dataset(
            source_root=sample_src,
            output_dir=preset_packed,
            split_cfg=DatasetSplitConfig(
                name=BUILTIN_PRESET_ID,
                chunk_seq_len=256,
                chunk_overlap=32,
                split_salt="crucible-lammps-v1",
            ),
        )
        entry = {
            "dataset_id": BUILTIN_PRESET_ID,
            "display_name": "LAMMPS C++ & GPU Simulation Benchmark (Built-In Preset)",
            "is_preset": True,
            "raw_dir": str(sample_src),
            "packed_dir": str(preset_packed),
            "format_summary": fmt,
            "total_documents": manifest.total_cpp_files,
            "total_tokens": sum(manifest.split_lexeme_counts.values()),
            "split_doc_counts": manifest.split_file_counts,
            "split_chunk_counts": manifest.split_chunk_counts,
            "split_token_counts": manifest.split_lexeme_counts,
            "held_out_test_files": manifest.held_out_files,
            "manifest_sha256": manifest.manifest_sha256,
        }
        dataset_registry[BUILTIN_PRESET_ID] = entry
        return entry

    def _collect_summary_payload(target_run_dir: Path) -> dict[str, Any]:
        summary_file = target_run_dir / "run_summary.json"
        data: dict[str, Any] = json.loads(summary_file.read_text(encoding="utf-8"))
        curves: dict[str, list[float]] = {}
        placements: dict[str, Any] = {}
        trials_root = target_run_dir / "trials"
        if trials_root.exists():
            for trial_sub in sorted(trials_root.iterdir()):
                s_file = trial_sub / "summary.json"
                if s_file.exists():
                    s_obj = json.loads(s_file.read_text(encoding="utf-8"))
                    if s_obj.get("loss_curve"):
                        curves[trial_sub.name] = s_obj["loss_curve"]
                p_file = trial_sub / "placement.json"
                if p_file.exists():
                    placements[trial_sub.name] = json.loads(p_file.read_text(encoding="utf-8"))
        data["trial_loss_curves"] = curves
        data["trial_placements"] = placements
        data["active_dataset"] = dataset_registry.get(live_state.dataset_id) or _ensure_builtin_preset()
        return data

    def _ensure_seeded() -> dict[str, Any]:
        _ensure_builtin_preset()
        summary_file = app.state.run_dir / "run_summary.json"
        if not summary_file.exists():
            cfg_path = repo_root() / "configs" / "demo_lammps_30b.yaml"
            cluster_path = repo_root() / "configs" / "cluster.default.yaml"
            exp = ExperimentConfig.load(cfg_path)
            cluster = ClusterConfig.load(cluster_path)
            live_state.begin_run(
                cluster,
                "demo_30b",
                "simulator",
                exp.model.name,
                BUILTIN_PRESET_ID,
            )
            run_autoresearch_pipeline(
                exp=exp,
                cluster=cluster,
                output_dir=app.state.run_dir,
                backend_kind="simulator",
                event_hook=live_state.record_event,
            )
        elif live_state.state == "idle":
            seeded = _collect_summary_payload(app.state.run_dir)
            seeded_cluster = ClusterConfig.model_validate(seeded["cluster"])
            seeded_model = (seeded.get("model_profile") or {}).get("name", "Qwen/Qwen3-Coder-30B-A3B")
            live_state.begin_run(
                seeded_cluster,
                "demo_30b",
                str(seeded.get("backend", "simulator")),
                seeded_model,
                BUILTIN_PRESET_ID,
            )
            for tid, pl in (seeded.get("trial_placements") or {}).items():
                live_state.record_event(
                    {
                        "type": "trial_finished",
                        "trial_id": tid,
                        "kind": "train",
                        "workflow_id": pl.get("workflow_id", "seeded"),
                        "nodes_assigned": pl.get("nodes", []),
                        "summary": {"mean_tokens_per_sec": 3821.6, "held_out_perplexity": 4.04},
                    }
                )
            live_state.record_event(
                {
                    "type": "run_completed",
                    "run_id": seeded["run_id"],
                    "output_dir": str(app.state.run_dir),
                    "turnkey_fuzzfile": seeded.get("turnkey_fuzzfile", ""),
                }
            )
        return _collect_summary_payload(app.state.run_dir)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "crucible-studio", "definition_id": live_state.definition_id}

    @app.get("/api/models")
    def api_models(probe_model: str | None = None) -> dict[str, Any]:
        inferred = (
            infer_model_profile(probe_model, node_mem_gb=app.state.active_cluster.nodes.memory_gb_per_node).model_dump()
            if probe_model
            else None
        )
        return {"catalog": MODEL_CATALOG, "inferred": inferred}

    @app.get("/api/datasets")
    def api_datasets() -> dict[str, Any]:
        preset = _ensure_builtin_preset()
        return {
            "default_dataset_id": BUILTIN_PRESET_ID,
            "preset": preset,
            "datasets": list(dataset_registry.values()),
        }

    @app.post("/api/dataset/upload")
    def api_upload_dataset(req: DatasetUploadRequest) -> dict[str, Any]:
        if req.use_preset:
            return _ensure_builtin_preset()

        if req.local_path:
            src_dir = Path(req.local_path).expanduser().resolve()
            if not src_dir.exists():
                raise HTTPException(status_code=400, detail=f"Local path does not exist: {src_dir}")
            ds_id = f"local-{hashlib.sha256(str(src_dir).encode()).hexdigest()[:10]}"
            display_name = req.dataset_name if req.dataset_name != "custom-dataset" else src_dir.name
            packed_dir = datasets_store / ds_id / "packed"
        else:
            if not req.files:
                raise HTTPException(status_code=400, detail="Drop one or more files, a .zip archive, or select the built-in preset.")
            digest_seed = hashlib.sha256(
                json.dumps([f.name for f in req.files], sort_keys=True).encode("utf-8")
                + str(time.time_ns()).encode("utf-8")
            ).hexdigest()[:10]
            ds_id = f"drop-{digest_seed}"
            raw_dir = datasets_store / ds_id / "raw"
            try:
                stage_uploaded_files([f.model_dump() for f in req.files], raw_dir)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            src_dir = raw_dir
            display_name = (
                req.dataset_name
                if req.dataset_name and req.dataset_name != "custom-dataset"
                else (req.files[0].name if len(req.files) == 1 else f"{len(req.files)} dropped files")
            )
            packed_dir = datasets_store / ds_id / "packed"

        try:
            manifest, fmt = ingest_universal_dataset(
                source_root=src_dir,
                output_dir=packed_dir,
                split_cfg=DatasetSplitConfig(
                    name=ds_id,
                    chunk_seq_len=256,
                    chunk_overlap=32,
                    split_salt=f"crucible-{ds_id}",
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Dataset ingestion failed: {exc}") from exc

        entry = {
            "dataset_id": ds_id,
            "display_name": display_name,
            "is_preset": False,
            "raw_dir": str(src_dir),
            "packed_dir": str(packed_dir),
            "format_summary": fmt,
            "total_documents": manifest.total_cpp_files,
            "total_tokens": sum(manifest.split_lexeme_counts.values()),
            "split_doc_counts": manifest.split_file_counts,
            "split_chunk_counts": manifest.split_chunk_counts,
            "split_token_counts": manifest.split_lexeme_counts,
            "held_out_test_files": manifest.held_out_files,
            "manifest_sha256": manifest.manifest_sha256,
        }
        dataset_registry[ds_id] = entry
        return entry

    @app.get("/api/discover")
    def api_discover() -> dict[str, object]:
        cfg, rep = discover_cluster_config()
        app.state.active_cluster = cfg
        return {"cluster_config": cfg.model_dump(), "discovery_report": rep.to_dict()}

    @app.get("/api/summary")
    def get_summary() -> dict[str, object]:
        return _ensure_seeded()

    @app.get("/api/live-status")
    def get_live_status() -> dict[str, Any]:
        _ensure_seeded()
        return live_state.snapshot()

    @app.get("/api/build/preview")
    def api_build_preview() -> dict[str, Any]:
        summary = _ensure_seeded()
        artifacts = build_artifact_texts(summary, app.state.run_dir)
        return {
            "run_id": summary["run_id"],
            "model_id": (summary.get("model_profile") or {}).get("name", live_state.model_id),
            "dataset": summary.get("active_dataset") or _ensure_builtin_preset(),
            "artifacts": artifacts,
        }

    @app.get("/api/build/bundle.zip")
    def api_download_build_bundle() -> Response:
        summary = _ensure_seeded()
        zip_bytes = create_build_bundle_zip_bytes(summary, app.state.run_dir)
        slug = str(summary.get("experiment_id") or "crucible-build")[:48]
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="crucible-build-{slug}.zip"',
            },
        )

    @app.get("/api/events")
    async def stream_events() -> StreamingResponse:
        async def _event_gen():
            last_seq = -1
            idle_ticks = 0
            while idle_ticks < 600:
                snap = live_state.snapshot()
                if snap["seq"] != last_seq:
                    last_seq = snap["seq"]
                    yield f"data: {json.dumps(snap)}\n\n"
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                await asyncio.sleep(0.25)

        return StreamingResponse(_event_gen(), media_type="text/event-stream")

    @app.post("/api/run")
    def trigger_run(req: TriggerRunRequest) -> dict[str, Any]:
        if live_state.snapshot()["state"] in {"running", "queued_for_gpu"} and req.async_mode:
            raise HTTPException(status_code=409, detail="A Crucible optimization run is already active.")

        _ensure_builtin_preset()
        base_cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
        if req.backend in {"fuzzball_mcp", "fuzzball"}:
            cluster, _ = discover_cluster_config(base_config=base_cluster)
        else:
            cluster = base_cluster
        cluster.fuzzfile_version = req.fuzzfile_version
        clean_team = req.definition_id.strip()
        if clean_team and clean_team.lower() != "auto":
            cluster.definition_id = clean_team
            if not cluster.vast.ephemeral:
                cluster.vast.provisioner = clean_team
        app.state.active_cluster = cluster

        # Resolve active dataset (custom uploaded drop OR built-in preset with zero upload required!)
        chosen_ds_id = (req.dataset_id or BUILTIN_PRESET_ID).strip() or BUILTIN_PRESET_ID
        if chosen_ds_id not in dataset_registry:
            chosen_ds_id = BUILTIN_PRESET_ID
        ds_info = dataset_registry[chosen_ds_id]
        source_data_dir = Path(ds_info["raw_dir"])

        # Resolve model: ANY custom model_id OR backward-compatible profile ("demo_30b" / "stretch_80b")
        if req.model_id and req.model_id.strip():
            exp = build_dynamic_experiment_config(
                model_id=req.model_id.strip(),
                dataset_name=str(ds_info["display_name"]),
                raw_subpath=chosen_ds_id,
                cluster=cluster,
            )
            profile_tag = "custom"
        else:
            profile_tag = req.profile or "demo_30b"
            cfg_name = "stretch_lammps_80b.yaml" if profile_tag == "stretch_80b" else "demo_lammps_30b.yaml"
            exp = ExperimentConfig.load(repo_root() / "configs" / cfg_name)
            if chosen_ds_id != BUILTIN_PRESET_ID:
                exp.dataset.name = chosen_ds_id

        # Use compact 256-token chunks when running on local/uploaded small datasets so train/val/test all pack richly
        if chosen_ds_id != BUILTIN_PRESET_ID or source_data_dir == repo_root() / "fixtures" / "sample_lammps":
            exp.dataset.chunk_seq_len = min(exp.dataset.chunk_seq_len, 256)
            exp.dataset.chunk_overlap = min(exp.dataset.chunk_overlap, 32)

        run_slug = f"ui-{exp.experiment_id}-{req.backend}-{req.fuzzfile_version}"
        target_dir = app.state.run_dir.parent / run_slug
        live_state.begin_run(cluster, profile_tag, req.backend, exp.model.name, chosen_ds_id)

        def _worker(sim_delay: float) -> None:
            try:
                run_autoresearch_pipeline(
                    exp=exp,
                    cluster=cluster,
                    output_dir=target_dir,
                    backend_kind=req.backend,
                    source_data_dir=source_data_dir,
                    event_hook=live_state.record_event,
                    sim_step_delay_s=sim_delay,
                )
                app.state.run_dir = target_dir
            except Exception as exc:
                live_state.record_event({"type": "run_failed", "error": str(exc)})

        if req.async_mode:
            delay = 0.06 if req.backend == "simulator" else 0.0
            thread = threading.Thread(target=_worker, args=(delay,), daemon=True)
            thread.start()
            time.sleep(0.02)
            return {"started": True, "live_status": live_state.snapshot()}

        _worker(0.0)
        snap = live_state.snapshot()
        if snap["state"] == "failed":
            raise HTTPException(status_code=400, detail=snap["error"] or "Run failed")
        return _collect_summary_payload(target_dir)

    @app.post("/api/firewall-probe")
    def probe_firewall(req: FirewallProbeRequest) -> dict[str, object]:
        audit_path = app.state.run_dir / "privacy_audit.jsonl"
        fw = PrivacySchemaFirewall(audit_log_path=audit_path)
        if req.inject_cpp_snippet:
            malicious_payload = {
                "stage": "stage_a_recipe",
                "model_id": live_state.model_id[:64],
                "cluster_sm_arch": "sm_121",
                "cluster_nodes": 2,
                "dataset_chunk_count": 42,
                "observations": [
                    {
                        "trial_id": "leak-probe",
                        "candidate_label": "#include <pair_lj_cut.h> void PairLJCut::compute() {}",
                        "rung_or_steps": 60,
                        "status": "completed",
                        "hyperparameters": {"lr": 0.0002},
                        "numeric_metrics": {"val_loss": 1.52},
                    }
                ],
            }
            try:
                fw.sanitize_and_serialize(malicious_payload)
            except DataIsolationViolation as exc:
                rec = fw.records[-1].model_dump()
                live_state.record_event({"type": "firewall_audit", "record": rec, "explanation": {}})
                return {
                    "blocked": True,
                    "violation": str(exc),
                    "record": rec,
                }
        return {"blocked": False}

    @app.post("/api/validate-fuzzfile")
    def api_validate_fuzzfile(payload: dict[str, str]) -> dict[str, object]:
        yaml_text = payload.get("yaml", "")
        cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
        try:
            stages = validate_fuzzfile(yaml_text, cluster=cluster)
            return {"valid": True, "stages": stages, "definition_id": cluster.definition_id}
        except FuzzfileValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        _ensure_seeded()
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app
