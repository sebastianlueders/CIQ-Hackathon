"""FastAPI application exposing Crucible run telemetry, live re-runs, and firewall verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from crucible.config import ClusterConfig, ExperimentConfig, repo_root
from crucible.firewall import DataIsolationViolation, PrivacySchemaFirewall
from crucible.fuzzball.discovery import discover_cluster_config
from crucible.fuzzball.validator import FuzzfileValidationError, validate_fuzzfile
from crucible.optimizer.pipeline import run_autoresearch_pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TriggerRunRequest(BaseModel):
    profile: Literal["demo_30b", "stretch_80b"] = "demo_30b"
    backend: Literal["simulator", "local_torch"] = "simulator"
    fuzzfile_version: Literal["v4", "v1"] = "v4"


class FirewallProbeRequest(BaseModel):
    inject_cpp_snippet: bool = True


def create_app(run_dir: Path | None = None) -> FastAPI:
    resolved_run_dir = (run_dir or (repo_root() / "runs" / "ui-default")).resolve()
    app = FastAPI(
        title="Crucible Autoresearch Studio",
        version="0.1.0",
        description="Local fine-tuning & sm_121 performance-engineering control plane on CIQ Fuzzball + VAST.",
    )
    app.state.run_dir = resolved_run_dir

    def _ensure_seeded() -> dict[str, object]:
        summary_file = app.state.run_dir / "run_summary.json"
        if not summary_file.exists():
            cfg_path = repo_root() / "configs" / "demo_lammps_30b.yaml"
            cluster_path = repo_root() / "configs" / "cluster.default.yaml"
            exp = ExperimentConfig.load(cfg_path)
            cluster = ClusterConfig.load(cluster_path)
            run_autoresearch_pipeline(
                exp=exp,
                cluster=cluster,
                output_dir=app.state.run_dir,
                backend_kind="simulator",
            )
        return json.loads(summary_file.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "crucible-studio"}

    @app.get("/api/discover")
    def api_discover() -> dict[str, object]:
        cfg, rep = discover_cluster_config()
        return {"cluster_config": cfg.model_dump(), "discovery_report": rep.to_dict()}

    @app.get("/api/summary")
    def get_summary() -> dict[str, object]:
        data = _ensure_seeded()
        # Attach loss curves from Stage B baseline & winner trial summaries for curve overlay chart
        curves: dict[str, list[float]] = {}
        trials_root = app.state.run_dir / "trials"
        if trials_root.exists():
            for trial_sub in sorted(trials_root.iterdir()):
                s_file = trial_sub / "summary.json"
                if s_file.exists():
                    s_obj = json.loads(s_file.read_text(encoding="utf-8"))
                    if s_obj.get("loss_curve"):
                        curves[trial_sub.name] = s_obj["loss_curve"]
        data["trial_loss_curves"] = curves
        return data

    @app.post("/api/run")
    def trigger_run(req: TriggerRunRequest) -> dict[str, object]:
        cfg_name = "demo_lammps_30b.yaml" if req.profile == "demo_30b" else "stretch_lammps_80b.yaml"
        exp = ExperimentConfig.load(repo_root() / "configs" / cfg_name)
        cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
        cluster.fuzzfile_version = req.fuzzfile_version
        target_dir = app.state.run_dir.parent / f"ui-{req.profile}-{req.fuzzfile_version}"
        run_autoresearch_pipeline(
            exp=exp,
            cluster=cluster,
            output_dir=target_dir,
            backend_kind=req.backend,
        )
        app.state.run_dir = target_dir
        return get_summary()

    @app.post("/api/firewall-probe")
    def probe_firewall(req: FirewallProbeRequest) -> dict[str, object]:
        audit_path = app.state.run_dir / "privacy_audit.jsonl"
        fw = PrivacySchemaFirewall(audit_log_path=audit_path)
        if req.inject_cpp_snippet:
            malicious_payload = {
                "stage": "stage_a_recipe",
                "model_id": "Qwen/Qwen3-Coder-30B-A3B",
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
                return {
                    "blocked": True,
                    "violation": str(exc),
                    "record": fw.records[-1].model_dump(),
                }
        return {"blocked": False}

    @app.post("/api/validate-fuzzfile")
    def api_validate_fuzzfile(payload: dict[str, str]) -> dict[str, object]:
        yaml_text = payload.get("yaml", "")
        try:
            stages = validate_fuzzfile(yaml_text)
            return {"valid": True, "stages": stages}
        except FuzzfileValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        _ensure_seeded()
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app
