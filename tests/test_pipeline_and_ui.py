"""End-to-end tests for the 4-stage pipeline, local PyTorch trainer, CLI, and FastAPI studio."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from crucible.cli import main as cli_main
from crucible.config import (
    ClusterConfig,
    DatasetSplitConfig,
    ExperimentConfig,
    RecipeCandidate,
    TrainingPerfCandidate,
    repo_root,
)
from crucible.data.lammps_ingest import ingest_lammps_corpus
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.optimizer.pipeline import run_autoresearch_pipeline
from crucible.server.app import create_app
from crucible.trainer.lora_trainer import run_micro_torch_training


def test_end_to_end_pipeline_simulator_and_cli(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo_run"
    rc = cli_main(
        [
            "run",
            "--config",
            str(repo_root() / "configs" / "demo_lammps_30b.yaml"),
            "--backend",
            "simulator",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "final-crucible-workflow.yaml").exists()
    assert (out_dir / "run_report.md").exists()
    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "privacy_audit.jsonl").exists()

    assert validate_fuzzfile(out_dir / "final-crucible-workflow.yaml") == [
        "eval-held-out",
        "ingest-lammps",
        "serve-vllm",
        "train-lora",
    ]
    assert cli_main(["inspect", str(out_dir)]) == 0
    assert cli_main(["validate-yaml", str(out_dir / "final-crucible-workflow.yaml")]) == 0
    assert (
        cli_main(
            [
                "discover",
                "--mcp-snapshot",
                str(repo_root() / "configs" / "fuzzball_mcp_snapshot.example.json"),
                "--write-config",
                str(tmp_path / "discovered.yaml"),
            ]
        )
        == 0
    )
    assert (tmp_path / "discovered.yaml").exists()


def test_micro_torch_trainer_runs_real_pytorch_loop(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    ingest_lammps_corpus(
        repo_root() / "fixtures" / "sample_lammps",
        packed_dir,
        DatasetSplitConfig(chunk_seq_len=128, chunk_overlap=16),
        vocab_size=2048,
    )
    recipe = RecipeCandidate(
        lr=3e-3,
        lora_rank=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        seq_len=128,
    )
    perf = TrainingPerfCandidate(
        label="sm121-micro",
        nodes=1,
        micro_batch_size=2,
        grad_accum_steps=1,
        activation_checkpointing=False,
        use_chunked_ce_kernel=True,
        use_grouped_moe_kernel=True,
        use_fused_lora_kernel=True,
    )
    summary = run_micro_torch_training(
        trial_id="micro-01",
        processed_data_dir=packed_dir,
        output_dir=tmp_path / "trial_out",
        recipe=recipe,
        perf=perf,
        max_steps=6,
    )
    assert summary.status == "completed"
    assert summary.steps_completed == 6
    assert summary.final_train_loss < summary.loss_curve[0] + 0.5
    assert (tmp_path / "trial_out" / "metrics.jsonl").exists()
    assert (tmp_path / "trial_out" / "adapter_model.pt").exists()


def test_stretch_80b_profile_pipeline(tmp_path: Path) -> None:
    exp = ExperimentConfig.load(repo_root() / "configs" / "stretch_lammps_80b.yaml")
    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    res = run_autoresearch_pipeline(exp, cluster, tmp_path / "stretch80b", backend_kind="simulator")
    assert res.stage_b.winning_candidate.nodes == 2
    assert validate_fuzzfile(res.workflow_path) == [
        "eval-held-out",
        "ingest-lammps",
        "serve-vllm",
        "train-lora",
    ]


def test_fastapi_dashboard_endpoints_and_firewall_probe(tmp_path: Path) -> None:
    app = create_app(run_dir=tmp_path / "ui_state")
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    summary = client.get("/api/summary").json()
    assert summary["stage_a"]["winning_label"]
    assert summary["stage_b"]["overall_speedup_x"] > 1.0
    assert summary["stage_c"]["serving_speedup_x"] > 1.0

    discover_res = client.get("/api/discover")
    assert discover_res.status_code == 200
    assert discover_res.json()["cluster_config"]["fuzzfile_version"] == "v4"

    probe = client.post("/api/firewall-probe", json={"inject_cpp_snippet": True}).json()
    assert probe["blocked"] is True
