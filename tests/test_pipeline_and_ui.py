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
    import pytest

    pytest.importorskip("torch")
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
    assert discover_res.json()["cluster_config"]["definition_id"] == "team3"

    live_res = client.get("/api/live-status")
    assert live_res.status_code == 200
    live_json = live_res.json()
    assert live_json["definition_id"] == "team3"
    assert len(live_json["nodes"]) == 2

    # 1. Built-in preset dataset active out-of-the-box with Train / Val / Held-Out Test splits
    ds_list = client.get("/api/datasets").json()
    assert ds_list["default_dataset_id"] == "builtin-lammps-cpp-bench"
    assert ds_list["preset"]["split_doc_counts"]["train"] >= 1
    assert ds_list["preset"]["split_doc_counts"]["val"] >= 1
    assert ds_list["preset"]["split_doc_counts"]["held_out"] >= 1

    # 2. Custom file drop upload (JSON + code, zero python-multipart needed!)
    upload_res = client.post(
        "/api/dataset/upload",
        json={
            "dataset_name": "acme-sim-drop",
            "files": [
                {
                    "name": "kernels/integrator.py",
                    "content": "\n".join(
                        [f"def step_{i}(x, v, dt):\n    return x + v * dt * {i + 1}\n" for i in range(12)]
                    ),
                },
                {
                    "name": "tasks/tuning.jsonl",
                    "content": "\n".join(
                        [
                            f'{{"instruction": "Write CUDA helper {i}", "output": "__device__ float fn_{i}(float x) {{ return x * {i}.0f; }}"}}'
                            for i in range(10)
                        ]
                    ),
                },
            ],
        },
    )
    assert upload_res.status_code == 200, upload_res.text
    uploaded_ds = upload_res.json()
    assert uploaded_ds["dataset_id"].startswith("drop-")
    assert uploaded_ds["split_chunk_counts"]["train"] >= 1
    assert uploaded_ds["split_chunk_counts"]["val"] >= 1
    assert uploaded_ds["split_chunk_counts"]["held_out"] >= 1

    # 3. Optimize ANY custom HuggingFace model (Dense 14B + MoE Lite) on the dropped dataset
    for custom_model in (
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ):
        custom_run = client.post(
            "/api/run",
            json={
                "model_id": custom_model,
                "dataset_id": uploaded_ds["dataset_id"],
                "backend": "simulator",
                "fuzzfile_version": "v4",
                "definition_id": "team3",
                "async_mode": False,
            },
        )
        assert custom_run.status_code == 200, custom_run.text
        run_payload = custom_run.json()
        assert run_payload["model_profile"]["name"] == custom_model
        assert run_payload["cluster"]["vast"]["provisioner"] == "team3"

    # 4. Download complete self-contained `.zip` build bundle and verify every artifact
    import io
    import zipfile

    bundle_resp = client.get("/api/build/bundle.zip")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(bundle_resp.content)) as zf:
        names = set(zf.namelist())
        expected_files = {
            "fuzzfile.yaml",
            "configs/train_recipe.yaml",
            "configs/vllm_serve.yaml",
            "scripts/launch_fuzzball.sh",
            "scripts/train_and_serve_local.py",
            "data/train.jsonl",
            "data/val.jsonl",
            "data/test.jsonl",
            "data/held_out.jsonl",
            "data/dataset_manifest.json",
            "BUILD_REPORT.md",
        }
        assert expected_files.issubset(names)
        assert "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in zf.read("fuzzfile.yaml").decode()
        assert "team3" in zf.read("fuzzfile.yaml").decode()
        assert "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in zf.read("configs/train_recipe.yaml").decode()

    probe = client.post("/api/firewall-probe", json={"inject_cpp_snippet": True}).json()
    assert probe["blocked"] is True


def test_zero_torch_import_across_control_plane(tmp_path: Path) -> None:
    """Guarantees thin control-plane laptops without PyTorch run CLI, discovery, and simulator cleanly."""
    import subprocess
    import sys

    code = (
        "import builtins, pathlib\n"
        "real_import = builtins.__import__\n"
        "def block_torch(name, *a, **kw):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'torch'\")\n"
        "    return real_import(name, *a, **kw)\n"
        "builtins.__import__ = block_torch\n"
        "from crucible.cli import main\n"
        "assert main(['discover']) == 0\n"
        "assert main(['bench-kernels']) == 0\n"
        f"assert main(['run', '--backend', 'simulator', '--output-dir', '{tmp_path / 'notorch'}']) == 0\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Failed without torch:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
