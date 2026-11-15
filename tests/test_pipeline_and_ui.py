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
from crucible.data.teacher_curriculum import is_allowed_private_teacher_url
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
    assert live_json["lifecycle_phase"] == "pre_run"
    assert live_json["output_ready"] is False

    # Private teacher URL guard: allow Fuzzball enclave / localhost, block public internet domains
    assert is_allowed_private_teacher_url("http://127.0.0.1:8001/v1") is True
    assert is_allowed_private_teacher_url("https://vllm.fb-hack.civilfritz.net/v1") is True
    assert is_allowed_private_teacher_url("https://api.openai.com/v1") is False

    # 1. Built-in preset dataset active out-of-the-box with Train / Val / Held-Out Test splits
    ds_list = client.get("/api/datasets").json()
    assert ds_list["default_dataset_id"] == "builtin-lammps-cpp-bench"
    assert ds_list["preset"]["split_doc_counts"]["train"] >= 1
    assert ds_list["preset"]["split_doc_counts"]["val"] >= 1
    assert ds_list["preset"]["split_doc_counts"]["held_out"] >= 1

    # 1b. Client-server architecture, 40+ vast model catalog, & remote Fuzzball connection probe
    arch = client.get("/api/architecture").json()
    assert arch["mode"] == "client_server_split"
    assert arch["fuzzball_server"]["endpoint"] == "api.fb-hack.civilfritz.net:443"
    assert arch["measured_reference"]["measured_bf16_gemm_tflops"] == 99.1

    models_all = client.get("/api/models").json()
    assert models_all["total_count"] >= 35
    assert "Frontier Multi-Node" in models_all["categories"]
    assert "Compact & Edge" in models_all["categories"]
    frontier_only = client.get("/api/models?category=Frontier%20Multi-Node").json()
    assert len(frontier_only["catalog"]) >= 4
    assert all(m["min_nodes_required"] == 2 for m in frontier_only["catalog"])

    conn_probe = client.post(
        "/api/cluster/connect",
        json={"cluster_endpoint": "api.fb-hack.civilfritz.net:443", "definition_id": "team3"},
    )
    assert conn_probe.status_code == 200
    assert conn_probe.json()["cluster_config"]["definition_id"] == "team3"

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
    tc_upload = uploaded_ds["teacher_curriculum"]
    assert tc_upload["zero_cross_split_leakage"] is True
    assert tc_upload["raw_cloud_egress_bytes"] == 0
    assert tc_upload["split_pair_counts"]["train"] >= 1
    assert tc_upload["split_pair_counts"]["val"] >= 1
    assert tc_upload["split_pair_counts"]["test"] >= 1

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

    bundle_resp = client.get("/api/build/bundle.zip?company_name=Apex%20Dynamics")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"] == "application/zip"
    assert "apex-dynamics-private-llm-" in bundle_resp.headers.get("content-disposition", "")
    with zipfile.ZipFile(io.BytesIO(bundle_resp.content)) as zf:
        names = set(zf.namelist())
        expected_files = {
            "fuzzfile.yaml",
            "Modelfile.ollama",
            "configs/train_recipe.yaml",
            "configs/vllm_serve.yaml",
            "scripts/run_company_app.py",
            "scripts/launch_fuzzball.sh",
            "scripts/train_and_serve_local.py",
            "scripts/client_query_llm.py",
            "data/train.jsonl",
            "data/val.jsonl",
            "data/test.jsonl",
            "data/held_out.jsonl",
            "data/dataset_manifest.json",
            "data/teacher_sft_train.jsonl",
            "data/teacher_sft_val.jsonl",
            "data/teacher_sft_test.jsonl",
            "data/teacher_curriculum_manifest.json",
            "BUILD_REPORT.md",
        }
        assert expected_files.issubset(names)
        assert zf.read("data/teacher_sft_train.jsonl").decode().strip() != ""
        assert zf.read("data/teacher_sft_val.jsonl").decode().strip() != ""
        assert zf.read("data/teacher_sft_test.jsonl").decode().strip() != ""
        assert "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in zf.read("fuzzfile.yaml").decode()
        assert "team3" in zf.read("fuzzfile.yaml").decode()
        assert "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in zf.read("configs/train_recipe.yaml").decode()
        assert "FROM deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in zf.read("Modelfile.ollama").decode()
        company_app_code = zf.read("scripts/run_company_app.py").decode()
        assert "Apex Dynamics" in company_app_code
        compile(company_app_code, "run_company_app.py", "exec")

    # Verify live dual-GPU telemetry, step diagnostics, & FINISHED output-ready lifecycle after run
    post_live = client.get("/api/live-status").json()
    assert post_live["lifecycle_phase"] == "finished"
    assert post_live["output_ready"] is True
    assert post_live["live_teacher_curriculum"]["total_teaching_pairs"] >= 3
    assert "gpu_telemetry" in post_live
    assert len(post_live["nodes"]) == 2
    assert post_live["live_step_metrics"]["tokens_per_sec"] > 0

    pg = client.post(
        "/api/playground/complete",
        json={"prompt": "Write SIMD neighbor loop", "compare_base": True},
    ).json()
    assert "PairLJCut" in pg["tuned_output"]
    assert pg["base_output"] is not None

    probe = client.post("/api/firewall-probe", json={"inject_cpp_snippet": True}).json()
    assert probe["blocked"] is True

    # 5. Verify all 39 catalog models are Pre-Loaded + custom HF IDs support 1-click VAST staging + Continuous Teacher budget
    m_cat = client.get("/api/models?probe_model=Qwen/Qwen2.5-Coder-1.5B-Instruct").json()
    assert m_cat["preloaded_count"] == m_cat["total_count"]
    assert m_cat["preset_count"] >= 10
    assert m_cat["inferred"]["has_hardware_preset"] is True
    assert m_cat["inferred"]["vast_state"] == "preloaded"

    m_custom_before = client.get("/api/models?probe_model=acme-org/Custom-Coder-9B-Instruct").json()
    assert m_custom_before["inferred"]["vast_state"] == "needs_load"

    stage_res = client.post(
        "/api/models/stage",
        json={"model_id": "acme-org/Custom-Coder-9B-Instruct", "backend": "simulator", "async_mode": False},
    )
    assert stage_res.status_code == 200, stage_res.text
    stage_json = stage_res.json()
    assert stage_json["staged"] is True
    assert Path(stage_json["fuzzfile_path"]).exists()

    m_after = client.get("/api/models?probe_model=acme-org/Custom-Coder-9B-Instruct").json()
    assert m_after["inferred"]["vast_state"] == "preloaded"
    assert m_after["inferred"]["preloaded_on_vast"] is True

    cont_run = client.post(
        "/api/run",
        json={
            "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "dataset_id": uploaded_ds["dataset_id"],
            "backend": "simulator",
            "continuous_mode": True,
            "time_budget_minutes": 5.0,
            "cost_budget_usd": 3.50,
            "use_company_presets": True,
            "async_mode": False,
        },
    )
    assert cont_run.status_code == 200, cont_run.text
    cont_payload = cont_run.json()
    cont_meta = cont_payload["continuous_optimization"]
    assert cont_meta["enabled"] is True
    assert cont_meta["rounds_completed"] >= 1
    assert cont_meta["preset_savings_usd"] == 1.45
    assert cont_meta["spent_usd"] <= cont_meta["cost_budget_usd"] + 0.5
    assert any(e["stage"].startswith("teacher_round_") for e in cont_payload["advisor_explanations"])


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
