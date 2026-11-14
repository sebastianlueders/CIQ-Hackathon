"""Tests for CIQ Fuzzball 4.2 MCP stdio client, Team 20 Fuzzfile constraints, and discovery."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from crucible.config import ClusterConfig, ExperimentConfig, repo_root
from crucible.fuzzball.discovery import discover_cluster_config
from crucible.fuzzball.mcp_client import FuzzballMCPStdioClient
from crucible.fuzzball.renderer import (
    render_final_turnkey_fuzzfile,
    render_training_trial_fuzzfile,
)
from crucible.fuzzball.runner import FuzzballMCPBackend
from crucible.fuzzball.validator import FuzzfileValidationError, validate_fuzzfile


def _write_fake_fuzzball_mcp_server(script_path: Path) -> Path:
    """Creates an executable mock `fuzzball mcp serve` JSON-RPC 2.0 stdio server returning Team 20 nodes."""
    code = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, sys

        TOOLS = [
            {"name": "cluster_list"},
            {"name": "node_list"},
            {"name": "volume_list"},
            {"name": "workflow_list"},
            {"name": "workflow_validate"},
            {"name": "workflow_describe"},
            {"name": "workflow_start"},
            {"name": "workflow_wait"},
            {"name": "object_get"},
        ]

        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            msg = json.loads(raw)
            req_id = msg.get("id")
            method = msg.get("method")
            if req_id is None:
                continue

            if method == "initialize":
                res = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "fuzzball-mcp", "version": "4.2.0"}}
            elif method == "tools/list":
                res = {"tools": TOOLS}
            elif method == "tools/call":
                params = msg.get("params") or {}
                tname = params.get("name")
                targs = params.get("arguments") or {}
                if tname == "cluster_list":
                    payload = {"clusters": [{"name": "ciq-gb10-team20"}]}
                elif tname == "volume_list":
                    payload = {"volumes": [{"name": "vast-live-vol", "provisioner": "vast-nvme", "mount_path": "/vast"}]}
                elif tname == "node_list":
                    payload = {
                        "nodes": [
                            {
                                "node": {"id": "192.168.99.48/7331", "hostname": "CIQ-CHI-SPARK39", "status": 2},
                                "resource": {
                                    "total": {
                                        "cpu": {"cores": 20},
                                        "mem": {"bytes": 137331589120},
                                        "devices": {"nvidia.com/gpu": 1, "cpu/arm64/sve": 1},
                                    },
                                    "annotations": {"fuzzball.io/provisioner_definition_id": "team20"},
                                },
                            },
                            {
                                "node": {"id": "192.168.99.49/7331", "hostname": "CIQ-CHI-SPARK40", "status": 2},
                                "resource": {
                                    "total": {
                                        "cpu": {"cores": 20},
                                        "mem": {"bytes": 137331589120},
                                        "devices": {"nvidia.com/gpu": 1, "cpu/arm64/sve": 1},
                                    },
                                    "annotations": {"fuzzball.io/provisioner_definition_id": "team20"},
                                },
                            },
                        ]
                    }
                elif tname == "workflow_list":
                    payload = {"workflows": [{"id": "wf-fetch-01", "name": "crucible-00-fetch", "status": "finished"}]}
                elif tname == "workflow_validate":
                    payload = {"valid": True, "errors": [], "warnings": []}
                elif tname == "workflow_describe":
                    payload = {"workflowId": targs.get("workflow_id"), "node": "CIQ-CHI-SPARK39 (192.168.99.48/7331)"}
                elif tname == "workflow_start":
                    payload = {"workflowId": "11111111-2222-3333-4444-555555555555"}
                elif tname == "workflow_wait":
                    payload = {"workflowId": targs.get("workflow_id"), "status": "STAGE_STATUS_FINISHED"}
                elif tname == "object_get":
                    uri = targs.get("uri", "")
                    if "serving-" in uri:
                        payload = {
                            "trial_id": "srv-mcp-01",
                            "label": "nvfp4-2replicas-tp1",
                            "precision": "NVFP4",
                            "tensor_parallel": 1,
                            "replicas": 2,
                            "max_num_seqs": 96,
                            "use_low_prec_decode_kernel": True,
                            "throughput_tokens_per_sec": 3460.3,
                            "ttft_ms": 44.0,
                            "p50_latency_ms": 140.0,
                            "p99_latency_ms": 410.0,
                            "held_out_loss": 1.41,
                            "held_out_perplexity": 4.096,
                            "perplexity_regression_pct": 1.42,
                            "completion_pass_rate": 0.88,
                            "passed_quality_floor": True,
                            "passed_latency_ceiling": True,
                            "accepted": True,
                            "rejection_reasons": [],
                        }
                    else:
                        payload = {
                            "trial_id": "train-mcp-01",
                            "status": "completed",
                            "steps_completed": 60,
                            "initial_val_loss": 1.92,
                            "final_train_loss": 1.42,
                            "final_val_loss": 1.44,
                            "held_out_eval_loss": 1.45,
                            "held_out_perplexity": 4.263,
                            "mean_tokens_per_sec": 3820.0,
                            "hardware_utilization_pct": 92.0,
                            "peak_mem_gb": 88.5,
                            "wall_time_s": 42.0,
                            "loss_curve": [1.92, 1.65, 1.42],
                            "world_size": 1,
                            "kernels_enabled": {
                                "chunked_cross_entropy": True,
                                "grouped_moe_gemm": True,
                                "fused_lora_linear": True,
                            },
                        }
                else:
                    payload = {"error": f"unknown tool {tname}"}
                res = {"content": [{"type": "text", "text": json.dumps(payload)}]}
            else:
                res = {}

            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}) + "\\n")
            sys.stdout.flush()
        """
    )
    script_path.write_text(code, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def test_team20_fuzzfile_pinning_devices_and_egress_isolation() -> None:
    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    assert cluster.definition_id == "team20"
    assert [n.hostname for n in cluster.nodes.inventory] == ["CIQ-CHI-SPARK39", "CIQ-CHI-SPARK40"]

    exp = ExperimentConfig.load(repo_root() / "configs" / "demo_lammps_30b.yaml")
    yml = render_training_trial_fuzzfile(
        cluster=cluster,
        exp=exp,
        trial_id="t01",
        recipe=exp.stage_a_recipe.candidates[0],
        perf=exp.stage_b_training_perf.candidates[0],
        max_steps=60,
        run_id="run-01",
    )
    stages = validate_fuzzfile(yml, cluster=cluster)
    assert stages == ["train-t01"]
    parsed = yaml.safe_load(yml)
    assert parsed["annotations"]["fuzzball.io/provisioner_definition_id"] == "team20"
    res = parsed["jobs"]["train-t01"]["resource"]
    assert res["cpu"]["cores"] == 16
    assert res["memory"]["size"] == "112GiB"
    assert res["devices"] == {"nvidia.com/gpu": 1, "cpu/arm64/sve": 1}

    egress = parsed["volumes"]["vast-shared"]["egress"]
    assert egress[0]["destination"]["uri"] == "fuzzball://user/crucible/run-01/t01/metrics.json"

    # Attempt to egress raw LAMMPS data -> validator MUST reject
    parsed["volumes"]["vast-shared"]["egress"].append(
        {
            "source": {"uri": "file://data/lammps/src/pair_lj_cut.cpp"},
            "destination": {"uri": "fuzzball://user/leak.txt"},
        }
    )
    with pytest.raises(FuzzfileValidationError, match="Data isolation guard"):
        validate_fuzzfile(parsed, cluster=cluster)


def test_validator_enforces_team20_resource_bounds_and_reserved_annotations() -> None:
    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")

    # Exceeds 20 physical cores per Spark node
    over_cores = {
        "version": "v4",
        "jobs": {
            "over": {
                "image": {"uri": "docker://nvcr.io/nvidia/pytorch:26.03-py3"},
                "command": ["echo", "hi"],
                "resource": {
                    "cpu": {"cores": 24},
                    "memory": {"size": "64GiB"},
                    "devices": {"nvidia.com/gpu": 1},
                },
            }
        },
    }
    with pytest.raises(FuzzfileValidationError, match="exceeding node physical limit 20"):
        validate_fuzzfile(over_cores, cluster=cluster)

    # Custom fuzzball.io/* annotation not on platform allowlist -> rejected
    bad_ann = {
        "version": "v4",
        "annotations": {"fuzzball.io/custom-team-tag": "team20"},
        "jobs": {
            "w": {
                "image": {"uri": "docker://nvcr.io/nvidia/pytorch:26.03-py3"},
                "command": ["echo", "hi"],
            }
        },
    }
    with pytest.raises(FuzzfileValidationError, match="reserved 'fuzzball.io/' prefix"):
        validate_fuzzfile(bad_ann, cluster=cluster)

    # Validate bundled repo workflows against Team 20 hardware bounds
    assert validate_fuzzfile(repo_root() / "workflows" / "00-fetch.yaml", cluster=cluster) == [
        "fetch-base-model",
        "fetch-lammps-source",
    ]
    assert validate_fuzzfile(repo_root() / "workflows" / "crucible-studio.yaml", cluster=cluster) == [
        "crucible-studio"
    ]


def test_validator_catches_duplicate_stage_names_and_embedded_secrets() -> None:
    bad_dup = {
        "version": "v4",
        "volumes": {"shared": {"use": "vast", "name": "v1"}},
        "jobs": {
            "worker": {
                "image": {"uri": "docker://alpine:latest"},
                "command": ["echo", "ok"],
            }
        },
        "services": {
            "worker": {
                "image": {"uri": "docker://alpine:latest"},
                "command": ["sleep", "10"],
            }
        },
    }
    with pytest.raises(FuzzfileValidationError, match="unique names"):
        validate_fuzzfile(bad_dup)

    bad_secret = {
        "version": "v4",
        "jobs": {
            "worker": {
                "image": {"uri": "docker://alpine:latest"},
                "command": ["echo", "ok"],
                "env": ["OPENAI_API_KEY=sk-hardcoded"],
            }
        },
    }
    with pytest.raises(FuzzfileValidationError, match="Hardcoded secret forbidden"):
        validate_fuzzfile(bad_secret)


def test_native_fuzzball_mcp_stdio_team20_discovery_and_live_events(tmp_path: Path) -> None:
    fake_server = _write_fake_fuzzball_mcp_server(tmp_path / "mock_fuzzball_mcp.py")
    mcp_cmd = [sys.executable, str(fake_server)]

    with FuzzballMCPStdioClient(command=mcp_cmd) as client:
        assert "workflow_validate" in client.advertised_tools
        assert "object_get" in client.advertised_tools

    discovered_cfg, report = discover_cluster_config(mcp_command=mcp_cmd)
    assert report.mcp_source_used == "fuzzball_mcp_stdio"
    assert discovered_cfg.definition_id == "team20"
    assert discovered_cfg.nodes.arch_device == "cpu/arm64/sve"
    assert [n.hostname for n in discovered_cfg.nodes.inventory] == [
        "CIQ-CHI-SPARK39",
        "CIQ-CHI-SPARK40",
    ]
    assert discovered_cfg.vast.persistent_volume_name == "vast-live-vol"
    assert discovered_cfg.nodes.count == 2

    events: list[dict] = []
    exp = ExperimentConfig.load(repo_root() / "configs" / "demo_lammps_30b.yaml")
    backend = FuzzballMCPBackend(
        cluster=discovered_cfg,
        exp=exp,
        run_id="mcp-test-run",
        local_run_dir=tmp_path / "mcp-run",
        mcp_command=mcp_cmd,
        event_hook=events.append,
    )
    train_summary = backend.run_training_trial(
        trial_id="train-mcp-01",
        recipe=exp.stage_a_recipe.candidates[0],
        perf=exp.stage_b_training_perf.candidates[3],
        max_steps=60,
    )
    assert train_summary.status == "completed"
    assert any(ev["type"] == "trial_started" and "CIQ-CHI-SPARK39" in ev["nodes_assigned"] for ev in events)
    assert any(ev["type"] == "trial_finished" for ev in events)

    srv_summary = backend.run_serving_trial(
        trial_id="srv-mcp-01",
        cand=exp.stage_c_inference_perf.candidates[3],
        baseline_finetuned_loss=1.45,
    )
    assert srv_summary.accepted is True
    assert srv_summary.precision == "NVFP4"


def test_turnkey_v4_and_v1_rendering() -> None:
    exp = ExperimentConfig.load(repo_root() / "configs" / "demo_lammps_30b.yaml")
    for ver in ("v4", "v1"):
        cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
        cluster.fuzzfile_version = ver
        yml = render_final_turnkey_fuzzfile(
            cluster=cluster,
            exp=exp,
            winning_recipe=exp.stage_a_recipe.candidates[1],
            winning_train_perf=exp.stage_b_training_perf.candidates[4],
            winning_serve_perf=exp.stage_c_inference_perf.candidates[3],
            metrics_summary={"base_perplexity": 6.85, "finetuned_perplexity": 4.04},
        )
        stages = validate_fuzzfile(yml, cluster=cluster)
        assert stages == ["eval-held-out", "ingest-lammps", "serve-vllm", "train-lora"]
