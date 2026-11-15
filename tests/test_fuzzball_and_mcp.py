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
            {
                "name": "context_show",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "cluster_list",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "node_list",
                "inputSchema": {
                    "type": "object",
                    "properties": {"fields": {"type": "array", "items": {"type": "string"}}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "volume_list",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "workflow_list",
                "inputSchema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "workflow_validate",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fuzzfile": {"type": "string", "description": "Fuzzfile YAML string or path"},
                        "cluster_id": {"type": "string"},
                    },
                    "required": ["fuzzfile"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "workflow_describe",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "include_spec": {"type": "boolean"},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "workflow_start",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fuzzfile": {"type": "string"},
                        "name": {"type": "string"},
                        "confirm_endpoint_takeover": {"type": "boolean"},
                    },
                    "required": ["fuzzfile"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "workflow_wait",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "timeout": {"type": "string", "description": "Go duration e.g. 7200s"},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "workflow_log",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "job": {"type": "string"},
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "object_get",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "max_inline_bytes": {"type": "integer"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            },
        ]
        SCHEMAS = {t["name"]: t["inputSchema"] for t in TOOLS}

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
                schema = SCHEMAS.get(tname, {})
                allowed_props = set((schema.get("properties") or {}).keys())
                unexpected = sorted(set(targs.keys()) - allowed_props)
                if unexpected:
                    err_txt = f'validating "arguments": validating root: unexpected additional properties {json.dumps(unexpected)}'
                    res = {"content": [{"type": "text", "text": err_txt}], "isError": True}
                    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}) + "\\n")
                    sys.stdout.flush()
                    continue

                if tname == "context_show":
                    payload = {"user": "team3@fb-hack.local", "group": "team3@fb-hack.local"}
                elif tname == "cluster_list":
                    payload = {"clusters": [{"name": "ciq-gb10-team3"}]}
                elif tname == "volume_list":
                    # Include org-wide/foreign "datasets" volumes to verify Crucible NEVER pairs them with `use: team3`!
                    payload = {
                        "volumes": [
                            {"name": "datasets"},
                            {"name": "datasets", "provisioner": "shared"},
                            {"name": "runs", "provisioner": "team3", "mount_path": "/vast"},
                        ]
                    }
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
                                    "annotations": {},
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
                                    "annotations": {},
                                },
                            },
                        ]
                    }
                elif tname == "workflow_list":
                    payload = {"workflows": [{"id": "wf-fetch-01", "name": "crucible-00-fetch", "status": "finished"}]}
                elif tname == "workflow_validate":
                    assert isinstance(targs.get("fuzzfile"), str) and "version:" in targs["fuzzfile"]
                    payload = {"valid": True, "errors": [], "warnings": []}
                elif tname == "workflow_describe":
                    payload = {"workflowId": targs.get("id"), "node": "CIQ-CHI-SPARK39 (192.168.99.48/7331)"}
                elif tname == "workflow_start":
                    assert isinstance(targs.get("fuzzfile"), str)
                    payload = {"workflowId": "11111111-2222-3333-4444-555555555555"}
                elif tname == "workflow_wait":
                    assert isinstance(targs.get("timeout"), str) and targs["timeout"].endswith("s")
                    payload = {"workflowId": targs.get("id"), "status": "STAGE_STATUS_FINISHED"}
                elif tname == "workflow_log":
                    job_nm = targs.get("job", "")
                    if job_nm.startswith("bench-"):
                        s_obj = {
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
                        s_obj = {
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
                    payload = "CRUCIBLE_SUMMARY_JSON=" + json.dumps(s_obj)
                    res = {"content": [{"type": "text", "text": payload}]}
                    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": res}) + "\\n")
                    sys.stdout.flush()
                    continue
                elif tname == "object_get":
                    uri = targs.get("ref", "")
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


def test_team3_fuzzfile_pinning_devices_and_egress_isolation() -> None:
    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    assert cluster.definition_id == "team3"
    assert cluster.vast.provisioner == "team3"

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
    assert "annotations" not in parsed
    assert parsed["volumes"]["vast"] == {"name": "runs", "use": "team3"}
    assert parsed["jobs"]["train-t01"]["mounts"] == {"/vast": {"volume": "vast"}}
    res = parsed["jobs"]["train-t01"]["resource"]
    assert res == {
        "cpu": {"cores": 8},
        "memory": {"size": "96GiB"},
        "devices": {"nvidia.com/gpu": 1},
    }

    # Opt-in object-cache egress still works when enabled, while raw data egress is blocked
    cluster_egress = cluster.model_copy(deep=True)
    cluster_egress.vast.egress_metrics_to_object_cache = True
    parsed_eg = yaml.safe_load(
        render_training_trial_fuzzfile(
            cluster=cluster_egress,
            exp=exp,
            trial_id="t01",
            recipe=exp.stage_a_recipe.candidates[0],
            perf=exp.stage_b_training_perf.candidates[0],
            max_steps=60,
            run_id="run-01",
        )
    )
    egress = parsed_eg["volumes"]["vast"]["egress"]
    assert egress[0]["destination"]["uri"] == "fuzzball://user/crucible/run-01/t01/metrics.json"

    parsed_eg["volumes"]["vast"]["egress"].append(
        {
            "source": {"uri": "file://data/lammps/src/pair_lj_cut.cpp"},
            "destination": {"uri": "fuzzball://user/leak.txt"},
        }
    )
    with pytest.raises(FuzzfileValidationError, match="Data isolation guard"):
        validate_fuzzfile(parsed_eg, cluster=cluster)


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

    # Start from a mismatched `team20` config and verify `context_show` (`team3@fb-hack.local`) auto-switches to `team3`!
    wrong_base = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    wrong_base.definition_id = "team20"
    wrong_base.vast.provisioner = "team20"

    discovered_cfg, report = discover_cluster_config(base_config=wrong_base, mcp_command=mcp_cmd)
    assert report.mcp_source_used == "fuzzball_mcp_stdio"
    assert discovered_cfg.definition_id == "team3"
    assert discovered_cfg.vast.provisioner == "team3"
    assert discovered_cfg.nodes.arch_device == ""
    assert [n.hostname for n in discovered_cfg.nodes.inventory] == [
        "CIQ-CHI-SPARK39",
        "CIQ-CHI-SPARK40",
    ]
    assert discovered_cfg.vast.persistent_volume_name == "runs"
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


def test_discovery_never_pairs_unscoped_datasets_volume_with_team3() -> None:
    from crucible.fuzzball.discovery import ClusterDiscoveryReport, apply_mcp_discovery_payload

    cfg = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    cfg_data = cfg.model_dump()
    rep = ClusterDiscoveryReport()
    # Even if MCP `volume_list` returns ONLY org-wide `datasets` without `provisioner: team3`,
    # `persistent_volume_name` must stay `runs` (never `datasets`!).
    apply_mcp_discovery_payload(
        {
            "definition_id": "team3",
            "volumes": [
                {"name": "datasets"},
                {"name": "datasets", "provisioner": "public"},
            ],
        },
        cfg_data,
        rep,
        "fuzzball_mcp_stdio",
    )
    assert cfg_data["vast"]["provisioner"] == "team3"
    assert cfg_data["vast"]["persistent_volume_name"] == "runs"


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


def test_self_contained_remote_bootstrap_executes_without_crucible_installed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import subprocess

    from crucible.fuzzball.remote_bootstrap import REMOTE_SERVE_BENCH_PY, REMOTE_TRAIN_RUNNER_PY

    train_py = tmp_path / "remote_train.py"
    train_py.write_text(REMOTE_TRAIN_RUNNER_PY, encoding="utf-8")
    out_dir = tmp_path / "remote_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(train_py),
            "--trial-id",
            "remote-t01",
            "--base-model-dir",
            str(tmp_path / "missing_model"),
            "--processed-data-dir",
            str(tmp_path / "missing_data"),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "6",
            "--recipe-json",
            '{"lr":0.0002,"lora_rank":16,"lora_alpha":32,"target_modules":["q_proj","v_proj"],"seq_len":2048}',
            "--perf-json",
            '{"label":"sm121","nodes":1,"micro_batch_size":2,"grad_accum_steps":1,"activation_checkpointing":false,"use_chunked_ce_kernel":true,"use_grouped_moe_kernel":true,"use_fused_lora_kernel":true}',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "CRUCIBLE_SUMMARY_JSON=" in proc.stdout
    assert (out_dir / "summary.json").exists()

    srv_py = tmp_path / "remote_srv.py"
    srv_py.write_text(REMOTE_SERVE_BENCH_PY, encoding="utf-8")
    srv_json = tmp_path / "srv_summary.json"
    sproc = subprocess.run(
        [
            sys.executable,
            str(srv_py),
            "--endpoint",
            "http://127.0.0.1:9",
            "--processed-data-dir",
            str(tmp_path / "missing_data"),
            "--trial-id",
            "remote-s01",
            "--baseline-loss",
            "1.40",
            "--output-json",
            str(srv_json),
            "--candidate-json",
            '{"label":"nvfp4-2replicas-tp1","precision":"NVFP4","tensor_parallel":1,"replicas":2,"max_num_seqs":96,"use_low_prec_decode_kernel":true}',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sproc.returncode == 0, sproc.stderr
    assert srv_json.exists()


def test_fuzzball_v43_mcp_stage_inspection_multinode_ssh_quoting_and_team3_volumes(tmp_path: Path) -> None:
    """Proves Fuzzball v4.3.1 MCP stage shapes, exitCode error detection, 2-node SSH quoting,
    and team3 shared volumes (`models`, `data`, `envs`, `runs`) work cleanly on real cluster payloads."""
    import subprocess

    from crucible.fuzzball.discovery import ClusterDiscoveryReport, apply_mcp_discovery_payload
    from crucible.fuzzball.runner import inspect_workflow_stages

    # 1. Fuzzball v4.3.1 top-level `stages[]` while workflow top-level `status` is `"started"`
    queued_desc = {
        "id": "wf-1",
        "status": "started",
        "stages": [
            {"name": "vast", "kind": "volume", "status": "finished"},
            {"name": "train-t01", "kind": "job", "status": "pending"},
        ],
    }
    assert inspect_workflow_stages(queued_desc)["phase"] == "QUEUED"

    running_desc = {
        "id": "wf-1",
        "status": "started",
        "stages": [
            {"name": "vast", "kind": "volume", "status": "finished"},
            {"name": "train-t01", "kind": "job", "status": "started"},
        ],
    }
    assert inspect_workflow_stages(running_desc)["phase"] == "RUNNING"

    failed_exit_desc = {
        "id": "wf-1",
        "status": "finished",
        "stages": [
            {"name": "train-t01", "kind": "job", "status": "finished", "exitCode": 1},
        ],
    }
    st_err = inspect_workflow_stages(failed_exit_desc)
    assert st_err["terminal"] is True
    assert st_err["phase"] == "ERRORED"

    # 2. Team 3 shared volumes (`models`, `data`, `runs`, `envs`) + camelCase `hostName`/`memoryGb`
    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    cfg_data = cluster.model_dump()
    rep = ClusterDiscoveryReport()
    apply_mcp_discovery_payload(
        {
            "definition_id": "team3",
            "volumes": [
                {"name": "runs", "provisioner": "team3"},
                {"name": "models", "provisioner": "team3"},
                {"name": "data", "provisioner": "team3"},
                {"name": "envs", "provisioner": "team3"},
            ],
            "nodes": [
                {"id": "192.168.99.48/7331", "hostName": "CIQ-CHI-SPARK39", "cores": 20, "memoryGb": 127.9, "status": "READY"},
                {"id": "192.168.99.49/7331", "hostName": "CIQ-CHI-SPARK40", "cores": 20, "memoryGb": 127.9, "status": "READY"},
            ],
        },
        cfg_data,
        rep,
        "fuzzball_mcp_stdio",
    )
    resolved_cluster = ClusterConfig.model_validate(cfg_data)
    assert resolved_cluster.vast.persistent_volume_name == "runs"
    assert resolved_cluster.vast.extra_volumes == ["data", "envs", "models"]
    assert [n.hostname for n in resolved_cluster.nodes.inventory] == ["CIQ-CHI-SPARK39", "CIQ-CHI-SPARK40"]

    # 3. Render 2-node multinode Fuzzfile and execute its MULTINODE_SSH_WRAPPER command in bash
    exp = ExperimentConfig.load(repo_root() / "configs" / "demo_lammps_30b.yaml")
    perf_2n = next(c for c in exp.stage_b_training_perf.candidates if c.nodes == 2)
    yml_2n = render_training_trial_fuzzfile(
        cluster=resolved_cluster,
        exp=exp,
        trial_id="dp2-live",
        recipe=exp.stage_a_recipe.candidates[0],
        perf=perf_2n,
        max_steps=20,
        run_id="run-live",
    )
    validate_fuzzfile(yml_2n, cluster=resolved_cluster)
    parsed_2n = yaml.safe_load(yml_2n)
    assert set(parsed_2n["volumes"].keys()) == {"vast", "data", "envs", "models"}
    assert set(parsed_2n["jobs"]["train-dp2-live"]["mounts"].keys()) == {"/vast", "/data", "/envs", "/models"}

    # Verify bash syntax (`bash -n`) of the rendered 2-node launcher script has zero quoting issues
    script_2n = parsed_2n["jobs"]["train-dp2-live"]["script"]
    script_file = tmp_path / "dp2_launch.sh"
    script_file.write_text(script_2n, encoding="utf-8")
    syntax_check = subprocess.run(["bash", "-n", str(script_file)], capture_output=True, text=True, check=False)
    assert syntax_check.returncode == 0, syntax_check.stderr


def test_live_mid_run_curve_equivalence_stage_waterfall_and_nvidia_smi_streaming() -> None:
    """Verifies that LiveRunState updates loss curves, Max Delta equivalence, Stage B waterfall,
    Stage C table, and physical nvidia-smi GPU sensors mid-run BEFORE run_completed."""
    from crucible.server.app import LiveRunState

    cluster = ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    live = LiveRunState(cluster)
    live.begin_run(cluster, "custom", "fuzzball_mcp", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "builtin-lammps-cpp-bench")

    # Reference Stage B baseline finishes
    live.record_event(
        {
            "type": "trial_finished",
            "trial_id": "stageb-00-pytorch-baseline-1node",
            "kind": "train",
            "workflow_id": "wf-b0",
            "nodes_assigned": ["gb10-node-01"],
            "summary": {
                "status": "completed",
                "mean_tokens_per_sec": 675.0,
                "held_out_perplexity": 3.12,
                "final_val_loss": 1.14,
                "peak_mem_gb": 62.0,
                "hardware_utilization_pct": 42.0,
                "world_size": 1,
                "loss_curve": [1.80, 1.50, 1.30, 1.18],
            },
        }
    )

    # Next Stage B candidate streams mid-run step 3/4 from real Fuzzball GB10 GPU with nvidia-smi sensors
    live.record_event(
        {
            "type": "trial_step",
            "trial_id": "stageb-01-sm121-all-kernels-2node-ddp",
            "step": 3,
            "max_steps": 4,
            "train_loss": 1.306,
            "val_loss": 1.321,
            "tokens_per_sec": 2006.0,
            "allreduce_ms": 53.3,
            "inter_gpu_gbps": 48.2,
            "payload_mb": 32.0,
            "peak_mem_gb": 64.9,
            "world_size": 2,
            "nodes_assigned": ["gb10-node-01", "gb10-node-02"],
            "partial_curve": [1.802, 1.505, 1.306],
            "partial_val_curve": [1.820, 1.522, 1.321],
            "gpu_util_pct": 91.0,
            "gpu_temp_c": 63.0,
            "gpu_power_w": 118.0,
        }
    )

    snap = live.snapshot()
    assert snap["state"] == "running"
    assert len(snap["live_stage_b_outcomes"]) == 1
    assert snap["live_curve_equivalence"]["reference_trial_id"] == "stageb-00-pytorch-baseline-1node"
    assert snap["live_curve_equivalence"]["candidate_trial_id"] == "stageb-01-sm121-all-kernels-2node-ddp"
    assert 0.0 < snap["live_curve_equivalence"]["max_delta"] <= 0.01
    assert snap["live_curve_equivalence"]["accepted"] is True
    assert snap["nodes"][0]["telemetry_source"] == "nvidia-smi"
    assert snap["nodes"][0]["sm_util_pct"] == 91.0
    assert snap["nodes"][0]["temp_c"] == 63
    assert snap["nodes"][0]["power_w"] == 118


