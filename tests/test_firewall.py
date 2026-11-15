"""Tests proving the Privacy Schema Firewall blocks any raw C++ code, file paths, or dataset text."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.firewall import (
    AdvisorStageRequest,
    AdvisorTrialObservation,
    DataIsolationViolation,
    PrivacySchemaFirewall,
)


def test_firewall_allows_clean_numeric_and_config_payload(tmp_path: Path) -> None:
    audit_file = tmp_path / "privacy_audit.jsonl"
    fw = PrivacySchemaFirewall(audit_log_path=audit_file)
    req = AdvisorStageRequest(
        stage="stage_a_recipe",
        model_id="Qwen/Qwen3-Coder-30B-A3B",
        cluster_sm_arch="sm_121",
        cluster_nodes=2,
        dataset_chunk_count=128,
        observations=[
            AdvisorTrialObservation(
                trial_id="stagea-r0-c1",
                candidate_label="r32-a64-lr1e-04-7m",
                rung_or_steps=60,
                status="completed",
                hyperparameters={
                    "lr": 0.00015,
                    "lora_rank": 32,
                    "target_modules": ["q_proj", "v_proj"],
                },
                numeric_metrics={"val_loss": 1.48, "held_out_loss": 1.49, "tokens_per_sec": 1520.0},
            )
        ],
    )
    serialized = fw.sanitize_and_serialize(req)
    assert "r32-a64-lr1e-04-7m" in serialized
    assert len(fw.records) == 1
    assert fw.records[0].status == "ALLOWED"
    assert fw.records[0].contains_raw_data is False
    assert audit_file.exists()


@pytest.mark.parametrize(
    "bad_label",
    [
        "#include <pair_lj_cut.h>",
        "namespace LAMMPS_NS",
        "src/pair_lj_cut.cpp",
        "std::vector<double> force;",
        "/vast/data/lammps/pair.h",
        "multiline\ncode_snippet",
        "x" * 80,
    ],
)
def test_firewall_blocks_code_and_data_markers(tmp_path: Path, bad_label: str) -> None:
    audit_file = tmp_path / "privacy_audit.jsonl"
    fw = PrivacySchemaFirewall(audit_log_path=audit_file)
    payload = {
        "stage": "stage_a_recipe",
        "model_id": "Qwen/Qwen3-Coder-30B-A3B",
        "cluster_sm_arch": "sm_121",
        "cluster_nodes": 2,
        "dataset_chunk_count": 10,
        "observations": [
            {
                "trial_id": "t0",
                "candidate_label": bad_label,
                "rung_or_steps": 10,
                "status": "completed",
                "hyperparameters": {"lr": 0.0002},
                "numeric_metrics": {"val_loss": 1.5},
            }
        ],
    }
    with pytest.raises(DataIsolationViolation):
        fw.sanitize_and_serialize(payload)
    assert fw.records[-1].status == "BLOCKED"
    assert fw.records[-1].contains_raw_data is True


def test_firewall_blocks_unapproved_extra_fields(tmp_path: Path) -> None:
    fw = PrivacySchemaFirewall(audit_log_path=tmp_path / "audit.jsonl")
    payload = {
        "stage": "stage_a_recipe",
        "model_id": "Qwen/Qwen3-Coder-30B-A3B",
        "cluster_sm_arch": "sm_121",
        "cluster_nodes": 2,
        "dataset_chunk_count": 10,
        "raw_dataset_sample": "double evdwl = 0.0;",
        "observations": [],
    }
    with pytest.raises(DataIsolationViolation):
        fw.sanitize_and_serialize(payload)
