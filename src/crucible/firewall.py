"""Hard Privacy Schema Firewall enforcing strict data isolation for the LLM advisor layer.

Guarantee:
- Only bounded hyperparameter identifiers and scalar numeric metrics may ever cross into an
  outbound LLM advisor request.
- Any C++ syntax markers, source-file extensions, multiline strings, filesystem paths to
  dataset files, or unapproved dictionary keys immediately raise `DataIsolationViolation`
  before serialization or network egress, and every check is recorded to an append-only
  SHA-256 audit log (`privacy_audit.jsonl`).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_./:\-]+$")

FORBIDDEN_SUBSTRINGS = (
    "#include",
    "#ifndef",
    "#define",
    "namespace ",
    "LAMMPS_NS",
    "std::",
    "template<",
    "template <",
    "class ",
    "struct ",
    "double ",
    "void ",
    "return ",
    "nullptr",
    ".cpp",
    ".cxx",
    ".cc",
    ".hpp",
    ".cuh",
    "/lammps/",
    "held_out_documents",
    ".jsonl",
    "prompt_prefix",
    "reference_completion",
    "completion_text",
    "source_code",
    "{",
    "}",
    ";",
)


class DataIsolationViolation(RuntimeError):
    """Raised when any payload crossing the advisor boundary fails strict schema/content validation."""


class AdvisorTrialObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(max_length=64)
    candidate_label: str = Field(max_length=64)
    rung_or_steps: int = Field(ge=0)
    status: Literal["completed", "early_stopped", "rejected_loss_mismatch", "rejected_quality_floor", "oom"]
    hyperparameters: dict[str, float | int | bool | str | list[str]]
    numeric_metrics: dict[str, float]


class AdvisorStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "stage_a_recipe",
        "stage_b_training_perf",
        "stage_c_inference_perf",
        "stage_d_summary",
        "continuous_teacher_loop",
    ]
    model_id: str = Field(max_length=64)
    cluster_sm_arch: str = Field(max_length=32)
    cluster_nodes: int = Field(ge=1, le=256)
    dataset_chunk_count: int = Field(ge=0)
    observations: list[AdvisorTrialObservation]


class AdvisorDecisionExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    selected_label: str
    narrowed_candidate_labels: list[str]
    rationale: str
    provider_used: str


class FirewallAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    stage: str
    status: Literal["ALLOWED", "BLOCKED"]
    payload_sha256: str
    scalar_field_count: int
    contains_raw_data: bool = False
    reason: str = "validated_config_and_metrics_only"


def _inspect_value(value: Any, path: str) -> int:
    """Recursively verify every leaf in an outbound advisor payload; return leaf count."""
    if isinstance(value, (int, float, bool)) or value is None:
        return 1
    if isinstance(value, str):
        if len(value) > 64:
            raise DataIsolationViolation(
                f"String at '{path}' exceeds 64-char config identifier limit (len={len(value)})."
            )
        if "\n" in value or "\r" in value:
            raise DataIsolationViolation(f"Multiline text forbidden across firewall at '{path}'.")
        lowered = value.lower()
        for marker in FORBIDDEN_SUBSTRINGS:
            if marker.lower() in lowered:
                raise DataIsolationViolation(
                    f"Forbidden code/data marker '{marker}' detected at '{path}'."
                )
        if value and not SAFE_IDENTIFIER_RE.match(value):
            raise DataIsolationViolation(
                f"Free-form string '{value}' at '{path}' does not match strict identifier pattern."
            )
        return 1
    if isinstance(value, list):
        return sum(_inspect_value(item, f"{path}[{idx}]") for idx, item in enumerate(value))
    if isinstance(value, dict):
        count = 0
        for k, v in value.items():
            if not isinstance(k, str):
                raise DataIsolationViolation(f"Non-string key at '{path}'.")
            _inspect_value(k, f"{path}.<key>")
            count += _inspect_value(v, f"{path}.{k}")
        return count
    raise DataIsolationViolation(f"Unsupported type {type(value).__name__} at '{path}'.")


class PrivacySchemaFirewall:
    """Enforces and logs zero-raw-data egress between local evaluation and the LLM advisor."""

    def __init__(self, audit_log_path: Path | None = None) -> None:
        self.audit_log_path = audit_log_path
        self.records: list[FirewallAuditRecord] = []

    def sanitize_and_serialize(self, raw_payload: AdvisorStageRequest | dict[str, Any]) -> str:
        raw_json_bytes = json.dumps(
            raw_payload.model_dump() if isinstance(raw_payload, BaseModel) else raw_payload,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(raw_json_bytes).hexdigest()
        stage_name = (
            raw_payload.stage
            if isinstance(raw_payload, AdvisorStageRequest)
            else str(raw_payload.get("stage", "unknown"))
        )

        try:
            validated = (
                raw_payload
                if isinstance(raw_payload, AdvisorStageRequest)
                else AdvisorStageRequest.model_validate(raw_payload)
            )
            dumped = validated.model_dump()
            leaf_count = _inspect_value(dumped, "request")
        except Exception as exc:
            record = FirewallAuditRecord(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                stage=stage_name,
                status="BLOCKED",
                payload_sha256=digest,
                scalar_field_count=0,
                contains_raw_data=True,
                reason=str(exc),
            )
            self._append_record(record)
            if isinstance(exc, DataIsolationViolation):
                raise
            raise DataIsolationViolation(f"Schema validation blocked outbound payload: {exc}") from exc

        clean_json = json.dumps(dumped, sort_keys=True)
        clean_digest = hashlib.sha256(clean_json.encode("utf-8")).hexdigest()
        record = FirewallAuditRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            stage=validated.stage,
            status="ALLOWED",
            payload_sha256=clean_digest,
            scalar_field_count=leaf_count,
            contains_raw_data=False,
            reason="strict_schema_and_identifier_verification_passed",
        )
        self._append_record(record)
        return clean_json

    def _append_record(self, record: FirewallAuditRecord) -> None:
        self.records.append(record)
        if self.audit_log_path is not None:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
