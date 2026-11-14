"""Structural and security validator for CIQ Fuzzball workflows (`v4` and `v1`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VALID_IMAGE_SCHEMES = ("docker://", "oras://")
VALID_DEPENDS_STATUS = frozenset({"RUNNING", "FINISHED", "FAILED"})
SUSPICIOUS_ENV_PREFIXES = ("OPENAI_API_KEY=", "ANTHROPIC_API_KEY=", "HF_TOKEN=", "PASSWORD=", "SECRET=")


class FuzzfileValidationError(ValueError):
    """Raised when a Fuzzball workflow violates CIQ Fuzzball schema or security invariants."""


def validate_fuzzfile(workflow_or_path: dict[str, Any] | str | Path) -> list[str]:
    if isinstance(workflow_or_path, Path):
        raw_text = workflow_or_path.read_text(encoding="utf-8")
        doc = yaml.safe_load(raw_text)
    elif isinstance(workflow_or_path, str):
        if "\n" in workflow_or_path or workflow_or_path.lstrip().startswith(("version:", "#")):
            doc = yaml.safe_load(workflow_or_path)
        else:
            doc = yaml.safe_load(Path(workflow_or_path).read_text(encoding="utf-8"))
    else:
        doc = workflow_or_path

    if not isinstance(doc, dict):
        raise FuzzfileValidationError("Fuzzfile root must be a YAML mapping.")

    version = doc.get("version")
    if version not in {"v4", "v1"}:
        raise FuzzfileValidationError(f"Fuzzfile 'version' must be 'v4' or 'v1', got {version!r}")

    volumes = doc.get("volumes") or {}
    jobs = doc.get("jobs") or {}
    services = doc.get("services") or {}

    if not jobs and not services:
        raise FuzzfileValidationError("Fuzzfile must define at least one entry in 'jobs' or 'services'.")

    # Fuzzball v4 enforces a single namespace across volumes, jobs, and services
    seen_names: dict[str, str] = {}
    for section_name, mapping in (("volume", volumes), ("job", jobs), ("service", services)):
        if not isinstance(mapping, dict):
            raise FuzzfileValidationError(f"Section '{section_name}s' must be a mapping.")
        for name in mapping:
            if name in seen_names:
                raise FuzzfileValidationError(
                    f"Name '{name}' is used by both a {seen_names[name]} and a {section_name}; "
                    "Fuzzball requires unique names across volumes, jobs, and services."
                )
            seen_names[name] = section_name

    # Validate volumes
    for vol_key, vol_spec in volumes.items():
        if not isinstance(vol_spec, dict):
            raise FuzzfileValidationError(f"Volume '{vol_key}' must be a mapping.")
        if version == "v4":
            if "name" in vol_spec and ("size" in vol_spec or "annotations" in vol_spec):
                raise FuzzfileValidationError(
                    f"Persistent volume '{vol_key}' ('name' set) cannot combine 'size' or 'annotations'."
                )

    executable_names = set(jobs.keys()) | set(services.keys())
    adj: dict[str, list[str]] = {name: [] for name in executable_names}

    for kind, mapping in (("job", jobs), ("service", services)):
        for stage_name, spec in mapping.items():
            if not isinstance(spec, dict):
                raise FuzzfileValidationError(f"{kind.capitalize()} '{stage_name}' must be a mapping.")

            image = spec.get("image")
            if not isinstance(image, dict) or not isinstance(image.get("uri"), str):
                raise FuzzfileValidationError(f"{kind.capitalize()} '{stage_name}' requires 'image.uri'.")
            if not image["uri"].startswith(VALID_IMAGE_SCHEMES):
                raise FuzzfileValidationError(
                    f"{kind.capitalize()} '{stage_name}' image.uri '{image['uri']}' must start with "
                    f"one of {VALID_IMAGE_SCHEMES}."
                )

            has_cmd = "command" in spec
            has_script = "script" in spec
            if has_cmd == has_script:
                raise FuzzfileValidationError(
                    f"{kind.capitalize()} '{stage_name}' must specify exactly one of 'command' or 'script'."
                )

            # Security check: forbid hardcoded inline credentials in env
            for env_entry in spec.get("env") or []:
                if not isinstance(env_entry, str) or "=" not in env_entry:
                    raise FuzzfileValidationError(
                        f"{kind.capitalize()} '{stage_name}' env entry {env_entry!r} must be 'KEY=VALUE'."
                    )
                for bad_prefix in SUSPICIOUS_ENV_PREFIXES:
                    if env_entry.upper().startswith(bad_prefix) and not env_entry.split("=", 1)[1].startswith(
                        "secret://"
                    ):
                        raise FuzzfileValidationError(
                            f"Hardcoded secret forbidden in '{stage_name}' env: {bad_prefix}..."
                        )

            mounts = spec.get("mounts") or {}
            if not isinstance(mounts, dict):
                raise FuzzfileValidationError(f"'{stage_name}' mounts must be a mapping.")
            if version == "v4":
                for m_path, vol_ref in mounts.items():
                    if not str(m_path).startswith("/"):
                        raise FuzzfileValidationError(
                            f"v4 mount key '{m_path}' in '{stage_name}' must be an absolute container path."
                        )
                    if vol_ref not in volumes:
                        raise FuzzfileValidationError(
                            f"Mount '{m_path}' in '{stage_name}' references undefined volume '{vol_ref}'."
                        )
            else:
                for vol_ref, loc_obj in mounts.items():
                    if vol_ref not in volumes:
                        raise FuzzfileValidationError(
                            f"v1 mount in '{stage_name}' references undefined volume '{vol_ref}'."
                        )
                    if not isinstance(loc_obj, dict) or not str(loc_obj.get("location", "")).startswith("/"):
                        raise FuzzfileValidationError(
                            f"v1 mount '{vol_ref}' in '{stage_name}' requires 'location: /abs/path'."
                        )

            if kind == "job":
                exclusives = sum(1 for k in ("multinode", "task-array", "network") if k in spec)
                if exclusives > 1:
                    raise FuzzfileValidationError(
                        f"Job '{stage_name}' cannot combine mutually exclusive multinode, task-array, or network."
                    )

            for dep in spec.get("depends-on") or []:
                if not isinstance(dep, dict) or "name" not in dep or "status" not in dep:
                    raise FuzzfileValidationError(
                        f"'{stage_name}' depends-on entries require 'name' and 'status'."
                    )
                if dep["name"] not in executable_names:
                    raise FuzzfileValidationError(
                        f"'{stage_name}' depends on unknown job/service '{dep['name']}'."
                    )
                if dep["status"] not in VALID_DEPENDS_STATUS:
                    raise FuzzfileValidationError(
                        f"'{stage_name}' depends-on status '{dep['status']}' invalid; "
                        f"expected one of {sorted(VALID_DEPENDS_STATUS)}."
                    )
                adj[dep["name"]].append(stage_name)

            for req_name in spec.get("requires") or []:
                if req_name not in executable_names:
                    raise FuzzfileValidationError(
                        f"'{stage_name}' requires unknown job/service '{req_name}'."
                    )
                adj[req_name].append(stage_name)

    # Cycle check via DFS
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node: str) -> None:
        if node in visiting:
            raise FuzzfileValidationError(f"Dependency cycle detected involving '{node}'.")
        if node in visited:
            return
        visiting.add(node)
        for nxt in adj.get(node, ()):
            _dfs(nxt)
        visiting.remove(node)
        visited.add(node)

    for name in executable_names:
        _dfs(name)

    return sorted(executable_names)
