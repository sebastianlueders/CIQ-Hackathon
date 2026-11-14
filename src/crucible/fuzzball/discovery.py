"""Automatic CIQ Fuzzball cluster discovery via `00-fetch.yaml`, Fuzzball MCP, and `fuzzball` CLI.

Treats all CLI and MCP responses strictly as untrusted data (validated against identifier patterns
and numeric bounds; never executed as shell commands or used to mutate workspace settings).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from crucible.config import ClusterConfig, repo_root
from crucible.fuzzball.mcp_client import FuzzballMCPStdioClient

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_./:\-]{1,128}$")
SAFE_IMAGE_RE = re.compile(r"^(docker|oras)://[A-Za-z0-9_./:@\-]{1,240}$")


def _clean_ident(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.strip()
    return s if SAFE_NAME_RE.match(s) else None


def _clean_image(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.strip()
    return s if SAFE_IMAGE_RE.match(s) else None


@dataclass
class ClusterDiscoveryReport:
    fuzzball_cli_available: bool = False
    mcp_source_used: str | None = None
    fetch_workflow_inspected: str | None = None
    active_cluster_context: str | None = None
    discovered_volumes: list[dict[str, str]] = field(default_factory=list)
    discovered_provisioners: list[str] = field(default_factory=list)
    discovered_workflows: list[dict[str, str]] = field(default_factory=list)
    discovered_node_queues: list[dict[str, Any]] = field(default_factory=list)
    applied_overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_fetch_workflow(
    fetch_yaml_path: Path,
    cfg_data: dict[str, Any],
    report: ClusterDiscoveryReport,
) -> None:
    """Extract volume names, provisioners, mount paths, syntax version, and image URIs from `00-fetch.yaml`."""
    if not fetch_yaml_path.exists():
        return
    try:
        doc = yaml.safe_load(fetch_yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.notes.append(f"Skipped malformed {fetch_yaml_path}: {exc}")
        return
    if not isinstance(doc, dict):
        return

    report.fetch_workflow_inspected = str(fetch_yaml_path)
    ver = doc.get("version")
    if ver in {"v4", "v1"} and cfg_data.get("fuzzfile_version") != ver:
        cfg_data["fuzzfile_version"] = ver
        report.applied_overrides["fuzzfile_version"] = ver

    annotations = doc.get("annotations")
    if isinstance(annotations, dict):
        def_id = _clean_ident(annotations.get("fuzzball.io/provisioner_definition_id"))
        if def_id:
            cfg_data["definition_id"] = def_id
            report.applied_overrides["definition_id"] = def_id

    volumes = doc.get("volumes")
    if isinstance(volumes, dict) and volumes:
        vol_key, vol_spec = next(iter(volumes.items()))
        clean_key = _clean_ident(vol_key)
        if clean_key:
            cfg_data["vast"]["volume_key"] = clean_key
        if isinstance(vol_spec, dict):
            if ver == "v4":
                prov = _clean_ident(vol_spec.get("use"))
                vname = _clean_ident(vol_spec.get("name"))
                if prov:
                    cfg_data["vast"]["provisioner"] = prov
                    report.applied_overrides["vast.provisioner"] = prov
                if vname:
                    cfg_data["vast"]["persistent_volume_name"] = vname
                    report.applied_overrides["vast.persistent_volume_name"] = vname
                    report.discovered_volumes.append(
                        {"name": vname, "provisioner": prov or "auto", "source": "00-fetch.yaml"}
                    )
            else:
                ref = vol_spec.get("reference", "")
                if isinstance(ref, str) and ref.startswith("volume://"):
                    parts = ref.removeprefix("volume://").split("/")
                    if len(parts) >= 3:
                        prov = _clean_ident(parts[1])
                        vname = _clean_ident(parts[2])
                        if prov:
                            cfg_data["vast"]["provisioner"] = prov
                        if vname:
                            cfg_data["vast"]["persistent_volume_name"] = vname

    jobs = doc.get("jobs")
    if isinstance(jobs, dict):
        for job_spec in jobs.values():
            if not isinstance(job_spec, dict):
                continue
            img = _clean_image((job_spec.get("image") or {}).get("uri"))
            if img:
                cfg_data["images"]["fetch"] = img
                break


def _parse_pipe_table(stdout: str) -> list[dict[str, str]]:
    """Parse Fuzzball CLI pipe-delimited tables (`ID | CLUSTER | STATUS | NAME`)."""
    lines = [ln.strip() for ln in stdout.splitlines() if "|" in ln]
    if len(lines) < 2:
        return []
    headers = [h.strip().lower().replace(" ", "_") for h in lines[0].split("|") if h.strip()]
    rows: list[dict[str, str]] = []
    for raw_ln in lines[1:]:
        if set(raw_ln.replace("|", "").strip()) <= {"-", "="}:
            continue
        cols = [c.strip() for c in raw_ln.split("|")]
        if len(cols) < len(headers):
            continue
        row = {headers[i]: cols[i] for i in range(len(headers))}
        rows.append(row)
    return rows


def _run_fuzzball_read_cmd(fuzzball_bin: str, subargs: list[str], timeout_s: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(
            [fuzzball_bin, *subargs],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
    except Exception:
        return None
    return None


def probe_fuzzball_cli(
    cfg_data: dict[str, Any],
    report: ClusterDiscoveryReport,
    fuzzball_bin: str = "fuzzball",
) -> None:
    if shutil.which(fuzzball_bin) is None:
        report.fuzzball_cli_available = False
        report.notes.append("fuzzball CLI binary not found on PATH; using config/MCP/00-fetch values.")
        return

    report.fuzzball_cli_available = True

    ctx_out = _run_fuzzball_read_cmd(fuzzball_bin, ["context", "list"])
    if ctx_out:
        for row in _parse_pipe_table(ctx_out):
            name = _clean_ident(row.get("name") or row.get("context") or row.get("cluster"))
            is_current = "*" in "".join(row.values()) or row.get("current", "").lower() in {"true", "yes", "*"}
            if name and (is_current or report.active_cluster_context is None):
                report.active_cluster_context = name
                cfg_data["cluster_name"] = name
                report.applied_overrides["cluster_name"] = name

    prov_out = _run_fuzzball_read_cmd(fuzzball_bin, ["volume", "provisioner", "list"])
    if prov_out:
        for row in _parse_pipe_table(prov_out):
            pname = _clean_ident(row.get("name") or row.get("id"))
            if pname:
                report.discovered_provisioners.append(pname)
        vast_provs = [p for p in report.discovered_provisioners if "vast" in p.lower()]
        if vast_provs:
            cfg_data["vast"]["provisioner"] = vast_provs[0]
            report.applied_overrides["vast.provisioner"] = vast_provs[0]

    vol_out = _run_fuzzball_read_cmd(fuzzball_bin, ["volume", "list"])
    if vol_out:
        for row in _parse_pipe_table(vol_out):
            vname = _clean_ident(row.get("name") or row.get("volume"))
            vprov = _clean_ident(row.get("provisioner") or row.get("storage_class") or "")
            if vname:
                report.discovered_volumes.append(
                    {"name": vname, "provisioner": vprov or "unknown", "source": "fuzzball_cli"}
                )
        preferred = [
            v
            for v in report.discovered_volumes
            if any(tok in v["name"].lower() for tok in ("vast", "crucible", "lammps", "hackathon", "shared"))
        ]
        chosen = preferred[0] if preferred else (report.discovered_volumes[0] if report.discovered_volumes else None)
        if chosen:
            cfg_data["vast"]["persistent_volume_name"] = chosen["name"]
            report.applied_overrides["vast.persistent_volume_name"] = chosen["name"]
            if chosen["provisioner"] != "unknown":
                cfg_data["vast"]["provisioner"] = chosen["provisioner"]
                report.applied_overrides["vast.provisioner"] = chosen["provisioner"]

    wf_out = _run_fuzzball_read_cmd(fuzzball_bin, ["workflow", "list", "--order-by", "-created"])
    if wf_out:
        for row in _parse_pipe_table(wf_out)[:10]:
            wf_id = _clean_ident(row.get("id"))
            wf_name = _clean_ident(row.get("name"))
            wf_status = _clean_ident(row.get("status"))
            if wf_id:
                report.discovered_workflows.append(
                    {"id": wf_id, "name": wf_name or "unnamed", "status": wf_status or "unknown"}
                )

    node_out = _run_fuzzball_read_cmd(fuzzball_bin, ["node", "list"]) or _run_fuzzball_read_cmd(
        fuzzball_bin, ["queue", "list"]
    )
    if node_out:
        parsed_nodes = _parse_pipe_table(node_out)
        if parsed_nodes:
            report.discovered_node_queues = parsed_nodes
            cfg_data["nodes"]["count"] = max(1, len(parsed_nodes))
            report.applied_overrides["nodes.count"] = cfg_data["nodes"]["count"]


def apply_mcp_discovery_payload(
    payload: dict[str, Any],
    cfg_data: dict[str, Any],
    report: ClusterDiscoveryReport,
    source_label: str,
) -> None:
    """Ingest a structured Fuzzball MCP discovery snapshot (treated strictly as untrusted data)."""
    report.mcp_source_used = source_label

    cluster_name = _clean_ident(payload.get("cluster_name") or payload.get("context"))
    if cluster_name:
        report.active_cluster_context = cluster_name
        cfg_data["cluster_name"] = cluster_name
        report.applied_overrides["cluster_name"] = cluster_name

    def_id = _clean_ident(payload.get("definition_id") or payload.get("definitionId"))
    if def_id:
        cfg_data["definition_id"] = def_id
        report.applied_overrides["definition_id"] = def_id

    fz_ver = payload.get("fuzzfile_version")
    if fz_ver in {"v4", "v1"}:
        cfg_data["fuzzfile_version"] = fz_ver
        report.applied_overrides["fuzzfile_version"] = fz_ver

    for raw_prov in payload.get("provisioners") or []:
        pname = _clean_ident(raw_prov if isinstance(raw_prov, str) else raw_prov.get("name"))
        if pname and pname not in report.discovered_provisioners:
            report.discovered_provisioners.append(pname)

    for raw_vol in payload.get("volumes") or []:
        if not isinstance(raw_vol, dict):
            continue
        vname = _clean_ident(raw_vol.get("name"))
        vprov = _clean_ident(raw_vol.get("provisioner") or raw_vol.get("use"))
        mount = _clean_ident(raw_vol.get("mount_path"))
        if vname:
            report.discovered_volumes.append(
                {"name": vname, "provisioner": vprov or "auto", "source": source_label}
            )
            if raw_vol.get("selected") or "vast" in (vprov or "").lower() or "vast" in vname.lower():
                cfg_data["vast"]["persistent_volume_name"] = vname
                report.applied_overrides["vast.persistent_volume_name"] = vname
                if vprov:
                    cfg_data["vast"]["provisioner"] = vprov
                    report.applied_overrides["vast.provisioner"] = vprov
                if mount and mount.startswith("/"):
                    cfg_data["vast"]["mount_path"] = mount
                    cfg_data["vast"]["models_dir"] = f"{mount.rstrip('/')}/models"
                    cfg_data["vast"]["datasets_dir"] = f"{mount.rstrip('/')}/data"
                    cfg_data["vast"]["runs_dir"] = f"{mount.rstrip('/')}/runs"
                    report.applied_overrides["vast.mount_path"] = mount

    images = payload.get("images")
    if isinstance(images, dict):
        for role in ("fetch", "train", "serve"):
            clean_uri = _clean_image(images.get(role))
            if clean_uri:
                cfg_data["images"][role] = clean_uri
                report.applied_overrides[f"images.{role}"] = clean_uri

    nodes_list = payload.get("nodes")
    if isinstance(nodes_list, list) and nodes_list:
        valid_nodes = [n for n in nodes_list if isinstance(n, dict)]
        if valid_nodes:
            inventory: list[dict[str, Any]] = []
            discovered_queues: list[dict[str, Any]] = []
            detected_def_id: str | None = None
            detected_arch_dev: str | None = None

            for idx, n in enumerate(valid_nodes):
                # Support both Fuzzball 4.2 nested `{"node": {...}, "resource": {...}}` and flat dicts
                node_meta = n.get("node") if isinstance(n.get("node"), dict) else {}
                res_meta = n.get("resource") if isinstance(n.get("resource"), dict) else {}
                total_res = res_meta.get("total") if isinstance(res_meta.get("total"), dict) else res_meta
                ann = res_meta.get("annotations") if isinstance(res_meta.get("annotations"), dict) else {}

                if detected_def_id is None:
                    cand_def = _clean_ident(ann.get("fuzzball.io/provisioner_definition_id"))
                    if cand_def:
                        detected_def_id = cand_def

                raw_id = _clean_ident(node_meta.get("id") or n.get("node_id") or n.get("id") or n.get("name"))
                hostname = _clean_ident(node_meta.get("hostname") or n.get("hostname") or n.get("name")) or (
                    raw_id or f"node-{idx}"
                )
                node_id = raw_id or hostname
                address = node_id.split("/")[0] if "/" in node_id else str(n.get("address", hostname))[:64]

                raw_status = node_meta.get("status", n.get("status", "READY"))
                status_str = "READY" if raw_status in (2, "2", "READY", "Ready", "ready") else str(raw_status)

                devices = total_res.get("devices") if isinstance(total_res.get("devices"), dict) else {}
                gpus = int(devices.get("nvidia.com/gpu", n.get("gpus", 1)) or 0)
                for dev_key in ("cpu/arm64/sve", "cpu/arm64", "cpu/x86_64", "cpu/x86"):
                    if dev_key in devices and detected_arch_dev is None:
                        detected_arch_dev = dev_key

                cpu_block = total_res.get("cpu") if isinstance(total_res.get("cpu"), dict) else {}
                phys_cores = int(cpu_block.get("cores") or n.get("cores") or 20)
                mem_block = total_res.get("mem") if isinstance(total_res.get("mem"), dict) else {}
                if isinstance(mem_block.get("bytes"), (int, float)) and mem_block["bytes"] > 0:
                    phys_mem_gib = round(float(mem_block["bytes"]) / float(1024**3), 1)
                else:
                    phys_mem_gib = float(n.get("memory_gib") or n.get("memory_gb") or 127.9)

                gpu_prod = str(n.get("gpu_product") or n.get("gpu_model") or "NVIDIA-GB10")[:64]
                inv_entry = {
                    "hostname": hostname,
                    "node_id": node_id,
                    "address": address,
                    "status": status_str,
                    "gpu_product": gpu_prod,
                    "gpus": max(gpus, 1) if "gpu" in gpu_prod.lower() or "gb10" in gpu_prod.lower() else gpus,
                    "cores": max(1, min(phys_cores, 256)),
                    "memory_gib": max(8.0, min(phys_mem_gib, 4096.0)),
                }
                inventory.append(inv_entry)
                discovered_queues.append(inv_entry)

            # Filter to target definition_id when multiple node pools exist on the cluster
            target_def = cfg_data.get("definition_id")
            if detected_def_id and (not target_def or detected_def_id == target_def):
                cfg_data["definition_id"] = detected_def_id
                report.applied_overrides["definition_id"] = detected_def_id

            report.discovered_node_queues = discovered_queues
            cfg_data["nodes"]["inventory"] = inventory
            cfg_data["nodes"]["count"] = len(inventory)
            report.applied_overrides["nodes.count"] = len(inventory)

            first_inv = inventory[0]
            phys_c = int(first_inv["cores"])
            phys_m = float(first_inv["memory_gib"])
            cfg_data["nodes"]["physical_cores_per_node"] = phys_c
            cfg_data["nodes"]["physical_memory_gib_per_node"] = phys_m
            # Leave safe Substrate/OS & helper co-scheduling headroom (e.g. 16 of 20 cores, 112 of 127.9 GiB)
            safe_c = max(1, phys_c - 4) if phys_c >= 8 else phys_c
            safe_m = max(8, int(phys_m * 0.88))
            cfg_data["nodes"]["cpu_cores_per_node"] = min(cfg_data["nodes"]["cpu_cores_per_node"], phys_c) or safe_c
            if cfg_data["nodes"]["cpu_cores_per_node"] > safe_c:
                cfg_data["nodes"]["cpu_cores_per_node"] = safe_c
            if float(cfg_data["nodes"]["memory_gb_per_node"]) > phys_m - 4.0:
                cfg_data["nodes"]["memory_gb_per_node"] = safe_m
            if detected_arch_dev:
                cfg_data["nodes"]["arch_device"] = detected_arch_dev
                report.applied_overrides["nodes.arch_device"] = detected_arch_dev

    for wf in payload.get("workflows") or []:
        if isinstance(wf, dict):
            wid = _clean_ident(wf.get("id"))
            wname = _clean_ident(wf.get("name"))
            wstatus = _clean_ident(wf.get("status"))
            if wid or wname:
                report.discovered_workflows.append(
                    {"id": wid or "n/a", "name": wname or "unnamed", "status": wstatus or "unknown"}
                )


def _probe_mcp_http_endpoint(url: str) -> dict[str, Any] | None:
    """Query a local/HTTP Fuzzball MCP JSON-RPC endpoint if configured via `CRUCIBLE_FUZZBALL_MCP_URL`."""
    try:
        with httpx.Client(timeout=4.0) as client:
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            resp = client.post(url, json=init_req)
            resp.raise_for_status()
            tools_body = resp.json()
            tools = (tools_body.get("result") or {}).get("tools") or []
            tool_names = {t.get("name"): t for t in tools if isinstance(t, dict)}

            aggregated: dict[str, Any] = {}
            for key, candidates in (
                ("clusters", ("cluster_list", "list_clusters")),
                ("volumes", ("volume_list", "list_volumes", "volumes_list")),
                ("workflows", ("workflow_list", "list_workflows", "workflows_list")),
                ("nodes", ("node_list", "list_nodes", "nodes_list", "cluster_resources")),
            ):
                matched = next((c for c in candidates if c in tool_names), None)
                if matched:
                    call_resp = client.post(
                        url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {"name": matched, "arguments": {}},
                        },
                    )
                    if call_resp.status_code == 200:
                        content = ((call_resp.json().get("result") or {}).get("content")) or []
                        if content and isinstance(content[0], dict) and "text" in content[0]:
                            try:
                                aggregated[key] = json.loads(content[0]["text"])
                            except Exception:
                                pass
            return aggregated if aggregated else None
    except Exception:
        return None


def probe_fuzzball_mcp_stdio(
    cfg_data: dict[str, Any],
    report: ClusterDiscoveryReport,
    mcp_command: list[str] | None = None,
) -> bool:
    """Query `fuzzball mcp serve` (CIQ Fuzzball 4.2 MCP) via native JSON-RPC stdio client."""
    try:
        with FuzzballMCPStdioClient(command=mcp_command, allow_write=False, allow_exec=False) as mcp:
            payload: dict[str, Any] = {}
            if "cluster_list" in mcp.advertised_tools:
                clusters_raw = mcp.cluster_list()
                clusters = (
                    clusters_raw.get("clusters", [])
                    if isinstance(clusters_raw, dict)
                    else (clusters_raw if isinstance(clusters_raw, list) else [])
                )
                if clusters and isinstance(clusters[0], dict):
                    cname = clusters[0].get("name") or clusters[0].get("id")
                    if isinstance(cname, str):
                        payload["cluster_name"] = cname

            if "volume_list" in mcp.advertised_tools:
                vols_raw = mcp.volume_list()
                vols = (
                    vols_raw.get("volumes", [])
                    if isinstance(vols_raw, dict)
                    else (vols_raw if isinstance(vols_raw, list) else [])
                )
                if isinstance(vols, list):
                    payload["volumes"] = vols

            if "node_list" in mcp.advertised_tools:
                nodes_raw = mcp.node_list(fields=["id", "resource"])
                raw_list = (
                    nodes_raw.get("nodes", [])
                    if isinstance(nodes_raw, dict)
                    else (nodes_raw if isinstance(nodes_raw, list) else [])
                )
                valid_raw = [n for n in raw_list if isinstance(n, dict)]
                target_def = cfg_data.get("definition_id")
                if target_def:
                    matched_def = [
                        n
                        for n in valid_raw
                        if (
                            isinstance(n.get("resource"), dict)
                            and isinstance(n["resource"].get("annotations"), dict)
                            and n["resource"]["annotations"].get("fuzzball.io/provisioner_definition_id")
                            == target_def
                        )
                    ]
                    if matched_def:
                        valid_raw = matched_def
                if valid_raw:
                    payload["nodes"] = valid_raw

            if "workflow_list" in mcp.advertised_tools:
                wfs_raw = mcp.workflow_list()
                wfs = (
                    wfs_raw.get("workflows", [])
                    if isinstance(wfs_raw, dict)
                    else (wfs_raw if isinstance(wfs_raw, list) else [])
                )
                if isinstance(wfs, list):
                    payload["workflows"] = wfs

            apply_mcp_discovery_payload(payload, cfg_data, report, "fuzzball_mcp_stdio")
            return True
    except Exception as exc:
        report.notes.append(f"fuzzball mcp serve stdio probe skipped: {exc}")
        return False


def discover_cluster_config(
    base_config: ClusterConfig | None = None,
    fetch_workflow_path: Path | None = None,
    mcp_snapshot_path: Path | None = None,
    mcp_command: list[str] | None = None,
    fuzzball_bin: str = "fuzzball",
) -> tuple[ClusterConfig, ClusterDiscoveryReport]:
    """Discover live Fuzzball resources across `00-fetch.yaml`, Fuzzball 4.2 MCP, and `fuzzball` CLI."""
    cfg = base_config or ClusterConfig.load(repo_root() / "configs" / "cluster.default.yaml")
    cfg_data = cfg.model_dump()
    report = ClusterDiscoveryReport()

    # 1. Inspect local workflows/00-fetch.yaml if present
    resolved_fetch = fetch_workflow_path or (repo_root() / "workflows" / "00-fetch.yaml")
    inspect_fetch_workflow(resolved_fetch, cfg_data, report)

    # 2. Check Fuzzball MCP snapshot file, HTTP MCP URL, or native `fuzzball mcp serve` stdio
    snap_env = os.getenv("CRUCIBLE_FUZZBALL_SNAPSHOT_JSON", "").strip()
    resolved_snap = mcp_snapshot_path or (Path(snap_env) if snap_env else None)
    mcp_succeeded = False
    if resolved_snap is not None and resolved_snap.exists():
        raw_snap = json.loads(resolved_snap.read_text(encoding="utf-8"))
        if isinstance(raw_snap, dict):
            apply_mcp_discovery_payload(raw_snap, cfg_data, report, f"mcp_snapshot:{resolved_snap.name}")
            mcp_succeeded = True
    else:
        mcp_url = os.getenv("CRUCIBLE_FUZZBALL_MCP_URL", "").strip()
        if mcp_url:
            http_payload = _probe_mcp_http_endpoint(mcp_url)
            if http_payload:
                apply_mcp_discovery_payload(http_payload, cfg_data, report, f"mcp_http:{mcp_url}")
                mcp_succeeded = True
        elif mcp_command is not None or FuzzballMCPStdioClient.is_binary_available(fuzzball_bin):
            cmd = mcp_command or (
                shlex.split(os.environ["CRUCIBLE_FUZZBALL_MCP_CMD"])
                if os.getenv("CRUCIBLE_FUZZBALL_MCP_CMD")
                else [fuzzball_bin, "mcp", "serve"]
            )
            mcp_succeeded = probe_fuzzball_mcp_stdio(cfg_data, report, mcp_command=cmd)

    # 3. Probe live `fuzzball` CLI tables when MCP did not already populate live state
    if not mcp_succeeded:
        probe_fuzzball_cli(cfg_data, report, fuzzball_bin=fuzzball_bin)
    else:
        report.fuzzball_cli_available = shutil.which(fuzzball_bin) is not None

    # 4. Re-apply explicit CRUCIBLE_* environment variable overrides on top
    final_cfg = ClusterConfig.model_validate(cfg_data).with_env_overrides()
    return final_cfg, report
