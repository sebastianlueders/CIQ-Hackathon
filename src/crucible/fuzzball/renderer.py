"""Renders trials and the final turnkey workflow into CIQ Fuzzball YAML (`v4` and `v1`).

Aligned with the live CIQ Hackathon Grace Blackwell GB10 cluster (`CIQ-CHI-SPARK39` &
`CIQ-CHI-SPARK40`, VAST storage provisioner `use: team20`):
- `volumes: { vast: { name: crucible-vast, use: team20 } }`
- `mounts: { /vast: { volume: vast } }` (Fuzzball v4 object mount syntax)
- `resource: { cpu: { cores: 16 }, memory: { size: 112GiB }, devices: { nvidia.com/gpu: 1 } }`
  (no `cpu/arm64/sve` device plugin constraint, no `affinity`, no `fuzzball.io/*` compute annotation)
- Pre-cached images: `docker://nvcr.io/nvidia/pytorch:26.10-py3`, `docker://vllm/vllm-openai:v0.33.0`,
  `docker://python:3.12-slim`
- Multi-node coordination via `$MULTINODE_HOSTLIST_NOSLOTS` + `multinode: { nodes: 2, implementation: generic }`
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from crucible.config import (
    ClusterConfig,
    ExperimentConfig,
    InferencePerfCandidate,
    RecipeCandidate,
    TrainingPerfCandidate,
)
from crucible.fuzzball.remote_bootstrap import (
    REMOTE_SERVE_BENCH_PY,
    REMOTE_TRAIN_RUNNER_PY,
)
from crucible.serving.vllm_bench import build_vllm_launch_args

PROVISIONER_PIN_ANNOTATION_KEY = "fuzzball.io/provisioner_definition_id"


def trial_metrics_object_uri(cluster: ClusterConfig, run_id: str, trial_id: str) -> str:
    return f"fuzzball://{cluster.vast.object_cache_scope}/crucible/{run_id}/{trial_id}/metrics.json"


def _workflow_annotations_section(cluster: ClusterConfig) -> dict[str, Any]:
    if (
        cluster.fuzzfile_version == "v4"
        and cluster.pin_definition_annotation
        and cluster.definition_id
    ):
        return {
            "annotations": {
                PROVISIONER_PIN_ANNOTATION_KEY: cluster.definition_id,
            }
        }
    return {}


def _volume_section(
    cluster: ClusterConfig,
    egress_entries: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if cluster.fuzzfile_version == "v4":
        if cluster.vast.ephemeral:
            vol_spec: dict[str, Any] = {
                "use": "ephemeral",
                "size": cluster.vast.ephemeral_size,
            }
        else:
            use_prov = cluster.vast.provisioner or cluster.definition_id or "team20"
            vol_spec = {
                "name": cluster.vast.persistent_volume_name,
                "use": use_prov,
            }
        if egress_entries and cluster.vast.egress_metrics_to_object_cache:
            vol_spec["egress"] = [
                {
                    "source": {"uri": f"file://{rel_src.lstrip('/')}"},
                    "destination": {"uri": dst_uri},
                }
                for rel_src, dst_uri in egress_entries
            ]
        return {cluster.vast.volume_key: vol_spec}
    return {
        cluster.vast.volume_key: {
            "reference": f"volume://user/{cluster.vast.provisioner or 'team20'}/{cluster.vast.persistent_volume_name}"
        }
    }


def _mounts_section(cluster: ClusterConfig) -> dict[str, Any]:
    if cluster.fuzzfile_version == "v4":
        return {cluster.vast.mount_path: {"volume": cluster.vast.volume_key}}
    return {cluster.vast.volume_key: {"location": cluster.vast.mount_path}}


def _dependency_field(
    cluster: ClusterConfig,
    deps: list[tuple[str, str]],
) -> dict[str, Any]:
    if not deps:
        return {}
    if cluster.fuzzfile_version == "v4":
        return {"depends-on": [{"name": name, "status": status} for name, status in deps]}
    return {"requires": [name for name, _ in deps]}


def _gpu_resource_block(
    cluster: ClusterConfig,
    gpus_per_node: int | None = None,
    *,
    multinode: bool = False,
) -> dict[str, Any]:
    gpus = gpus_per_node if gpus_per_node is not None else cluster.nodes.gpus_per_node
    cores = min(8, cluster.nodes.cpu_cores_per_node) if multinode else cluster.nodes.cpu_cores_per_node
    mem_gib = min(80, cluster.nodes.memory_gb_per_node) if multinode else cluster.nodes.memory_gb_per_node
    devices: dict[str, int] = {"nvidia.com/gpu": gpus}
    if cluster.nodes.arch_device:
        devices[cluster.nodes.arch_device] = 1
    return {
        "cpu": {"cores": cores},
        "memory": {"size": f"{mem_gib}GiB"},
        "devices": devices,
    }


def _cpu_helper_resource_block(
    cluster: ClusterConfig,
    cores: int,
    mem_gib: int,
) -> dict[str, Any]:
    safe_cores = min(cores, cluster.nodes.physical_cores_per_node)
    return {
        "cpu": {"cores": safe_cores},
        "memory": {"size": f"{mem_gib}GiB"},
    }


def _team_nodes_comment(cluster: ClusterConfig) -> str:
    if not cluster.nodes.inventory:
        return f"{cluster.nodes.count}x {cluster.nodes.gpu_model}"
    return ", ".join(f"{n.hostname} ({n.node_id})" for n in cluster.nodes.inventory)


def render_training_trial_fuzzfile(
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    trial_id: str,
    recipe: RecipeCandidate,
    perf: TrainingPerfCandidate,
    max_steps: int,
    run_id: str,
    inline_splits_json: str | None = None,
) -> str:
    base_model_path = f"{cluster.vast.models_dir}/{exp.model.weights_subpath}"
    processed_path = f"{cluster.vast.datasets_dir}/{exp.dataset.processed_subpath}"
    trial_out_path = f"{cluster.vast.runs_dir}/{run_id}/trials/{trial_id}"

    recipe_str = json.dumps(recipe.model_dump(), sort_keys=True)
    perf_str = json.dumps(perf.model_dump(), sort_keys=True)

    runner_entry = (
        "/opt/crucible_remote_train.py"
        if cluster.fuzzfile_version == "v4"
        else "-m crucible.trainer.lora_trainer"
    )
    if perf.nodes > 1:
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                'SELF=$(hostname); FQDN=$(hostname -f 2>/dev/null || echo "$SELF")',
                'HOSTS=$(echo "${MULTINODE_HOSTLIST_NOSLOTS:-$SELF}" | tr \',\' \'\\n\' | sort)',
                'MASTER=${MN_MASTER:-$(echo "$HOSTS" | head -1)}',
                'RANK=$(echo "$HOSTS" | grep -nx -e "$SELF" -e "$FQDN" | head -1 | cut -d: -f1 || echo 1)',
                'NODE_RANK=${MN_NODE_RANK:-$(( ${RANK:-1} - 1 ))}',
                "# Fuzzball generic multinode runs on head node; spawn peer GB10 workers via MULTINODE_SSH_WRAPPER",
                'if [ -z "${MN_NODE_RANK:-}" ] && [ -n "${MULTINODE_SSH_WRAPPER:-}" ]; then',
                '  HEAD_IP=${MULTINODE_NODE_IP:-$(getent hosts "$SELF" | awk \'{print $1}\' | head -1)}',
                "  i=0",
                "  for h in $HOSTS; do",
                '    [ "$h" = "$SELF" ] || [ "$h" = "$FQDN" ] && continue',
                "    i=$((i + 1))",
                f'    "$MULTINODE_SSH_WRAPPER" "$h" "MN_NODE_RANK=$i MN_MASTER=$HEAD_IP HOME=/tmp/home python3 -m torch.distributed.run --nnodes {perf.nodes} --nproc-per-node {cluster.nodes.gpus_per_node} --node-rank $i --master-addr $HEAD_IP --master-port 29500 {runner_entry} --trial-id {trial_id} --base-model-dir {base_model_path} --processed-data-dir {processed_path} --output-dir {trial_out_path} --max-steps {max_steps} --recipe-json \'{recipe_str}\' --perf-json \'{perf_str}\'" > /tmp/rank$i.log 2>&1 &',
                "  done",
                '  NODE_RANK=0; MASTER=$HEAD_IP',
                "fi",
                'echo "DIAG host=$SELF master=$MASTER node_rank=$NODE_RANK"',
                "python3 -m torch.distributed.run \\",
                f"  --nnodes {perf.nodes} \\",
                f"  --nproc-per-node {cluster.nodes.gpus_per_node} \\",
                '  --node-rank "$NODE_RANK" \\',
                '  --master-addr "$MASTER" \\',
                "  --master-port 29500 \\",
                f"  {runner_entry} \\",
                f"  --trial-id {trial_id} \\",
                f"  --base-model-dir {base_model_path} \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --output-dir {trial_out_path} \\",
                f"  --max-steps {max_steps} \\",
                f"  --recipe-json '{recipe_str}' \\",
                f"  --perf-json '{perf_str}'",
            ]
        )
    else:
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                f"python3 {runner_entry} \\",
                f"  --trial-id {trial_id} \\",
                f"  --base-model-dir {base_model_path} \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --output-dir {trial_out_path} \\",
                f"  --max-steps {max_steps} \\",
                f"  --recipe-json '{recipe_str}' \\",
                f"  --perf-json '{perf_str}'",
            ]
        )

    job_files: dict[str, str] = {}
    top_files: dict[str, str] = {}
    if cluster.fuzzfile_version == "v4":
        job_files["/opt/crucible_remote_train.py"] = "file://crucible-train-runner"
        top_files["crucible-train-runner"] = REMOTE_TRAIN_RUNNER_PY
        if inline_splits_json:
            job_files["/opt/crucible_splits.json"] = "file://crucible-splits-json"
            top_files["crucible-splits-json"] = inline_splits_json

    job_spec: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **({"files": job_files} if job_files else {}),
        "env": [
            "HOME=/tmp/home",
            "PYTHONUNBUFFERED=1",
            "NCCL_DEBUG=WARN",
        ],
        "resource": _gpu_resource_block(cluster, multinode=(perf.nodes > 1)),
        "policy": {"timeout": {"execute": "4h"}, "retry": {"attempts": 2}},
        "script": script + "\n",
    }
    if perf.nodes > 1:
        job_spec["multinode"] = {
            "nodes": perf.nodes,
            "implementation": "generic",
        }

    rel_runs_prefix = cluster.vast.runs_dir.removeprefix(cluster.vast.mount_path).lstrip("/")
    egress_rel_summary = f"{rel_runs_prefix}/{run_id}/trials/{trial_id}/summary.json"
    obj_uri = trial_metrics_object_uri(cluster, run_id, trial_id)

    doc: dict[str, Any] = {
        "version": cluster.fuzzfile_version,
        **_workflow_annotations_section(cluster),
        **({"files": top_files} if top_files else {}),
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_summary, obj_uri)]),
        "jobs": {f"train-{trial_id}": job_spec},
    }
    header = (
        f"# Crucible Autoresearch Trial Fuzzfile: {trial_id}\n"
        f"# VAST Storage Provisioner: use={cluster.vast.provisioner or cluster.definition_id} | Nodes: {_team_nodes_comment(cluster)}\n"
        f"# Model: {exp.model.name} | Target Hardware: {cluster.nodes.gpu_product} ({cluster.nodes.sm_arch})\n"
        f"# Recipe: rank={recipe.lora_rank}, alpha={recipe.lora_alpha}, lr={recipe.lr}\n"
        f"# Perf: nodes={perf.nodes}, micro_batch={perf.micro_batch_size}, "
        f"chunked_ce={perf.use_chunked_ce_kernel}, grouped_moe={perf.use_grouped_moe_kernel}, "
        f"fused_lora={perf.use_fused_lora_kernel}\n\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False)


def render_serving_trial_fuzzfile(
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    trial_id: str,
    adapter_vast_path: str,
    cand: InferencePerfCandidate,
    baseline_loss: float,
    run_id: str,
) -> str:
    processed_path = f"{cluster.vast.datasets_dir}/{exp.dataset.processed_subpath}"
    out_json = f"{cluster.vast.runs_dir}/{run_id}/serving/{trial_id}/summary.json"
    bench_job_name = f"bench-{trial_id}"
    cand_json = json.dumps(cand.model_dump(), sort_keys=True)

    bench_entry = (
        "/opt/crucible_remote_serve_bench.py"
        if cluster.fuzzfile_version == "v4"
        else "-m crucible.serving.vllm_bench"
    )
    bench_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **(
            {"files": {"/opt/crucible_remote_serve_bench.py": "file://crucible-serve-runner"}}
            if cluster.fuzzfile_version == "v4"
            else {}
        ),
        "env": [
            "HOME=/tmp/home",
            "PYTHONUNBUFFERED=1",
            f"CRUCIBLE_USE_LOW_PREC_DECODE={int(cand.use_low_prec_decode_kernel)}",
        ],
        "resource": _gpu_resource_block(
            cluster,
            gpus_per_node=min(cand.tensor_parallel, cluster.nodes.gpus_per_node),
            multinode=(cand.tensor_parallel > cluster.nodes.gpus_per_node),
        ),
        "policy": {"timeout": {"execute": "1h"}},
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                f"python3 {bench_entry} \\",
                "  --endpoint http://127.0.0.1:8000 \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --trial-id {trial_id} \\",
                f"  --baseline-loss {baseline_loss:.6f} \\",
                f"  --output-json {out_json} \\",
                f"  --candidate-json '{cand_json}'",
            ]
        )
        + "\n",
    }
    if cand.tensor_parallel > cluster.nodes.gpus_per_node:
        bench_job["multinode"] = {
            "nodes": cand.tensor_parallel // cluster.nodes.gpus_per_node,
            "implementation": "generic",
        }

    rel_runs_prefix = cluster.vast.runs_dir.removeprefix(cluster.vast.mount_path).lstrip("/")
    egress_rel_srv = f"{rel_runs_prefix}/{run_id}/serving/{trial_id}/summary.json"
    srv_obj_uri = trial_metrics_object_uri(cluster, run_id, f"serving-{trial_id}")

    doc: dict[str, Any] = {
        "version": cluster.fuzzfile_version,
        **_workflow_annotations_section(cluster),
        **(
            {"files": {"crucible-serve-runner": REMOTE_SERVE_BENCH_PY}}
            if cluster.fuzzfile_version == "v4"
            else {}
        ),
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_srv, srv_obj_uri)]),
        "jobs": {bench_job_name: bench_job},
    }
    header = (
        f"# Crucible Inference Search Fuzzfile: {trial_id}\n"
        f"# VAST Storage Provisioner: use={cluster.vast.provisioner or cluster.definition_id} | Nodes: {_team_nodes_comment(cluster)}\n"
        f"# Precision: {cand.precision} | TP={cand.tensor_parallel} | Replicas={cand.replicas} | "
        f"MaxSeqs={cand.max_num_seqs}\n\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False)


def render_final_turnkey_fuzzfile(
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    winning_recipe: RecipeCandidate,
    winning_train_perf: TrainingPerfCandidate,
    winning_serve_perf: InferencePerfCandidate,
    metrics_summary: dict[str, Any],
    production_steps: int = 480,
) -> str:
    """Renders Stage D's final deliverable: ONE reproducible, richly commented Fuzzball workflow
    executing `ingest-lammps -> train-lora -> eval-held-out -> serve-vllm`."""
    raw_data_path = f"{cluster.vast.datasets_dir}/{exp.dataset.raw_subpath}"
    processed_path = f"{cluster.vast.datasets_dir}/{exp.dataset.processed_subpath}"
    base_model_path = f"{cluster.vast.models_dir}/{exp.model.weights_subpath}"
    prod_out_dir = f"{cluster.vast.runs_dir}/{exp.experiment_id}-production"
    adapter_dir = f"{prod_out_dir}/lora_adapter"

    recipe_json = json.dumps(winning_recipe.model_dump(), sort_keys=True)
    perf_json = json.dumps(winning_train_perf.model_dump(), sort_keys=True)
    crucible_src_dir = f"{cluster.vast.mount_path.rstrip('/')}/crucible-src"

    ingest_job: dict[str, Any] = {
        "image": {"uri": cluster.images.fetch},
        "mounts": _mounts_section(cluster),
        "env": [
            "HOME=/tmp/home",
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={crucible_src_dir}/src",
        ],
        "resource": _cpu_helper_resource_block(cluster, cores=8, mem_gib=16),
        "policy": {"timeout": {"execute": "30m"}},
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                "pip install --quiet --no-cache-dir pyyaml pydantic numpy httpx",
                f'if ! python3 -c "import crucible" 2>/dev/null && [ ! -d "{crucible_src_dir}/src/crucible" ]; then',
                f'  python3 -c "import urllib.request,tarfile,os; os.makedirs(\'{crucible_src_dir}\',exist_ok=True); urllib.request.urlretrieve(\'https://codeload.github.com/sebastianlueders/CIQ-Hackathon/tar.gz/seb-branch\',\'/tmp/c.tgz\')"',
                f'  tar -xzf /tmp/c.tgz -C "{crucible_src_dir}" --strip-components=1',
                "fi",
                "python3 -m crucible.cli ingest \\",
                f"  --source-dir {raw_data_path} \\",
                f"  --output-dir {processed_path} \\",
                f"  --seq-len {winning_recipe.seq_len}",
            ]
        )
        + "\n",
    }

    if winning_train_perf.nodes > 1:
        train_script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                'SELF=$(hostname); FQDN=$(hostname -f 2>/dev/null || echo "$SELF")',
                'HOSTS=$(echo "${MULTINODE_HOSTLIST_NOSLOTS:-$SELF}" | tr \',\' \'\\n\' | sort)',
                'MASTER=${MN_MASTER:-$(echo "$HOSTS" | head -1)}',
                'RANK=$(echo "$HOSTS" | grep -nx -e "$SELF" -e "$FQDN" | head -1 | cut -d: -f1 || echo 1)',
                'NODE_RANK=${MN_NODE_RANK:-$(( ${RANK:-1} - 1 ))}',
                'if [ -z "${MN_NODE_RANK:-}" ] && [ -n "${MULTINODE_SSH_WRAPPER:-}" ]; then',
                '  HEAD_IP=${MULTINODE_NODE_IP:-$(getent hosts "$SELF" | awk \'{print $1}\' | head -1)}',
                "  i=0",
                "  for h in $HOSTS; do",
                '    [ "$h" = "$SELF" ] || [ "$h" = "$FQDN" ] && continue',
                "    i=$((i + 1))",
                f'    "$MULTINODE_SSH_WRAPPER" "$h" "MN_NODE_RANK=$i MN_MASTER=$HEAD_IP PYTHONPATH={crucible_src_dir}/src HOME=/tmp/home python3 -m torch.distributed.run --nnodes {winning_train_perf.nodes} --nproc-per-node {cluster.nodes.gpus_per_node} --node-rank $i --master-addr $HEAD_IP --master-port 29500 -m crucible.trainer.lora_trainer --trial-id production-lora --base-model-dir {base_model_path} --processed-data-dir {processed_path} --output-dir {prod_out_dir} --max-steps {production_steps} --recipe-json \'{recipe_json}\' --perf-json \'{perf_json}\'" > /tmp/rank$i.log 2>&1 &',
                "  done",
                '  NODE_RANK=0; MASTER=$HEAD_IP',
                "fi",
                "python3 -m torch.distributed.run \\",
                f"  --nnodes {winning_train_perf.nodes} \\",
                f"  --nproc-per-node {cluster.nodes.gpus_per_node} \\",
                '  --node-rank "$NODE_RANK" \\',
                '  --master-addr "$MASTER" \\',
                "  --master-port 29500 \\",
                "  -m crucible.trainer.lora_trainer \\",
                "  --trial-id production-lora \\",
                f"  --base-model-dir {base_model_path} \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --output-dir {prod_out_dir} \\",
                f"  --max-steps {production_steps} \\",
                f"  --recipe-json '{recipe_json}' \\",
                f"  --perf-json '{perf_json}'",
            ]
        )
    else:
        train_script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /tmp/home",
                "python3 -m crucible.trainer.lora_trainer \\",
                "  --trial-id production-lora \\",
                f"  --base-model-dir {base_model_path} \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --output-dir {prod_out_dir} \\",
                f"  --max-steps {production_steps} \\",
                f"  --recipe-json '{recipe_json}' \\",
                f"  --perf-json '{perf_json}'",
            ]
        )

    train_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **_dependency_field(cluster, [("ingest-lammps", "FINISHED")]),
        "env": [
            "HOME=/tmp/home",
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={crucible_src_dir}/src",
            "NCCL_DEBUG=WARN",
        ],
        "resource": _gpu_resource_block(cluster, multinode=(winning_train_perf.nodes > 1)),
        "policy": {"timeout": {"execute": "12h"}, "retry": {"attempts": 2}},
        "script": train_script + "\n",
    }
    if winning_train_perf.nodes > 1:
        train_job["multinode"] = {
            "nodes": winning_train_perf.nodes,
            "implementation": "generic",
        }

    eval_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **_dependency_field(cluster, [("train-lora", "FINISHED")]),
        "env": ["HOME=/tmp/home", "PYTHONUNBUFFERED=1"],
        "resource": _gpu_resource_block(cluster),
        "policy": {"timeout": {"execute": "45m"}},
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "python3 - <<'PY'",
                "import json, pathlib, sys",
                f"summary = json.loads(pathlib.Path('{prod_out_dir}/summary.json').read_text())",
                f"ceiling = {metrics_summary.get('quality_floor_perplexity_ceiling', 12.0):.4f}",
                "ppl = float(summary['held_out_perplexity'])",
                "print(f'Verified held-out perplexity: {ppl:.4f} (ceiling: {ceiling:.4f})')",
                "if ppl > ceiling:",
                "    sys.exit('Held-out quality floor check failed — aborting promotion to serving.')",
                "PY",
            ]
        )
        + "\n",
    }

    vllm_cmd = build_vllm_launch_args(
        model_path=base_model_path,
        adapter_path=adapter_dir,
        cand=winning_serve_perf,
        port=8000,
    )
    serve_service: dict[str, Any] = {
        "image": {"uri": cluster.images.serve},
        "mounts": _mounts_section(cluster),
        **_dependency_field(cluster, [("eval-held-out", "FINISHED")]),
        "env": [
            "HOME=/tmp",
            "PYTHONUNBUFFERED=1",
            f"CRUCIBLE_USE_LOW_PREC_DECODE={int(winning_serve_perf.use_low_prec_decode_kernel)}",
            f"CRUCIBLE_TARGET_REPLICAS={winning_serve_perf.replicas}",
        ],
        "command": vllm_cmd,
        "resource": _gpu_resource_block(
            cluster,
            gpus_per_node=min(winning_serve_perf.tensor_parallel, cluster.nodes.gpus_per_node),
            multinode=(winning_serve_perf.tensor_parallel > cluster.nodes.gpus_per_node),
        ),
        "persist": True,
        "network": {
            "ports": [{"name": "http", "port": 8000}],
            "endpoints": [
                {
                    "name": "crucible-coder-api",
                    "port-name": "http",
                    "protocol": "http",
                    "type": "subdomain",
                    "scope": "group",
                }
            ],
        },
    }
    if winning_serve_perf.tensor_parallel > cluster.nodes.gpus_per_node:
        serve_service["multinode"] = {
            "nodes": winning_serve_perf.tensor_parallel // cluster.nodes.gpus_per_node,
            "implementation": "generic",
        }

    doc: dict[str, Any] = {
        "version": cluster.fuzzfile_version,
        **_workflow_annotations_section(cluster),
        "volumes": _volume_section(cluster),
        "jobs": {
            "ingest-lammps": ingest_job,
            "train-lora": train_job,
            "eval-held-out": eval_job,
        },
        "services": {
            "serve-vllm": serve_service,
        },
    }

    comment_header = f"""# ==============================================================================
# Crucible Turnkey Autoresearch Workflow (ingest -> train -> eval -> serve)
# ==============================================================================
# Cluster        : {cluster.cluster_name} | VAST Storage Provisioner: use={cluster.vast.provisioner or cluster.definition_id}
# Nodes          : {_team_nodes_comment(cluster)}
# Node Hardware  : 1x {cluster.nodes.gpu_product} ({cluster.nodes.gpu_arch}, {cluster.nodes.sm_arch}), {cluster.nodes.physical_cores_per_node}c arm64, {cluster.nodes.physical_memory_gib_per_node} GiB/node
# Per-Node Alloc : {cluster.nodes.cpu_cores_per_node} cores, {cluster.nodes.memory_gb_per_node}GiB RAM, {cluster.nodes.gpus_per_node}x nvidia.com/gpu
# Shared Storage : VAST persistent volume '{cluster.vast.persistent_volume_name}' (use: {cluster.vast.provisioner or cluster.definition_id}) mounted at {cluster.vast.mount_path}
# Base Model     : {exp.model.name} (MoE, {exp.model.vocab_size:,} vocab, {exp.model.bf16_footprint_gb} GB BF16)
# Dataset        : {exp.dataset.name} (whole-file sealed held-out split; raw data never left VAST)
#
# Measured Autoresearch Decisions:
#   [Stage A - Winning LoRA Recipe]
#     rank={winning_recipe.lora_rank}, alpha={winning_recipe.lora_alpha}, lr={winning_recipe.lr},
#     modules={winning_recipe.target_modules}, seq_len={winning_recipe.seq_len}
#     Held-out perplexity: {metrics_summary.get('base_perplexity', 0.0):.3f} (base) -> {metrics_summary.get('finetuned_perplexity', 0.0):.3f} (fine-tuned)
#
#   [Stage B - Winning Training Performance Config ({winning_train_perf.label})]
#     nodes={winning_train_perf.nodes}, micro_batch={winning_train_perf.micro_batch_size}, grad_accum={winning_train_perf.grad_accum_steps},
#     activation_checkpointing={winning_train_perf.activation_checkpointing},
#     sm_121 kernels: chunked_ce={winning_train_perf.use_chunked_ce_kernel}, grouped_moe={winning_train_perf.use_grouped_moe_kernel}, fused_lora={winning_train_perf.use_fused_lora_kernel}
#     Throughput: {metrics_summary.get('baseline_train_tok_s', 0.0):.1f} -> {metrics_summary.get('best_train_tok_s', 0.0):.1f} tok/s ({metrics_summary.get('train_speedup_x', 1.0):.2f}x)
#
#   [Stage C - Winning vLLM Serving Config ({winning_serve_perf.label})]
#     precision={winning_serve_perf.precision}, replicas={winning_serve_perf.replicas}, tensor_parallel={winning_serve_perf.tensor_parallel},
#     max_num_seqs={winning_serve_perf.max_num_seqs}, low_prec_decode_kernel={winning_serve_perf.use_low_prec_decode_kernel}
#     Serving throughput: {metrics_summary.get('baseline_serve_tok_s', 0.0):.1f} -> {metrics_summary.get('best_serve_tok_s', 0.0):.1f} tok/s
#
# Launch end-to-end on CIQ Fuzzball (Team 20):
#   fuzzball workflow validate final-crucible-workflow.yaml
#   fuzzball workflow start final-crucible-workflow.yaml --name {exp.experiment_id}-team20
# ==============================================================================

"""
    return comment_header + yaml.safe_dump(doc, sort_keys=False)
