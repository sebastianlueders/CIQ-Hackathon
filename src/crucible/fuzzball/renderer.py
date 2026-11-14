"""Renders trials and the final turnkey workflow into CIQ Fuzzball YAML (`v4` and `v1`).

Targets Team 20 (`definitionId: team20`) by default across Grace Blackwell GB10 nodes
`CIQ-CHI-SPARK39` (`192.168.99.48/7331`) and `CIQ-CHI-SPARK40` (`192.168.99.49/7331`),
sizing container allocations safely inside each node's `1 GPU / 20 cpu/arm64/sve cores / 127.9 GiB`
envelope (`16 cores / 112GiB`) so Fuzzball never rejects for Substrate/OS overhead and CPU helper
stages (`bench-*`, `crucible-studio`) can co-schedule without deadlocking.
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
            vol_spec = {"name": cluster.vast.persistent_volume_name}
            if cluster.vast.provisioner:
                vol_spec = {
                    "use": cluster.vast.provisioner,
                    "name": cluster.vast.persistent_volume_name,
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
            "reference": f"volume://user/{cluster.vast.provisioner or 'vast'}/{cluster.vast.persistent_volume_name}"
        }
    }


def _mounts_section(cluster: ClusterConfig) -> dict[str, Any]:
    if cluster.fuzzfile_version == "v4":
        return {cluster.vast.mount_path: cluster.vast.volume_key}
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


def _cpu_devices_map(cluster: ClusterConfig) -> dict[str, int]:
    if cluster.nodes.arch_device:
        return {cluster.nodes.arch_device: 1}
    return {}


def _gpu_resource_block(cluster: ClusterConfig, gpus_per_node: int | None = None) -> dict[str, Any]:
    gpus = gpus_per_node if gpus_per_node is not None else cluster.nodes.gpus_per_node
    devices: dict[str, int] = {"nvidia.com/gpu": gpus}
    if cluster.nodes.arch_device:
        devices[cluster.nodes.arch_device] = 1
    return {
        "cpu": {
            "cores": cluster.nodes.cpu_cores_per_node,
            "affinity": "NUMA",
        },
        "memory": {
            "size": f"{cluster.nodes.memory_gb_per_node}GiB",
        },
        "devices": devices,
    }


def _cpu_helper_resource_block(
    cluster: ClusterConfig,
    cores: int,
    mem_gib: int,
    affinity: str = "CORE",
) -> dict[str, Any]:
    safe_cores = min(cores, cluster.nodes.physical_cores_per_node)
    block: dict[str, Any] = {
        "cpu": {"cores": safe_cores, "affinity": affinity},
        "memory": {"size": f"{mem_gib}GiB"},
    }
    devs = _cpu_devices_map(cluster)
    if devs:
        block["devices"] = devs
    return block


def _team_nodes_comment(cluster: ClusterConfig) -> str:
    if not cluster.nodes.inventory:
        return f"{cluster.nodes.count}x {cluster.nodes.gpu_model}"
    parts = [f"{n.hostname} ({n.node_id})" for n in cluster.nodes.inventory]
    return ", ".join(parts)


def render_training_trial_fuzzfile(
    cluster: ClusterConfig,
    exp: ExperimentConfig,
    trial_id: str,
    recipe: RecipeCandidate,
    perf: TrainingPerfCandidate,
    max_steps: int,
    run_id: str,
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
        rdzv_dir = f"{trial_out_path}/rdzv"
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"mkdir -p {rdzv_dir}",
                'NODE_RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"',
                'if [ "$NODE_RANK" = "0" ]; then hostname -i | awk \'{print $1}\' > ' + f"{rdzv_dir}/master_ip; fi",
                f"while [ ! -s {rdzv_dir}/master_ip ]; do sleep 1; done",
                f"MASTER_ADDR=$(cat {rdzv_dir}/master_ip)",
                "torchrun \\",
                f"  --nnodes={perf.nodes} \\",
                f"  --nproc_per_node={cluster.nodes.gpus_per_node} \\",
                '  --node_rank="$NODE_RANK" \\',
                '  --rdzv_id="' + trial_id + '" \\',
                "  --rdzv_backend=c10d \\",
                '  --rdzv_endpoint="${MASTER_ADDR}:29500" \\',
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

    job_spec: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **({"files": {"/opt/crucible_remote_train.py": "file://crucible-train-runner"}} if cluster.fuzzfile_version == "v4" else {}),
        "env": [
            "PYTHONUNBUFFERED=1",
            "NCCL_IB_DISABLE=0",
            "CUDA_DEVICE_MAX_CONNECTIONS=1",
            "TORCH_CUDA_ARCH_LIST=12.0;12.1",
        ],
        "resource": _gpu_resource_block(cluster),
        "policy": {"timeout": {"execute": "2h"}},
        "script": script + "\n",
    }
    if perf.nodes > 1:
        job_spec["multinode"] = {
            "nodes": perf.nodes,
            "implementation": "generic",
            "procs-per-node": 1,
        }

    rel_runs_prefix = cluster.vast.runs_dir.removeprefix(cluster.vast.mount_path).lstrip("/")
    egress_rel_summary = f"{rel_runs_prefix}/{run_id}/trials/{trial_id}/summary.json"
    obj_uri = trial_metrics_object_uri(cluster, run_id, trial_id)

    doc: dict[str, Any] = {
        "version": cluster.fuzzfile_version,
        **_workflow_annotations_section(cluster),
        **({"files": {"crucible-train-runner": REMOTE_TRAIN_RUNNER_PY}} if cluster.fuzzfile_version == "v4" else {}),
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_summary, obj_uri)]),
        "jobs": {f"train-{trial_id}": job_spec},
    }
    header = (
        f"# Crucible Autoresearch Trial Fuzzfile: {trial_id}\n"
        f"# DefinitionId: {cluster.definition_id} | Nodes: {_team_nodes_comment(cluster)}\n"
        f"# Model: {exp.model.name} | Target Hardware: {cluster.nodes.gpu_product} ({cluster.nodes.sm_arch}, {cluster.nodes.arch_device})\n"
        f"# Recipe: rank={recipe.lora_rank}, alpha={recipe.lora_alpha}, lr={recipe.lr}\n"
        f"# Perf: nodes={perf.nodes}, micro_batch={perf.micro_batch_size}, "
        f"chunked_ce={perf.use_chunked_ce_kernel}, grouped_moe={perf.use_grouped_moe_kernel}, "
        f"fused_lora={perf.use_fused_lora_kernel}\n"
        f"# Metrics Object Cache Egress (numeric only, zero raw data): {obj_uri}\n\n"
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
    base_model_path = f"{cluster.vast.models_dir}/{exp.model.weights_subpath}"
    processed_path = f"{cluster.vast.datasets_dir}/{exp.dataset.processed_subpath}"
    out_json = f"{cluster.vast.runs_dir}/{run_id}/serving/{trial_id}/summary.json"
    vllm_cmd = build_vllm_launch_args(base_model_path, adapter_vast_path, cand, port=8000)

    service_name = f"vllm-{trial_id}"
    bench_job_name = f"bench-{trial_id}"
    cand_json = json.dumps(cand.model_dump(), sort_keys=True)

    vllm_shell = " ".join(vllm_cmd)
    svc_spec: dict[str, Any] = {
        "image": {"uri": cluster.images.serve},
        "mounts": _mounts_section(cluster),
        "env": [
            "PYTHONUNBUFFERED=1",
            "FLASHINFER_DISABLE_VERSION_CHECK=1",
            f"CRUCIBLE_USE_LOW_PREC_DECODE={int(cand.use_low_prec_decode_kernel)}",
        ],
        "script": (
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f'if [ -f "{base_model_path}/config.json" ]; then\n'
            f"  exec {vllm_shell}\n"
            "else\n"
            "  exec python3 -m http.server 8000\n"
            "fi\n"
        ),
        "resource": _gpu_resource_block(
            cluster, gpus_per_node=min(cand.tensor_parallel, cluster.nodes.gpus_per_node)
        ),
    }
    if cand.tensor_parallel > cluster.nodes.gpus_per_node:
        svc_spec["multinode"] = {
            "nodes": cand.tensor_parallel // cluster.nodes.gpus_per_node,
            "implementation": "generic",
        }

    bench_entry = (
        "/opt/crucible_remote_serve_bench.py"
        if cluster.fuzzfile_version == "v4"
        else "-m crucible.serving.vllm_bench"
    )
    # 2 cores / 8GiB fits inside the 4-core / 15.9-GiB headroom left on SPARK39/SPARK40
    bench_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **({"files": {"/opt/crucible_remote_serve_bench.py": "file://crucible-serve-runner"}} if cluster.fuzzfile_version == "v4" else {}),
        **_dependency_field(cluster, [(service_name, "RUNNING")]),
        "resource": _cpu_helper_resource_block(cluster, cores=2, mem_gib=8, affinity="CORE"),
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"python3 {bench_entry} \\",
                f"  --endpoint http://{service_name}:8000 \\",
                f"  --processed-data-dir {processed_path} \\",
                f"  --trial-id {trial_id} \\",
                f"  --baseline-loss {baseline_loss:.6f} \\",
                f"  --output-json {out_json} \\",
                f"  --candidate-json '{cand_json}'",
            ]
        )
        + "\n",
    }

    rel_runs_prefix = cluster.vast.runs_dir.removeprefix(cluster.vast.mount_path).lstrip("/")
    egress_rel_srv = f"{rel_runs_prefix}/{run_id}/serving/{trial_id}/summary.json"
    srv_obj_uri = trial_metrics_object_uri(cluster, run_id, f"serving-{trial_id}")

    doc: dict[str, Any] = {
        "version": cluster.fuzzfile_version,
        **_workflow_annotations_section(cluster),
        **({"files": {"crucible-serve-runner": REMOTE_SERVE_BENCH_PY}} if cluster.fuzzfile_version == "v4" else {}),
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_srv, srv_obj_uri)]),
        "services": {service_name: svc_spec},
        "jobs": {bench_job_name: bench_job},
    }
    header = (
        f"# Crucible Inference Search Fuzzfile: {trial_id}\n"
        f"# DefinitionId: {cluster.definition_id} | Nodes: {_team_nodes_comment(cluster)}\n"
        f"# Precision: {cand.precision} | TP={cand.tensor_parallel} | Replicas={cand.replicas} | "
        f"MaxSeqs={cand.max_num_seqs}\n"
        f"# Metrics Object Cache Egress (numeric only, zero raw data): {srv_obj_uri}\n\n"
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
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        "env": [
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={crucible_src_dir}/src",
        ],
        "resource": _cpu_helper_resource_block(cluster, cores=8, mem_gib=32, affinity="NUMA"),
        "policy": {"timeout": {"execute": "30m"}},
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f'if ! python3 -c "import crucible" 2>/dev/null && [ ! -d "{crucible_src_dir}/src/crucible" ]; then',
                f'  git clone --depth 1 https://github.com/sebastianlueders/CIQ-Hackathon.git "{crucible_src_dir}"',
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
        rdzv_dir = f"{prod_out_dir}/rdzv"
        train_script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"mkdir -p {rdzv_dir}",
                'NODE_RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"',
                'if [ "$NODE_RANK" = "0" ]; then hostname -i | awk \'{print $1}\' > ' + f"{rdzv_dir}/master_ip; fi",
                f"while [ ! -s {rdzv_dir}/master_ip ]; do sleep 1; done",
                f"MASTER_ADDR=$(cat {rdzv_dir}/master_ip)",
                "torchrun \\",
                f"  --nnodes={winning_train_perf.nodes} \\",
                f"  --nproc_per_node={cluster.nodes.gpus_per_node} \\",
                '  --node_rank="$NODE_RANK" \\',
                '  --rdzv_id="crucible-prod-train" \\',
                "  --rdzv_backend=c10d \\",
                '  --rdzv_endpoint="${MASTER_ADDR}:29500" \\',
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
            "PYTHONUNBUFFERED=1",
            f"PYTHONPATH={crucible_src_dir}/src",
            "NCCL_IB_DISABLE=0",
            "CUDA_DEVICE_MAX_CONNECTIONS=1",
            "TORCH_CUDA_ARCH_LIST=12.0;12.1",
        ],
        "resource": _gpu_resource_block(cluster),
        "policy": {"timeout": {"execute": "4h"}},
        "script": train_script + "\n",
    }
    if winning_train_perf.nodes > 1:
        train_job["multinode"] = {
            "nodes": winning_train_perf.nodes,
            "implementation": "generic",
            "procs-per-node": 1,
        }

    eval_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **_dependency_field(cluster, [("train-lora", "FINISHED")]),
        "env": ["PYTHONUNBUFFERED=1"],
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
            "PYTHONUNBUFFERED=1",
            "FLASHINFER_DISABLE_VERSION_CHECK=1",
            f"CRUCIBLE_USE_LOW_PREC_DECODE={int(winning_serve_perf.use_low_prec_decode_kernel)}",
            f"CRUCIBLE_TARGET_REPLICAS={winning_serve_perf.replicas}",
        ],
        "command": vllm_cmd,
        "resource": _gpu_resource_block(
            cluster,
            gpus_per_node=min(winning_serve_perf.tensor_parallel, cluster.nodes.gpus_per_node),
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
# Cluster        : {cluster.cluster_name} | definitionId: {cluster.definition_id}
# Nodes          : {_team_nodes_comment(cluster)}
# Node Hardware  : 1x {cluster.nodes.gpu_product} ({cluster.nodes.gpu_arch}, {cluster.nodes.sm_arch}), {cluster.nodes.physical_cores_per_node}c ({cluster.nodes.arch_device}), {cluster.nodes.physical_memory_gib_per_node} GiB/node
# Per-Node Alloc : {cluster.nodes.cpu_cores_per_node} cores, {cluster.nodes.memory_gb_per_node}GiB RAM, {cluster.nodes.gpus_per_node}x nvidia.com/gpu
# Shared Storage : VAST persistent volume '{cluster.vast.persistent_volume_name}' mounted at {cluster.vast.mount_path}
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
#   fuzzball workflow start final-crucible-workflow.yaml -n {exp.experiment_id}-team20
# ==============================================================================

"""
    return comment_header + yaml.safe_dump(doc, sort_keys=False)
