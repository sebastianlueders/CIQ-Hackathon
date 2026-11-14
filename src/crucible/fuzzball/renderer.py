"""Renders trials and the final turnkey workflow into CIQ Fuzzball YAML (`v4` and `v1`)."""

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
from crucible.serving.vllm_bench import build_vllm_launch_args


def trial_metrics_object_uri(cluster: ClusterConfig, run_id: str, trial_id: str) -> str:
    return f"fuzzball://{cluster.vast.object_cache_scope}/crucible/{run_id}/{trial_id}/metrics.json"


def _volume_section(
    cluster: ClusterConfig,
    egress_entries: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if cluster.fuzzfile_version == "v4":
        vol_spec: dict[str, Any] = {
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
            "reference": f"volume://user/{cluster.vast.provisioner}/{cluster.vast.persistent_volume_name}"
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


def _gpu_resource_block(cluster: ClusterConfig, gpus_per_node: int | None = None) -> dict[str, Any]:
    gpus = gpus_per_node if gpus_per_node is not None else cluster.nodes.gpus_per_node
    return {
        "cpu": {
            "cores": cluster.nodes.cpu_cores_per_node,
            "affinity": "NUMA",
        },
        "memory": {
            "size": f"{cluster.nodes.memory_gb_per_node}GB",
        },
        "devices": {
            "nvidia.com/gpu": gpus,
        },
        "exclusive": True,
    }


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

    if perf.nodes > 1:
        rdzv_dir = f"{trial_out_path}/rdzv"
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"mkdir -p {rdzv_dir}",
                'NODE_RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"',
                'if [ "$NODE_RANK" = "0" ]; then hostname -i | awk \'{print $1}\' > ' + f"{rdzv_dir}/master_ip; fi",
                f'while [ ! -s {rdzv_dir}/master_ip ]; do sleep 1; done',
                f'MASTER_ADDR=$(cat {rdzv_dir}/master_ip)',
                "torchrun \\",
                f"  --nnodes={perf.nodes} \\",
                f"  --nproc_per_node={cluster.nodes.gpus_per_node} \\",
                '  --node_rank="$NODE_RANK" \\',
                '  --rdzv_id="' + trial_id + '" \\',
                "  --rdzv_backend=c10d \\",
                '  --rdzv_endpoint="${MASTER_ADDR}:29500" \\',
                "  -m crucible.trainer.lora_trainer \\",
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
                "python3 -m crucible.trainer.lora_trainer \\",
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
        "env": [
            "PYTHONUNBUFFERED=1",
            "NCCL_IB_DISABLE=0",
            "CUDA_DEVICE_MAX_CONNECTIONS=1",
            f"TORCH_CUDA_ARCH_LIST=12.1",
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
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_summary, obj_uri)]),
        "jobs": {f"train-{trial_id}": job_spec},
    }
    header = (
        f"# Crucible Autoresearch Trial Fuzzfile: {trial_id}\n"
        f"# Model: {exp.model.name} | Target Hardware: {cluster.nodes.gpu_model} ({cluster.nodes.sm_arch})\n"
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

    svc_spec: dict[str, Any] = {
        "image": {"uri": cluster.images.serve},
        "mounts": _mounts_section(cluster),
        "env": [
            "PYTHONUNBUFFERED=1",
            f"CRUCIBLE_USE_LOW_PREC_DECODE={int(cand.use_low_prec_decode_kernel)}",
        ],
        "command": vllm_cmd,
        "resource": _gpu_resource_block(cluster, gpus_per_node=min(cand.tensor_parallel, cluster.nodes.gpus_per_node)),
    }
    if cand.tensor_parallel > cluster.nodes.gpus_per_node:
        svc_spec["multinode"] = {
            "nodes": cand.tensor_parallel // cluster.nodes.gpus_per_node,
            "implementation": "generic",
        }

    bench_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        **_dependency_field(cluster, [(service_name, "RUNNING")]),
        "resource": {
            "cpu": {"cores": 4, "affinity": "CORE"},
            "memory": {"size": "16GB"},
        },
        "script": "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "python3 -m crucible.serving.vllm_bench \\",
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
        "volumes": _volume_section(cluster, egress_entries=[(egress_rel_srv, srv_obj_uri)]),
        "services": {service_name: svc_spec},
        "jobs": {bench_job_name: bench_job},
    }
    header = (
        f"# Crucible Inference Search Fuzzfile: {trial_id}\n"
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

    ingest_job: dict[str, Any] = {
        "image": {"uri": cluster.images.train},
        "mounts": _mounts_section(cluster),
        "env": ["PYTHONUNBUFFERED=1"],
        "resource": {
            "cpu": {"cores": 8, "affinity": "NUMA"},
            "memory": {"size": "32GB"},
        },
        "policy": {"timeout": {"execute": "30m"}},
        "command": [
            "crucible",
            "ingest",
            "--source-dir",
            raw_data_path,
            "--output-dir",
            processed_path,
            "--seq-len",
            str(winning_recipe.seq_len),
        ],
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
            "NCCL_IB_DISABLE=0",
            "CUDA_DEVICE_MAX_CONNECTIONS=1",
            "TORCH_CUDA_ARCH_LIST=12.1",
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
# Cluster        : {cluster.cluster_name} ({cluster.nodes.count}x {cluster.nodes.gpu_model}, {cluster.nodes.sm_arch})
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
# Launch end-to-end on CIQ Fuzzball:
#   fuzzball workflow validate final-crucible-workflow.yaml
#   fuzzball workflow start final-crucible-workflow.yaml -n {exp.experiment_id}-turnkey
# ==============================================================================

"""
    return comment_header + yaml.safe_dump(doc, sort_keys=False)
