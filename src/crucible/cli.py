"""Alchemy command-line interface (`alchemy`)."""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path

from crucible.config import ClusterConfig, DatasetSplitConfig, ExperimentConfig, repo_root
from crucible.data.lammps_ingest import ingest_lammps_corpus
from crucible.fuzzball.discovery import discover_cluster_config
from crucible.fuzzball.validator import validate_fuzzfile
from crucible.kernels.benchmark import run_kernel_suite, verify_training_loss_curve_equivalence
from crucible.optimizer.pipeline import run_autoresearch_pipeline


def _cmd_ingest(args: argparse.Namespace) -> int:
    split_cfg = DatasetSplitConfig(
        chunk_seq_len=args.seq_len,
        chunk_overlap=args.overlap,
    )
    manifest = ingest_lammps_corpus(
        source_root=args.source_dir,
        output_dir=args.output_dir,
        split_cfg=split_cfg,
        vocab_size=args.vocab_size,
    )
    print(f"Ingested {manifest.total_cpp_files} C++/CUDA files -> {args.output_dir}")
    print(f"  File splits  : {manifest.split_file_counts}")
    print(f"  Chunk splits : {manifest.split_chunk_counts}")
    print(f"  Sealed files : {manifest.held_out_files}")
    print(f"  Manifest SHA : {manifest.manifest_sha256}")
    return 0


def _cmd_bench_kernels(args: argparse.Namespace) -> int:
    caps, results = run_kernel_suite(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_tokens=args.num_tokens,
    )
    print(
        f"Detected runtime: device={caps.device_name} ({caps.sm_arch}) | "
        f"backend={caps.active_kernel_backend}"
    )
    all_ok = True
    for r in results:
        status = "PASS" if r.correctness_passed else "FAIL"
        all_ok = all_ok and r.correctness_passed
        print(
            f"  [{status}] {r.kernel_name:24s} | max_err={r.max_abs_error:.2e} | "
            f"mem_saving={r.memory_reduction_ratio:.2f}x | ref={r.reference_ms:.3f}ms vs custom={r.custom_ms:.3f}ms"
        )

    base_curve, custom_curve, max_diff = verify_training_loss_curve_equivalence(steps=10)
    curve_ok = max_diff < 1e-4
    all_ok = all_ok and curve_ok
    print(
        f"  [{'PASS' if curve_ok else 'FAIL'}] multi-step loss curve equivalence | "
        f"max_step_diff={max_diff:.2e} ({base_curve[0]:.4f} -> {base_curve[-1]:.4f})"
    )
    return 0 if all_ok else 1


def _cmd_discover(args: argparse.Namespace) -> int:
    if getattr(args, "dump_mcp_tools", False):
        from crucible.fuzzball.mcp_client import FuzzballMCPStdioClient

        with FuzzballMCPStdioClient(allow_write=True, allow_exec=True) as mcp:
            print(json.dumps(mcp.dump_tool_catalog(), indent=2, sort_keys=True))
        return 0

    base_cfg = ClusterConfig.load(args.cluster_config)
    discovered_cfg, report = discover_cluster_config(
        base_config=base_cfg,
        fetch_workflow_path=args.fetch_workflow,
        mcp_snapshot_path=args.mcp_snapshot,
    )
    if args.write_config is not None:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(
            yaml.safe_dump(discovered_cfg.model_dump(), sort_keys=False),
            encoding="utf-8",
        )
    if args.json:
        print(
            json.dumps(
                {
                    "cluster_config": discovered_cfg.model_dump(),
                    "discovery_report": report.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"Alchemy Cluster Discovery -> cluster='{discovered_cfg.cluster_name}' "
        f"| definitionId='{discovered_cfg.definition_id}' (Fuzzfile {discovered_cfg.fuzzfile_version})"
    )
    print(f"  fuzzball CLI on PATH : {report.fuzzball_cli_available}")
    print(f"  MCP source           : {report.mcp_source_used or 'none (using 00-fetch + config)'}")
    print(f"  00-fetch inspected   : {report.fetch_workflow_inspected or 'not found'}")
    print(
        f"  Active VAST volume   : {discovered_cfg.vast.persistent_volume_name} "
        f"(provisioner={discovered_cfg.vast.provisioner}, mount={discovered_cfg.vast.mount_path})"
    )
    inv_str = ", ".join(f"{n.hostname} ({n.node_id})" for n in discovered_cfg.nodes.inventory)
    print(
        f"  Nodes configured     : {discovered_cfg.nodes.count}x {discovered_cfg.nodes.gpu_product} "
        f"({discovered_cfg.nodes.sm_arch}, {discovered_cfg.nodes.arch_device}) -> [{inv_str}]"
    )
    print(
        f"  Safe per-node alloc  : {discovered_cfg.nodes.cpu_cores_per_node}/{discovered_cfg.nodes.physical_cores_per_node} cores, "
        f"{discovered_cfg.nodes.memory_gb_per_node}/{discovered_cfg.nodes.physical_memory_gib_per_node} GiB, "
        f"{discovered_cfg.nodes.gpus_per_node}x nvidia.com/gpu"
    )
    if report.applied_overrides:
        print(f"  Applied overrides    : {report.applied_overrides}")
    if args.write_config is not None:
        print(f"  Wrote resolved YAML  : {args.write_config}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    exp = ExperimentConfig.load(args.config)
    base_cluster = ClusterConfig.load(args.cluster_config)
    if args.no_auto_discover:
        cluster = base_cluster
    else:
        cluster, _ = discover_cluster_config(
            base_config=base_cluster,
            mcp_snapshot_path=args.mcp_snapshot,
        )
    if args.fuzzfile_version:
        cluster.fuzzfile_version = args.fuzzfile_version

    result = run_autoresearch_pipeline(
        exp=exp,
        cluster=cluster,
        output_dir=args.output_dir,
        backend_kind=args.backend,
        source_data_dir=args.source_data_dir,
    )
    print(f"Alchemy 4-stage run complete -> {result.output_dir}")
    print(
        f"  Stage A (Recipe)       : {result.stage_a.winning_label} | "
        f"Held-out PPL {result.stage_a.base_held_out_perplexity:.3f} -> "
        f"{result.stage_a.winning_summary.held_out_perplexity:.3f}"
    )
    print(
        f"  Stage B (Train Perf)   : {result.stage_b.winning_candidate.label} | "
        f"{result.stage_b.overall_speedup_x:.2f}x faster "
        f"({result.stage_b.single_node_kernel_speedup_x:.2f}x from sm_121 kernels on 1 node)"
    )
    print(
        f"  Stage C (Serve Perf)   : {result.stage_c.winning_candidate_label} | "
        f"{result.stage_c.winning_summary.throughput_tokens_per_sec:,.1f} tok/s "
        f"({result.stage_c.serving_speedup_x:.2f}x vs BF16)"
    )
    print(f"  Stage D (Turnkey YAML) : {result.workflow_path}")
    print(f"  Measured Report        : {result.report_path}")
    print(f"  Privacy Audit Log      : {result.audit_log_path}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    summary_path = args.run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No run_summary.json found in {args.run_dir}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Run ID        : {data['run_id']} (backend={data['backend']})")
    print(f"Stage A Winner: {data['stage_a']['winning_label']}")
    print(f"Stage B Winner: {data['stage_b']['winning_label']} ({data['stage_b']['overall_speedup_x']}x)")
    print(f"Stage C Winner: {data['stage_c']['winning_label']} ({data['stage_c']['serving_speedup_x']}x)")
    print(f"Firewall Log  : {len(data['privacy_audit'])} verified zero-raw-data advisor records")
    return 0


def _cmd_validate_yaml(args: argparse.Namespace) -> int:
    cluster = ClusterConfig.load(args.cluster_config)
    stages = validate_fuzzfile(args.fuzzfile, cluster=cluster)
    print(
        f"VALID CIQ Fuzzfile ({args.fuzzfile}) for definitionId='{cluster.definition_id}': stages={stages}"
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from crucible.server.app import create_app

    app = create_app(run_dir=args.run_dir)
    print(f"Alchemy Team 3 Live Studio -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alchemy",
        description="Alchemy: Privacy-preserving autoresearch & sm_121 performance agent for CIQ Fuzzball.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest LAMMPS C++ repo into sealed whole-file splits")
    p_ing.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root() / "fixtures" / "sample_lammps",
    )
    p_ing.add_argument("--output-dir", type=Path, required=True)
    p_ing.add_argument("--seq-len", type=int, default=2048)
    p_ing.add_argument("--overlap", type=int, default=128)
    p_ing.add_argument("--vocab-size", type=int, default=151936)
    p_ing.set_defaults(func=_cmd_ingest)

    p_ker = sub.add_parser("bench-kernels", help="Verify & benchmark custom sm_121 kernels")
    p_ker.add_argument("--vocab-size", type=int, default=16384)
    p_ker.add_argument("--hidden-size", type=int, default=256)
    p_ker.add_argument("--num-tokens", type=int, default=128)
    p_ker.set_defaults(func=_cmd_bench_kernels)

    p_dis = sub.add_parser(
        "discover",
        help="Auto-discover active CIQ Fuzzball volumes, workflows (00-fetch.yaml), and nodes via CLI/MCP",
    )
    p_dis.add_argument(
        "--cluster-config",
        type=Path,
        default=repo_root() / "configs" / "cluster.default.yaml",
    )
    p_dis.add_argument(
        "--fetch-workflow",
        type=Path,
        default=repo_root() / "workflows" / "00-fetch.yaml",
    )
    p_dis.add_argument("--mcp-snapshot", type=Path, default=None)
    p_dis.add_argument("--write-config", type=Path, default=None)
    p_dis.add_argument("--json", action="store_true")
    p_dis.add_argument(
        "--dump-mcp-tools",
        action="store_true",
        help="Connect to 'fuzzball mcp serve' and dump exact advertised tool inputSchemas as JSON",
    )
    p_dis.set_defaults(func=_cmd_discover)

    p_run = sub.add_parser("run", help="Execute the 4-stage autoresearch loop and emit turnkey Fuzzfile")
    p_run.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "demo_lammps_30b.yaml",
    )
    p_run.add_argument(
        "--cluster-config",
        type=Path,
        default=repo_root() / "configs" / "cluster.default.yaml",
    )
    p_run.add_argument(
        "--backend",
        choices=["simulator", "local_torch", "fuzzball_mcp", "fuzzball"],
        default="simulator",
    )
    p_run.add_argument("--fuzzfile-version", choices=["v4", "v1"], default=None)
    p_run.add_argument("--mcp-snapshot", type=Path, default=None)
    p_run.add_argument("--no-auto-discover", action="store_true")
    p_run.add_argument("--source-data-dir", type=Path, default=None)
    p_run.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    p_run.set_defaults(func=_cmd_run)

    p_ins = sub.add_parser("inspect", help="Inspect results of a completed Alchemy run directory")
    p_ins.add_argument("run_dir", type=Path)
    p_ins.set_defaults(func=_cmd_inspect)

    p_val = sub.add_parser("validate-yaml", help="Validate a CIQ Fuzzball workflow YAML file")
    p_val.add_argument("fuzzfile", type=Path)
    p_val.add_argument(
        "--cluster-config",
        type=Path,
        default=repo_root() / "configs" / "cluster.default.yaml",
    )
    p_val.set_defaults(func=_cmd_validate_yaml)

    p_srv = sub.add_parser(
        "serve",
        help="Launch the Alchemy Team 3 Live Web Studio (default http://127.0.0.1:8000)",
    )
    p_srv.add_argument("--run-dir", type=Path, default=Path("runs/team3-live"))
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.set_defaults(func=_cmd_serve)

    p_ui = sub.add_parser("ui", help="Alias for 'alchemy serve'")
    p_ui.add_argument("--run-dir", type=Path, default=Path("runs/team3-live"))
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8000)
    p_ui.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
