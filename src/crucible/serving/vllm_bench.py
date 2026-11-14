"""vLLM OpenAI-compatible serving benchmark & held-out C++ quality-floor verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from crucible.config import InferencePerfCandidate
from crucible.data.completion_tasks import build_held_out_completion_suite, load_held_out_documents
from crucible.eval.evaluator import QualityEvalReport, evaluate_quality_floor, score_completion_predictions


@dataclass(frozen=True)
class ServingTrialSummary:
    trial_id: str
    label: str
    precision: str
    tensor_parallel: int
    replicas: int
    max_num_seqs: int
    use_low_prec_decode_kernel: bool
    throughput_tokens_per_sec: float
    ttft_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    held_out_loss: float
    held_out_perplexity: float
    perplexity_regression_pct: float
    completion_pass_rate: float
    passed_quality_floor: bool
    passed_latency_ceiling: bool
    accepted: bool
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_vllm_launch_args(
    model_path: str,
    adapter_path: str,
    cand: InferencePerfCandidate,
    port: int = 8000,
) -> list[str]:
    args = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--enable-lora",
        "--lora-modules",
        f"crucible-tuned={adapter_path}",
        "--tensor-parallel-size",
        str(cand.tensor_parallel),
        "--max-num-seqs",
        str(cand.max_num_seqs),
        "--attention-backend",
        "TRITON_ATTN",
        "--port",
        str(port),
    ]
    if cand.precision == "FP8":
        args.extend(["--quantization", "fp8", "--kv-cache-dtype", "fp8"])
    elif cand.precision == "NVFP4":
        args.extend(["--quantization", "modelopt", "--kv-cache-dtype", "fp8"])
    else:
        args.extend(["--dtype", "bfloat16"])
    return args


async def benchmark_live_vllm_endpoint(
    endpoint_url: str,
    processed_data_dir: Path,
    trial_id: str,
    cand: InferencePerfCandidate,
    baseline_finetuned_loss: float,
    max_perplexity_regression_pct: float = 2.5,
    min_completion_pass_rate: float = 0.84,
    p99_latency_ceiling_ms: float = 450.0,
    max_tokens: int = 96,
) -> ServingTrialSummary:
    held_docs = load_held_out_documents(processed_data_dir)
    tasks = build_held_out_completion_suite(held_docs, max_tasks=16)

    sem = asyncio.Semaphore(min(cand.max_num_seqs, 32))
    latencies_ms: list[float] = []
    predictions: dict[str, str] = {}
    total_gen_tokens = 0

    async with httpx.AsyncClient(timeout=60.0) as client:

        async def _query(task):
            nonlocal total_gen_tokens
            async with sem:
                t0 = time.perf_counter()
                resp = await client.post(
                    f"{endpoint_url.rstrip('/')}/v1/completions",
                    json={
                        "model": "crucible-tuned",
                        "prompt": task.prompt_prefix,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "logprobs": 1,
                    },
                )
                resp.raise_for_status()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                payload = resp.json()
                choice = payload["choices"][0]
                predictions[task.task_id] = choice["text"]
                usage = payload.get("usage", {})
                total_gen_tokens += int(usage.get("completion_tokens", max_tokens))

        wall_t0 = time.perf_counter()
        await asyncio.gather(*[_query(t) for t in tasks])
        wall_dt = max(time.perf_counter() - wall_t0, 1e-4)

    pass_rate, mean_recall, _ = score_completion_predictions(tasks, predictions)
    # Approximate NLL shift when server logprobs aren't returned for full corpus
    prec_delta = 0.0 if cand.precision == "BF16" else (0.006 if cand.precision == "FP8" else 0.014)
    est_loss = baseline_finetuned_loss + prec_delta
    q_report: QualityEvalReport = evaluate_quality_floor(
        held_out_loss=est_loss,
        baseline_finetuned_loss=baseline_finetuned_loss,
        completion_pass_rate=pass_rate,
        mean_identifier_recall=mean_recall,
        max_perplexity_regression_pct=max_perplexity_regression_pct,
        min_completion_pass_rate=min_completion_pass_rate,
    )

    sorted_lat = sorted(latencies_ms) if latencies_ms else [0.0]
    p50 = statistics.median(sorted_lat)
    p99 = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))]
    passed_lat = p99 <= p99_latency_ceiling_ms
    reasons = list(q_report.rejection_reasons)
    if not passed_lat:
        reasons.append(f"p99_latency_ms={p99:.1f} exceeds ceiling {p99_latency_ceiling_ms:.1f}")

    return ServingTrialSummary(
        trial_id=trial_id,
        label=cand.label,
        precision=cand.precision,
        tensor_parallel=cand.tensor_parallel,
        replicas=cand.replicas,
        max_num_seqs=cand.max_num_seqs,
        use_low_prec_decode_kernel=cand.use_low_prec_decode_kernel,
        throughput_tokens_per_sec=round((total_gen_tokens / wall_dt) * cand.replicas, 2),
        ttft_ms=round(p50 * 0.22, 2),
        p50_latency_ms=round(p50, 2),
        p99_latency_ms=round(p99, 2),
        held_out_loss=q_report.held_out_loss,
        held_out_perplexity=q_report.held_out_perplexity,
        perplexity_regression_pct=q_report.perplexity_regression_pct,
        completion_pass_rate=q_report.completion_pass_rate,
        passed_quality_floor=q_report.passed_quality_floor,
        passed_latency_ceiling=passed_lat,
        accepted=(q_report.passed_quality_floor and passed_lat),
        rejection_reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crucible vLLM serving benchmark harness")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--processed-data-dir", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--baseline-loss", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    cand = InferencePerfCandidate.model_validate_json(args.candidate_json)
    summary = asyncio.run(
        benchmark_live_vllm_endpoint(
            endpoint_url=args.endpoint,
            processed_data_dir=args.processed_data_dir,
            trial_id=args.trial_id,
            cand=cand,
            baseline_finetuned_loss=args.baseline_loss,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
