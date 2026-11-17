"""Held-out NLL/perplexity evaluator and C++ completion quality floor gate."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from crucible.data.completion_tasks import CompletionTask


@dataclass(frozen=True)
class CompletionCaseScore:
    task_id: str
    identifier_recall: float
    syntax_balanced: bool
    passed: bool


@dataclass(frozen=True)
class QualityEvalReport:
    held_out_loss: float
    held_out_perplexity: float
    baseline_perplexity: float
    perplexity_regression_pct: float
    completion_pass_rate: float
    mean_identifier_recall: float
    passed_quality_floor: bool
    rejection_reasons: list[str]


def check_cpp_delimiter_balance(prefix: str, generated: str) -> bool:
    combined = prefix + generated
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}
    stack: list[str] = []
    in_line_comment = False
    i = 0
    while i < len(combined):
        ch = combined[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if ch == "/" and i + 1 < len(combined) and combined[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False
        i += 1
    return len(stack) == 0


def score_completion_predictions(
    tasks: list[CompletionTask],
    predictions: dict[str, str],
    min_identifier_recall: float = 0.5,
) -> tuple[float, float, list[CompletionCaseScore]]:
    if not tasks:
        return 1.0, 1.0, []
    scores: list[CompletionCaseScore] = []
    for task in tasks:
        pred = predictions.get(task.task_id, "")
        if task.expected_identifiers:
            hits = sum(1 for ident in task.expected_identifiers if ident in pred)
            recall = hits / len(task.expected_identifiers)
        else:
            recall = 1.0 if pred.strip() else 0.0
        balanced = check_cpp_delimiter_balance(task.prompt_prefix, pred)
        passed = balanced and (recall >= min_identifier_recall)
        scores.append(
            CompletionCaseScore(
                task_id=task.task_id,
                identifier_recall=round(recall, 4),
                syntax_balanced=balanced,
                passed=passed,
            )
        )
    pass_rate = sum(1 for s in scores if s.passed) / len(scores)
    mean_recall = sum(s.identifier_recall for s in scores) / len(scores)
    return round(pass_rate, 4), round(mean_recall, 4), scores


def evaluate_quality_floor(
    held_out_loss: float,
    baseline_finetuned_loss: float,
    completion_pass_rate: float,
    mean_identifier_recall: float,
    max_perplexity_regression_pct: float = 2.5,
    min_completion_pass_rate: float = 0.84,
) -> QualityEvalReport:
    ppl = math.exp(min(held_out_loss, 20.0))
    base_ppl = math.exp(min(baseline_finetuned_loss, 20.0))
    regression_pct = ((ppl - base_ppl) / max(base_ppl, 1e-8)) * 100.0

    reasons: list[str] = []
    if regression_pct > max_perplexity_regression_pct:
        reasons.append(
            f"perplexity_regression={regression_pct:.2f}% exceeds ceiling {max_perplexity_regression_pct:.2f}%"
        )
    if completion_pass_rate < min_completion_pass_rate:
        reasons.append(
            f"completion_pass_rate={completion_pass_rate:.3f} below floor {min_completion_pass_rate:.3f}"
        )

    return QualityEvalReport(
        held_out_loss=round(held_out_loss, 5),
        held_out_perplexity=round(ppl, 4),
        baseline_perplexity=round(base_ppl, 4),
        perplexity_regression_pct=round(regression_pct, 3),
        completion_pass_rate=round(completion_pass_rate, 4),
        mean_identifier_recall=round(mean_identifier_recall, 4),
        passed_quality_floor=(len(reasons) == 0),
        rejection_reasons=reasons,
    )


def compute_torch_causal_eval_loss(
    forward_logits_fn: Callable[[list[list[int]]], "object"],
    held_out_jsonl: Path,
    max_batches: int = 16,
    batch_size: int = 2,
) -> float:
    """Evaluate exact token-level cross-entropy loss over packed held-out JSONL chunks."""
    import torch
    import torch.nn.functional as F

    rows: list[list[int]] = []
    for line in held_out_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if len(obj["input_ids"]) >= 4:
                rows.append(obj["input_ids"])
    if not rows:
        raise ValueError(f"No valid evaluation chunks in {held_out_jsonl}")

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b_idx in range(0, min(len(rows), max_batches * batch_size), batch_size):
            batch = rows[b_idx : b_idx + batch_size]
            min_len = min(len(seq) for seq in batch)
            trimmed = [seq[:min_len] for seq in batch]
            logits = forward_logits_fn(trimmed)
            targets = torch.tensor([seq[1:] for seq in trimmed], dtype=torch.long, device=logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(targets.numel())
    return total_nll / max(1, total_tokens)


def save_quality_report(report: QualityEvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
