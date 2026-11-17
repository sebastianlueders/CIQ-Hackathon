"""Sealed held-out C++ completion suite extraction for code-quality floor gating."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from crucible.data.lammps_ingest import SourceDocument

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
CPP_STOPWORDS = frozenset(
    {
        "const",
        "double",
        "int",
        "void",
        "for",
        "return",
        "std",
        "vector",
        "size_t",
        "include",
        "namespace",
        "class",
        "public",
        "private",
        "auto",
        "static_cast",
    }
)


@dataclass(frozen=True)
class CompletionTask:
    task_id: str
    rel_path: str
    prompt_prefix: str
    reference_completion: str
    expected_identifiers: list[str]


def build_held_out_completion_suite(
    held_out_docs: list[SourceDocument],
    max_tasks: int = 24,
) -> list[CompletionTask]:
    tasks: list[CompletionTask] = []
    for doc in held_out_docs:
        lines = [ln for ln in doc.content.splitlines() if ln.strip()]
        if len(lines) < 6:
            continue
        split_idx = max(3, int(len(lines) * 0.6))
        prefix = "\n".join(lines[:split_idx]) + "\n"
        completion = "\n".join(lines[split_idx:])
        raw_idents = IDENTIFIER_RE.findall(completion)
        seen: list[str] = []
        for ident in raw_idents:
            if ident not in CPP_STOPWORDS and ident not in seen:
                seen.append(ident)
            if len(seen) >= 6:
                break
        if not seen:
            seen = raw_idents[:3]
        tasks.append(
            CompletionTask(
                task_id=f"holdout-{len(tasks):03d}",
                rel_path=doc.rel_path,
                prompt_prefix=prefix,
                reference_completion=completion,
                expected_identifiers=seen,
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def load_held_out_documents(processed_dir: Path) -> list[SourceDocument]:
    path = processed_dir / "held_out_documents.jsonl"
    if not path.exists():
        return []
    docs: list[SourceDocument] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(SourceDocument(**json.loads(line)))
    return docs
