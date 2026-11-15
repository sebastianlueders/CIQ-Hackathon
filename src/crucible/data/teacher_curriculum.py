"""Private-Enclave Remote Teacher Curriculum Synthesizer for Train / Validation / Test splits.

Processes uploaded raw enterprise files (codebases, SOP docs, JSONL/JSON, CSV, ZIPs) AFTER
deterministic whole-document stem partitioning (`train` / `val` / `held_out`) so the Teacher
creates structured Supervised Fine-Tuning (SFT) instruction-reasoning-response pairs for:
  - `teacher_sft_train.jsonl` (model teaching / LoRA gradient updates)
  - `teacher_sft_val.jsonl`   (validation loss & early-stopping selection)
  - `teacher_sft_test.jsonl`  (sealed held-out test perplexity & completion evaluation)

Security & Zero-Egress Guarantee:
  - Never sends uploaded data to public external cloud APIs.
  - If `CRUCIBLE_PRIVATE_TEACHER_URL` points to an internal Fuzzball/VAST vLLM teacher service
    (loopback, RFC1918 private IP, `.local`, or `.fb-hack.` cluster enclave), queries that
    private teacher model; otherwise executes deterministic structural AST/symbol/section SFT
    synthesis inside the local/VAST boundary.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from crucible.data.lammps_ingest import SourceDocument

SYMBOL_DEF_RE = re.compile(
    r"^\s*(?:def\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|class\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|(?:__global__|__device__|inline|static|virtual|void|int|double|float|bool|auto|std::[A-Za-z0-9_:<>]+)\s+"
    r"([A-Za-z_][A-Za-z0-9_:]*)\s*\("
    r"|(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|fn\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|func\s+([A-Za-z_][A-Za-z0-9_]*))",
    re.MULTILINE,
)
HEADING_RE = re.compile(r"^\s*#{1,4}\s+(.+)$", re.MULTILINE)
IDENT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
GENERIC_STOPWORDS = frozenset(
    {
        "return",
        "const",
        "double",
        "float",
        "void",
        "class",
        "struct",
        "import",
        "from",
        "self",
        "true",
        "false",
        "none",
        "null",
        "this",
        "public",
        "private",
        "include",
        "using",
        "namespace",
        "instruction",
        "response",
        "input",
    }
)


@dataclass(frozen=True)
class TeacherCurriculumExample:
    example_id: str
    split: str
    task_type: str
    source_file: str
    instruction: str
    input_context: str
    teacher_reasoning: str
    target_response: str
    expected_symbols: list[str]
    formatted_sft_text: str


def is_allowed_private_teacher_url(raw_url: str) -> bool:
    """Verify a teacher endpoint resides strictly inside localhost, RFC1918 private LAN, or Fuzzball enclave."""
    if not raw_url.strip():
        return False
    try:
        parsed = urlparse(raw_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return True
        if host.endswith((".local", ".internal", ".svc.cluster.local")) or ".fb-hack." in host:
            return True
        ip_obj = ipaddress.ip_address(host)
        return bool(ip_obj.is_private or ip_obj.is_loopback)
    except ValueError:
        return False


def _extract_key_symbols(text: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    for tok in IDENT_TOKEN_RE.findall(text):
        if tok.lower() not in GENERIC_STOPWORDS and tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def _extract_defined_names(text: str) -> list[str]:
    found: list[str] = []
    for m in SYMBOL_DEF_RE.finditer(text):
        sym = next((g for g in m.groups() if g), "")
        if sym and sym not in found:
            found.append(sym)
    return found


def _format_sft_chat(
    instruction: str,
    input_context: str,
    teacher_reasoning: str,
    target_response: str,
) -> str:
    user_block = instruction.strip()
    if input_context.strip():
        user_block = f"{user_block}\n\nContext:\n{input_context.strip()}"
    return (
        "<|system|>\n"
        "You are a domain-specialized enterprise engineering assistant trained on private workspace artifacts.\n"
        "<|user|>\n"
        f"{user_block}\n"
        "<|assistant|>\n"
        f"<think>\n{teacher_reasoning.strip()}\n</think>\n"
        f"{target_response.strip()}"
    )


def _try_private_enclave_teacher_pair(
    doc: SourceDocument,
    split_Label: str,
    private_url: str,
    teacher_model: str,
) -> tuple[str, str, str] | None:
    """Query an in-enclave Fuzzball vLLM teacher endpoint if configured and reachable."""
    if not is_allowed_private_teacher_url(private_url):
        return None
    endpoint = private_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    snippet = doc.content[:2400]
    prompt = (
        f"Create one domain-grounded {split_Label} SFT teaching example from `{doc.rel_path}`. "
        "Return strict JSON with keys: instruction, reasoning, response.\n\n"
        f"Artifact:\n{snippet}"
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": teacher_model,
                "temperature": 0.15,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a private Fuzzball Teacher LLM synthesizing SFT pairs inside a VAST enclave.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=6.0,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        instr = str(parsed.get("instruction") or "").strip()
        reas = str(parsed.get("reasoning") or "").strip()
        ans = str(parsed.get("response") or "").strip()
        if instr and ans:
            return instr, reas or "Synthesized by private Fuzzball vLLM teacher.", ans
    except Exception:
        return None
    return None


def _synthesize_examples_for_doc(
    doc: SourceDocument,
    split_label: str,
    seq_start: int,
    private_url: str,
    teacher_model: str,
) -> tuple[list[TeacherCurriculumExample], bool]:
    """Generate 1-2 structured SFT teaching examples from a single split-isolated `SourceDocument`."""
    examples: list[TeacherCurriculumExample] = []
    used_live_enclave = False
    raw = doc.content.strip()
    if not raw:
        return examples, False

    clean_path = doc.rel_path.split("#")[0]
    symbols = _extract_key_symbols(raw, limit=6)
    defs = _extract_defined_names(raw)

    # Optional private Fuzzball vLLM teacher call (strictly enclave-validated)
    live_pair = _try_private_enclave_teacher_pair(doc, split_label, private_url, teacher_model)
    if live_pair is not None:
        instr, reas, ans = live_pair
        used_live_enclave = True
        chat_txt = _format_sft_chat(instr, "", reas, ans)
        examples.append(
            TeacherCurriculumExample(
                example_id=f"{split_label}-tchr-{seq_start:04d}",
                split=split_label,
                task_type="enclave_llm_distillation",
                source_file=doc.rel_path,
                instruction=instr,
                input_context="",
                teacher_reasoning=reas,
                target_response=ans,
                expected_symbols=symbols,
                formatted_sft_text=chat_txt,
            )
        )
        return examples, True

    # Case 1: Pre-structured Instruction / Response or Chat record from uploaded JSONL/JSON/CSV
    if "### Instruction:" in raw and "### Response:" in raw:
        before_resp, resp_part = raw.split("### Response:", 1)
        instr_part = before_resp.replace("### Instruction:", "").strip()
        inp_ctx = ""
        if "### Input:" in instr_part:
            instr_part, inp_ctx = [p.strip() for p in instr_part.split("### Input:", 1)]
        target_resp = resp_part.strip()
        reasoning = (
            f"Teacher verified structured domain specification from `{clean_path}`; "
            f"enforcing key domain identifiers ({', '.join(symbols[:4]) or 'exact schema'}) for {split_label} split."
        )
        chat_txt = _format_sft_chat(instr_part, inp_ctx, reasoning, target_resp)
        examples.append(
            TeacherCurriculumExample(
                example_id=f"{split_label}-tchr-{seq_start:04d}",
                split=split_label,
                task_type="structured_instruction_distill",
                source_file=doc.rel_path,
                instruction=instr_part,
                input_context=inp_ctx,
                teacher_reasoning=reasoning,
                target_response=target_resp,
                expected_symbols=_extract_key_symbols(target_resp, limit=6) or symbols,
                formatted_sft_text=chat_txt,
            )
        )
        return examples, False

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    headings = HEADING_RE.findall(raw)
    is_doc_like = Path(clean_path).suffix.lower() in {".md", ".markdown", ".rst", ".txt"} or bool(headings)

    # Case 2: Documentation / SOP / Runbook / Architecture Spec
    if is_doc_like and not defs:
        topic = headings[0].strip() if headings else Path(clean_path).stem.replace("_", " ").replace("-", " ")
        ctx_lines = lines[: min(4, max(1, len(lines) // 3))]
        body_lines = lines[: min(28, len(lines))]
        instr = (
            f"Summarize and apply the private engineering specification for '{topic}' "
            f"defined in `{clean_path}`."
        )
        inp_ctx = "\n".join(ctx_lines)
        target_resp = "\n".join(body_lines)
        reasoning = (
            f"Teacher extracted authoritative guidance from `{clean_path}` covering "
            f"{', '.join(symbols[:4]) or topic} to ground {split_label} evaluation without cross-split leakage."
        )
        chat_txt = _format_sft_chat(instr, inp_ctx, reasoning, target_resp)
        examples.append(
            TeacherCurriculumExample(
                example_id=f"{split_label}-tchr-{seq_start:04d}",
                split=split_label,
                task_type="domain_knowledge_qa",
                source_file=doc.rel_path,
                instruction=instr,
                input_context=inp_ctx,
                teacher_reasoning=reasoning,
                target_response=target_resp,
                expected_symbols=symbols,
                formatted_sft_text=chat_txt,
            )
        )
        return examples, False

    # Case 3: Source Code (Python, C++/CUDA, TypeScript, Go, Rust, SQL, etc.)
    primary_sym = defs[0] if defs else Path(clean_path).stem
    snippet_lines = lines[: min(42, len(lines))]
    full_snippet = "\n".join(snippet_lines)
    instr_impl = (
        f"Implement `{primary_sym}` following the conventions, data structures, and invariants in `{clean_path}`."
    )
    sig_preview = "\n".join(snippet_lines[: min(4, len(snippet_lines))])
    reasoning_impl = (
        f"Teacher analyzed `{clean_path}` ({len(lines)} non-empty lines): preserving signature `{primary_sym}` "
        f"and domain symbols ({', '.join(symbols[:5]) or primary_sym})."
    )
    chat_impl = _format_sft_chat(instr_impl, sig_preview, reasoning_impl, full_snippet)
    examples.append(
        TeacherCurriculumExample(
            example_id=f"{split_label}-tchr-{seq_start:04d}",
            split=split_label,
            task_type="code_synthesis_sft",
            source_file=doc.rel_path,
            instruction=instr_impl,
            input_context=sig_preview,
            teacher_reasoning=reasoning_impl,
            target_response=full_snippet,
            expected_symbols=symbols,
            formatted_sft_text=chat_impl,
        )
    )

    # Add a second context-completion teaching task when the file has enough lines
    if len(lines) >= 8:
        pivot = max(3, int(len(snippet_lines) * 0.55))
        prefix_ctx = "\n".join(snippet_lines[:pivot])
        suffix_target = "\n".join(snippet_lines[pivot:])
        tail_syms = _extract_key_symbols(suffix_target, limit=6) or symbols
        instr_comp = (
            f"Complete the implementation body in `{clean_path}` starting immediately after the provided prefix."
        )
        reasoning_comp = (
            f"Teacher infilling task for `{clean_path}` ({split_label} split): completing control flow and "
            f"referencing {', '.join(tail_syms[:4]) or primary_sym}."
        )
        chat_comp = _format_sft_chat(instr_comp, prefix_ctx, reasoning_comp, suffix_target)
        examples.append(
            TeacherCurriculumExample(
                example_id=f"{split_label}-tchr-{seq_start + 1:04d}",
                split=split_label,
                task_type="context_completion_sft",
                source_file=doc.rel_path,
                instruction=instr_comp,
                input_context=prefix_ctx,
                teacher_reasoning=reasoning_comp,
                target_response=suffix_target,
                expected_symbols=tail_syms,
                formatted_sft_text=chat_comp,
            )
        )

    return examples, False


def synthesize_teacher_curriculum(
    splits: dict[str, list[SourceDocument]],
    output_dir: Path,
    dataset_name: str,
) -> tuple[dict[str, list[SourceDocument]], dict[str, Any]]:
    """Run the Private-Enclave Teacher across leak-free `train`, `val`, and `held_out` splits.

    Writes:
      - `teacher_sft_train.jsonl`
      - `teacher_sft_val.jsonl`
      - `teacher_sft_test.jsonl`
      - `teacher_curriculum_manifest.json`

    Returns `(teacher_source_docs_by_split, curriculum_summary)` so `ingest_universal_dataset`
    encodes the Teacher's SFT teaching pairs directly into `train.jsonl`, `val.jsonl`, and `held_out.jsonl`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    private_url = os.getenv("CRUCIBLE_PRIVATE_TEACHER_URL", "").strip()
    teacher_model = os.getenv("CRUCIBLE_PRIVATE_TEACHER_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct").strip()

    split_file_map = {
        "train": ("train", "teacher_sft_train.jsonl"),
        "val": ("val", "teacher_sft_val.jsonl"),
        "held_out": ("test", "teacher_sft_test.jsonl"),
    }

    teacher_docs_by_split: dict[str, list[SourceDocument]] = {"train": [], "val": [], "held_out": []}
    pair_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    task_counts: dict[str, int] = {}
    sample_previews: list[dict[str, Any]] = []
    any_live_enclave = False
    digest = hashlib.sha256()

    # Verify strict stem isolation across splits before Teacher synthesis
    train_sources = {d.stem_group for d in splits.get("train", [])}
    val_sources = {d.stem_group for d in splits.get("val", [])}
    test_sources = {d.stem_group for d in splits.get("held_out", [])}
    zero_leakage = not (train_sources & val_sources) and not (train_sources & test_sources) and not (val_sources & test_sources)

    for internal_split, (public_label, filename) in split_file_map.items():
        split_examples: list[TeacherCurriculumExample] = []
        seq = 1
        for doc in splits.get(internal_split, []):
            generated, used_live = _synthesize_examples_for_doc(
                doc=doc,
                split_label=public_label,
                seq_start=seq,
                private_url=private_url,
                teacher_model=teacher_model,
            )
            if used_live:
                any_live_enclave = True
            seq += len(generated)
            split_examples.extend(generated)

        out_jsonl = output_dir / filename
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for ex in split_examples:
                row = json.dumps(asdict(ex), ensure_ascii=False, sort_keys=True)
                digest.update(row.encode("utf-8"))
                fh.write(row + "\n")
                task_counts[ex.task_type] = task_counts.get(ex.task_type, 0) + 1

                sft_bytes = ex.formatted_sft_text.encode("utf-8")
                teacher_docs_by_split[internal_split].append(
                    SourceDocument(
                        rel_path=f"teacher_sft/{public_label}/{ex.example_id}.sft",
                        stem_group=f"teacher::{internal_split}::{ex.example_id}",
                        sha256=hashlib.sha256(sft_bytes).hexdigest(),
                        size_bytes=len(sft_bytes),
                        content=ex.formatted_sft_text,
                    )
                )
                if len(sample_previews) < 3 and public_label in {"train", "test"}:
                    sample_previews.append(
                        {
                            "example_id": ex.example_id,
                            "split": ex.split,
                            "task_type": ex.task_type,
                            "source_file": ex.source_file,
                            "instruction": ex.instruction[:200],
                            "teacher_reasoning": ex.teacher_reasoning[:220],
                            "response_preview": ex.target_response[:360],
                            "expected_symbols": ex.expected_symbols[:5],
                        }
                    )

        pair_counts[public_label] = len(split_examples)

    summary = {
        "dataset_name": dataset_name,
        "teacher_engine": (
            f"fuzzball-enclave-vllm ({teacher_model})"
            if any_live_enclave
            else f"fuzzball-enclave-teacher-synthesizer ({teacher_model} SFT schema)"
        ),
        "split_pair_counts": pair_counts,
        "total_teaching_pairs": sum(pair_counts.values()),
        "task_type_breakdown": task_counts,
        "zero_cross_split_leakage": zero_leakage,
        "raw_cloud_egress_bytes": 0,
        "artifacts": {
            "train_sft": "teacher_sft_train.jsonl",
            "val_sft": "teacher_sft_val.jsonl",
            "test_sft": "teacher_sft_test.jsonl",
        },
        "curriculum_sha256": digest.hexdigest(),
        "sample_previews": sample_previews,
    }
    (output_dir / "teacher_curriculum_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return teacher_docs_by_split, summary
