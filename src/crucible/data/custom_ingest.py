"""Universal local dataset ingestion for dropped codebases, JSONL/JSON, CSV, text/docs, and ZIPs.

Guarantees:
- 100% local execution (never transmits raw examples across any network or LLM boundary).
- Zero `python-multipart` requirement: accepts Base64-encoded browser file drops, `.zip` archives,
  or local folder paths.
- Deterministic leak-free `train` / `val` / sealed `held_out` splits:
  - Source/doc files are stem-grouped (`pair_lj_cut.h` + `pair_lj_cut.cpp` stay together).
  - Structured `.jsonl` / `.json` / `.csv` files split at the individual record level.
  - Single-file text/code drops automatically sub-segment into deterministic blocks when fewer
    than 3 files are dropped so every drop works on the first try.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from crucible.config import DatasetSplitConfig
from crucible.data.lammps_ingest import (
    SKIP_DIRS,
    DatasetManifest,
    SourceDocument,
    assign_whole_file_splits,
    build_encoder,
    pack_split_chunks,
)

TEXT_AND_CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".cu",
        ".cuh",
        ".rs",
        ".go",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".kt",
        ".scala",
        ".swift",
        ".cs",
        ".rb",
        ".php",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".markdown",
        ".rst",
        ".txt",
        ".cmake",
        ".f90",
        ".f",
        ".lua",
        ".r",
        ".jl",
    }
)
STRUCTURED_EXTENSIONS = frozenset({".jsonl", ".json", ".csv", ".tsv"})


def _safe_rel_path(raw_name: str) -> Path | None:
    cleaned = raw_name.replace("\\", "/").lstrip("/")
    parts = [p for p in Path(cleaned).parts if p not in ("", ".", "..")]
    if not parts or any(p in SKIP_DIRS or p.startswith("__MACOSX") for p in parts):
        return None
    if parts[-1].startswith("."):
        return None
    return Path(*parts)


def stage_uploaded_files(
    files_payload: list[dict[str, Any]],
    raw_dest_dir: Path,
) -> list[Path]:
    """Decode browser-dropped files (including automatic `.zip` extraction) into `raw_dest_dir`."""
    raw_dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for entry in files_payload:
        name = str(entry.get("name") or entry.get("path") or "").strip()
        if not name:
            continue
        if "content_b64" in entry and entry["content_b64"] is not None:
            b64_str = str(entry["content_b64"])
            if "," in b64_str and b64_str.lstrip().startswith("data:"):
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)
        elif "content_text" in entry and entry["content_text"] is not None:
            raw_bytes = str(entry["content_text"]).encode("utf-8")
        elif "content" in entry and entry["content"] is not None:
            raw_bytes = str(entry["content"]).encode("utf-8")
        else:
            continue

        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    rel = _safe_rel_path(member.filename)
                    if rel is None:
                        continue
                    target = (raw_dest_dir / rel).resolve()
                    if not str(target).startswith(str(raw_dest_dir.resolve())):
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = zf.read(member)
                    target.write_bytes(data)
                    written.append(target)
        else:
            rel = _safe_rel_path(name)
            if rel is None:
                continue
            target = (raw_dest_dir / rel).resolve()
            if not str(target).startswith(str(raw_dest_dir.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw_bytes)
            written.append(target)

    if not written:
        raise ValueError("No valid code, JSONL/JSON, CSV, or text files found in upload.")
    return written


def _format_record_obj(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)

    if isinstance(obj.get("messages"), list):
        turns: list[str] = []
        for m in obj["messages"]:
            if isinstance(m, dict):
                role = str(m.get("role", "user")).upper()
                content = str(m.get("content", "")).strip()
                if content:
                    turns.append(f"<|{role}|>\n{content}")
        if turns:
            return "\n".join(turns)

    if "instruction" in obj or "output" in obj or "response" in obj:
        instr = str(obj.get("instruction") or obj.get("prompt") or "").strip()
        inp = str(obj.get("input") or obj.get("context") or "").strip()
        out = str(obj.get("output") or obj.get("response") or obj.get("completion") or "").strip()
        blocks = []
        if instr:
            blocks.append(f"### Instruction:\n{instr}")
        if inp:
            blocks.append(f"### Input:\n{inp}")
        if out:
            blocks.append(f"### Response:\n{out}")
        if blocks:
            return "\n\n".join(blocks)

    if "prompt" in obj and "completion" in obj:
        return f"{str(obj['prompt']).strip()}\n{str(obj['completion']).strip()}"

    for key in ("text", "code", "content", "body"):
        if isinstance(obj.get(key), str) and obj[key].strip():
            return obj[key].strip()

    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _extract_structured_documents(rel_path: str, raw_bytes: bytes, suffix: str) -> list[SourceDocument]:
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    records: list[str] = []
    if suffix == ".jsonl":
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(_format_record_obj(json.loads(ln)))
            except json.JSONDecodeError:
                records.append(ln)
    elif suffix == ".json":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    records.append(_format_record_obj(item))
            elif isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                for item in parsed["data"]:
                    records.append(_format_record_obj(item))
            else:
                records.append(_format_record_obj(parsed))
        except json.JSONDecodeError:
            records.append(text)
    elif suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        for row in reader:
            formatted = _format_record_obj(row)
            if formatted:
                records.append(formatted)

    docs: list[SourceDocument] = []
    for idx, rec in enumerate(records):
        clean = rec.strip()
        if len(clean) < 8:
            continue
        rec_bytes = clean.encode("utf-8")
        docs.append(
            SourceDocument(
                rel_path=f"{rel_path}#record-{idx:05d}",
                stem_group=f"{rel_path.lower()}::rec-{idx:05d}",
                sha256=hashlib.sha256(rec_bytes).hexdigest(),
                size_bytes=len(rec_bytes),
                content=clean,
            )
        )
    return docs


def discover_universal_documents(source_root: Path) -> tuple[list[SourceDocument], str]:
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_root}")

    docs: list[SourceDocument] = []
    ext_counts: dict[str, int] = {}

    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts) or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        raw_bytes = path.read_bytes()
        if not raw_bytes or b"\x00" in raw_bytes[:1024]:
            continue
        rel = path.relative_to(source_root).as_posix()

        if suffix in STRUCTURED_EXTENSIONS:
            extracted = _extract_structured_documents(rel, raw_bytes, suffix)
            if extracted:
                docs.extend(extracted)
                ext_counts[suffix] = ext_counts.get(suffix, 0) + len(extracted)
            continue

        if suffix in TEXT_AND_CODE_EXTENSIONS or suffix == "":
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            if len(text) < 16:
                continue
            stem_group = str(Path(rel).with_suffix("")).lower()
            docs.append(
                SourceDocument(
                    rel_path=rel,
                    stem_group=stem_group,
                    sha256=hashlib.sha256(raw_bytes).hexdigest(),
                    size_bytes=len(raw_bytes),
                    content=text,
                )
            )
            ext_counts[suffix or ".txt"] = ext_counts.get(suffix or ".txt", 0) + 1

    if not docs:
        raise ValueError(f"No readable source code, JSONL/JSON, CSV, or text records found in {source_root}")

    # If user dropped only 1-2 files with < 3 stem groups, split into deterministic blocks so
    # train / val / sealed held_out all receive non-overlapping content without erroring.
    distinct_stems = {d.stem_group for d in docs}
    if len(distinct_stems) < 3:
        expanded: list[SourceDocument] = []
        for doc in docs:
            lines = [ln for ln in doc.content.splitlines() if ln.strip()]
            if len(lines) >= 6:
                n_parts = max(5, min(10, len(lines) // 2))
                step = max(1, len(lines) // n_parts)
                for b_idx, start in enumerate(range(0, len(lines), step)):
                    seg = "\n".join(lines[start : start + step]).strip()
                    if not seg:
                        continue
                    b_bytes = seg.encode("utf-8")
                    expanded.append(
                        SourceDocument(
                            rel_path=f"{doc.rel_path}#seg-{b_idx:03d}",
                            stem_group=f"{doc.stem_group}::seg-{b_idx:03d}",
                            sha256=hashlib.sha256(b_bytes).hexdigest(),
                            size_bytes=len(b_bytes),
                            content=seg,
                        )
                    )
            else:
                # Minimal fallback segments from character thirds
                txt = doc.content
                third = max(1, len(txt) // 3)
                for b_idx, seg in enumerate([txt[:third], txt[third : 2 * third], txt[2 * third :]]):
                    if seg.strip():
                        b_bytes = seg.encode("utf-8")
                        expanded.append(
                            SourceDocument(
                                rel_path=f"{doc.rel_path}#seg-{b_idx:03d}",
                                stem_group=f"{doc.stem_group}::seg-{b_idx:03d}",
                                sha256=hashlib.sha256(b_bytes).hexdigest(),
                                size_bytes=len(b_bytes),
                                content=seg.strip(),
                            )
                        )
        if len({d.stem_group for d in expanded}) >= 3:
            docs = expanded

    top_exts = ", ".join(f"{ext} ({cnt})" for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1])[:4])
    return docs, top_exts or "text/code"


def ingest_universal_dataset(
    source_root: Path,
    output_dir: Path,
    split_cfg: DatasetSplitConfig,
    vocab_size: int = 151936,
    hf_model_dir: Path | None = None,
) -> tuple[DatasetManifest, str]:
    """Ingest any code/JSONL/CSV/text corpus into `train.jsonl`, `val.jsonl`, and sealed `held_out.jsonl`."""
    docs, format_summary = discover_universal_documents(source_root)
    splits = assign_whole_file_splits(docs, split_cfg)
    encode_fn = build_encoder(vocab_size=vocab_size, hf_model_dir=hf_model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_chunk_counts: dict[str, int] = {}
    split_lexeme_counts: dict[str, int] = {}

    hasher = hashlib.sha256()
    for split_name in ("train", "val", "held_out"):
        chunks, lex_count = pack_split_chunks(
            split_name=split_name,
            docs=splits[split_name],
            encode_fn=encode_fn,
            seq_len=split_cfg.chunk_seq_len,
            overlap=split_cfg.chunk_overlap,
        )
        split_chunk_counts[split_name] = len(chunks)
        split_lexeme_counts[split_name] = lex_count
        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for ch in chunks:
                line = json.dumps(asdict(ch), sort_keys=True)
                hasher.update(line.encode("utf-8"))
                fh.write(line + "\n")

    held_out_sources_path = output_dir / "held_out_documents.jsonl"
    with held_out_sources_path.open("w", encoding="utf-8") as fh:
        for doc in splits["held_out"]:
            fh.write(json.dumps(asdict(doc), sort_keys=True) + "\n")

    manifest = DatasetManifest(
        dataset_name=split_cfg.name,
        source_root=str(source_root.resolve()),
        output_dir=str(output_dir.resolve()),
        split_salt=split_cfg.split_salt,
        vocab_size=vocab_size,
        chunk_seq_len=split_cfg.chunk_seq_len,
        total_cpp_files=len(docs),
        split_file_counts={k: len(v) for k, v in splits.items()},
        split_chunk_counts=split_chunk_counts,
        split_lexeme_counts=split_lexeme_counts,
        held_out_files=[d.rel_path for d in splits["held_out"]],
        manifest_sha256=hasher.hexdigest(),
        raw_data_export_permitted=False,
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest, format_summary
