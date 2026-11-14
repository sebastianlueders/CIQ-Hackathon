"""LAMMPS C++ repository ingestion, stem-grouped deterministic splits, and packed chunking.

Whole-file stem grouping (`pair_lj_cut.h` + `pair_lj_cut.cpp`) ensures header declarations never
leak into training when an implementation file is assigned to the sealed held-out split.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from crucible.config import DatasetSplitConfig

CPP_EXTENSIONS = frozenset({".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu", ".cuh"})
SKIP_DIRS = frozenset({".git", "build", "cmake-build", "__pycache__", "node_modules", "bench_bin"})
LEXEME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|\S|\n")


@dataclass(frozen=True)
class SourceDocument:
    rel_path: str
    stem_group: str
    sha256: str
    size_bytes: int
    content: str


@dataclass(frozen=True)
class PackedChunk:
    chunk_id: str
    split: str
    source_files: list[str]
    input_ids: list[int]
    length: int


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    source_root: str
    output_dir: str
    split_salt: str
    vocab_size: int
    chunk_seq_len: int
    total_cpp_files: int
    split_file_counts: dict[str, int]
    split_chunk_counts: dict[str, int]
    split_lexeme_counts: dict[str, int]
    held_out_files: list[str]
    manifest_sha256: str
    raw_data_export_permitted: bool = False


class DeterministicCodeEncoder:
    """Fast C++-aware subword/lexeme encoder compatible with Qwen3-Coder 151,936 vocab IDs.

    When a local HuggingFace model directory with `tokenizer.json` is provided and `transformers`
    is installed, `build_encoder` uses it directly; otherwise this deterministic C++ lexeme
    encoder produces stable IDs in `[4, vocab_size)` with reserved special IDs `0..3`.
    """

    BOS_ID = 1
    EOS_ID = 2
    SEP_ID = 3

    def __init__(self, vocab_size: int = 151936) -> None:
        self.vocab_size = vocab_size
        self._usable = vocab_size - 4

    def encode(self, text: str) -> list[int]:
        ids = [self.BOS_ID]
        for match in LEXEME_RE.finditer(text):
            piece = match.group(0)
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()
            idx = 4 + (int.from_bytes(digest, "little") % self._usable)
            ids.append(idx)
        ids.append(self.EOS_ID)
        return ids


def build_encoder(vocab_size: int = 151936, hf_model_dir: Path | None = None) -> Callable[[str], list[int]]:
    if hf_model_dir is not None and (hf_model_dir / "tokenizer.json").exists():
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(str(hf_model_dir), local_files_only=True)
            return lambda text: list(tok.encode(text, add_special_tokens=True))
        except Exception:
            pass
    fallback = DeterministicCodeEncoder(vocab_size=vocab_size)
    return fallback.encode


def discover_cpp_documents(source_root: Path) -> list[SourceDocument]:
    if not source_root.exists():
        raise FileNotFoundError(f"LAMMPS source directory does not exist: {source_root}")

    docs: list[SourceDocument] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in CPP_EXTENSIONS:
            continue
        raw_bytes = path.read_bytes()
        if not raw_bytes or b"\x00" in raw_bytes[:1024]:
            continue
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        if len(text) < 32:
            continue
        rel = path.relative_to(source_root).as_posix()
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
    if not docs:
        raise ValueError(f"No valid C++/CUDA files discovered under {source_root}")
    return docs


def assign_whole_file_splits(
    docs: list[SourceDocument],
    cfg: DatasetSplitConfig,
) -> dict[str, list[SourceDocument]]:
    groups: dict[str, list[SourceDocument]] = {}
    for doc in docs:
        groups.setdefault(doc.stem_group, []).append(doc)

    scored_groups: list[tuple[float, str]] = []
    for group_key in sorted(groups):
        digest = hashlib.sha256(f"{cfg.split_salt}:{group_key}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        scored_groups.append((unit, group_key))

    scored_groups.sort(key=lambda item: (item[0], item[1]))
    n_groups = len(scored_groups)

    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct file stem groups for train/val/held_out split, found {n_groups}"
        )

    n_held_out = max(1, round(n_groups * cfg.held_out_ratio))
    n_val = max(1, round(n_groups * cfg.val_ratio))
    n_train = n_groups - n_val - n_held_out
    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_held_out -= 1

    splits: dict[str, list[SourceDocument]] = {"train": [], "val": [], "held_out": []}
    for idx, (_, group_key) in enumerate(scored_groups):
        if idx < n_train:
            target = "train"
        elif idx < n_train + n_val:
            target = "val"
        else:
            target = "held_out"
        splits[target].extend(groups[group_key])

    for split_name in splits:
        splits[split_name].sort(key=lambda d: d.rel_path)
    return splits


def pack_split_chunks(
    split_name: str,
    docs: list[SourceDocument],
    encode_fn: Callable[[str], list[int]],
    seq_len: int,
    overlap: int,
) -> tuple[list[PackedChunk], int]:
    flat_ids: list[int] = []
    file_spans: list[tuple[int, int, str]] = []
    for doc in docs:
        header = f"// FILE: {doc.rel_path}\n"
        ids = encode_fn(header + doc.content)
        start = len(flat_ids)
        flat_ids.extend(ids)
        file_spans.append((start, len(flat_ids), doc.rel_path))

    total_lexemes = len(flat_ids)
    if total_lexemes == 0:
        return [], 0

    stride = max(1, seq_len - overlap)
    chunks: list[PackedChunk] = []
    pos = 0
    chunk_idx = 0
    while pos < total_lexemes:
        window = flat_ids[pos : pos + seq_len]
        if len(window) < min(32, seq_len) and chunks:
            break
        end_pos = pos + len(window)
        contributing = [
            rel for (s_start, s_end, rel) in file_spans if s_start < end_pos and s_end > pos
        ]
        chunks.append(
            PackedChunk(
                chunk_id=f"{split_name}-{chunk_idx:05d}",
                split=split_name,
                source_files=contributing,
                input_ids=window,
                length=len(window),
            )
        )
        chunk_idx += 1
        if end_pos >= total_lexemes:
            break
        pos += stride
    return chunks, total_lexemes


def ingest_lammps_corpus(
    source_root: Path,
    output_dir: Path,
    split_cfg: DatasetSplitConfig,
    vocab_size: int = 151936,
    hf_model_dir: Path | None = None,
) -> DatasetManifest:
    docs = discover_cpp_documents(source_root)
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
    return manifest
