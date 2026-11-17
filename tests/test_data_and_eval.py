"""Tests for LAMMPS C++ ingestion, stem-grouped whole-file splits, and C++ completion floor."""

from __future__ import annotations

from pathlib import Path

from crucible.config import DatasetSplitConfig, repo_root
from crucible.data.completion_tasks import build_held_out_completion_suite, load_held_out_documents
from crucible.data.lammps_ingest import ingest_lammps_corpus
from crucible.eval.evaluator import (
    check_cpp_delimiter_balance,
    evaluate_quality_floor,
    score_completion_predictions,
)


def test_whole_file_stem_grouped_ingestion_no_leakage(tmp_path: Path) -> None:
    src = repo_root() / "fixtures" / "sample_lammps"
    cfg = DatasetSplitConfig(chunk_seq_len=128, chunk_overlap=16)

    manifest_1 = ingest_lammps_corpus(src, tmp_path / "run1", cfg)
    manifest_2 = ingest_lammps_corpus(src, tmp_path / "run2", cfg)

    assert manifest_1.manifest_sha256 == manifest_2.manifest_sha256
    assert manifest_1.raw_data_export_permitted is False
    # README.txt must be filtered out; 15 C++/header files discovered
    assert manifest_1.total_cpp_files == 15
    assert sum(manifest_1.split_file_counts.values()) == 15
    assert manifest_1.split_chunk_counts["train"] >= 1
    assert manifest_1.split_chunk_counts["val"] >= 1
    assert manifest_1.split_chunk_counts["held_out"] >= 1

    # Stem grouping: if foo.cpp is held out, foo.h must ALSO be held out (never in train)
    held_stems = {Path(p).with_suffix("").as_posix() for p in manifest_1.held_out_files}
    train_text = (tmp_path / "run1" / "train.jsonl").read_text(encoding="utf-8")
    for stem in held_stems:
        assert f"{stem}.h" not in train_text
        assert f"{stem}.cpp" not in train_text


def test_completion_suite_and_quality_floor_gate(tmp_path: Path) -> None:
    src = repo_root() / "fixtures" / "sample_lammps"
    cfg = DatasetSplitConfig(chunk_seq_len=128, chunk_overlap=16)
    ingest_lammps_corpus(src, tmp_path / "packed", cfg)

    held_docs = load_held_out_documents(tmp_path / "packed")
    tasks = build_held_out_completion_suite(held_docs)
    assert len(tasks) >= 1

    # Ground-truth reference completions should pass syntax balance & identifier recall
    exact_preds = {t.task_id: t.reference_completion for t in tasks}
    pass_rate, recall, _ = score_completion_predictions(tasks, exact_preds)
    assert pass_rate == 1.0
    assert recall == 1.0

    # Broken C++ syntax should fail delimiter balance
    assert not check_cpp_delimiter_balance("void foo() {\n", "  int x = (1 + 2;\n")

    ok_report = evaluate_quality_floor(
        held_out_loss=1.41,
        baseline_finetuned_loss=1.40,
        completion_pass_rate=0.90,
        mean_identifier_recall=0.88,
        max_perplexity_regression_pct=2.5,
        min_completion_pass_rate=0.84,
    )
    assert ok_report.passed_quality_floor

    bad_report = evaluate_quality_floor(
        held_out_loss=1.48,
        baseline_finetuned_loss=1.40,
        completion_pass_rate=0.75,
        mean_identifier_recall=0.70,
        max_perplexity_regression_pct=2.5,
        min_completion_pass_rate=0.84,
    )
    assert not bad_report.passed_quality_floor
    assert len(bad_report.rejection_reasons) == 2
