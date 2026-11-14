"""Repository -> tokenized training set with a leak-free held-out split.

Usage: python ingest.py --repo /data/raw/lammps --tokenizer /models/.../tokenizer.json --out /data/tok/lammps

Outputs (in --out):
  train.bin       uint32 tokens, files concatenated in path order (repository-aware), EOS between files
  eval.bin        uint32 tokens of held-out files; eval_index.json gives per-file [start, end) offsets
  eval_small.bin  first ~EVAL_SMALL tokens of held-out files (fast eval during search trials)
  meta.json       counts: files kept / dropped by reason, tokens per split, split rule
"""
import argparse, hashlib, json, os, re
import numpy as np
from tokenizers import Tokenizer

CODE_EXT = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu", ".cuh", ".f", ".f90", ".py", ".cmake"}
DOC_EXT = {".rst", ".md", ".txt"}
SKIP_DIRS = {".git", ".github", "lib", "potentials", "bench", "fortran", "msi2lmp"}  # vendored libs, data files
MAX_BYTES = 400_000          # skip huge generated / data-like files
EVAL_PCT = 5                 # % of file groups held out
EVAL_SMALL = 200_000         # tokens in the fast eval set
# Accelerator variants of one style (pair_lj_cut.cpp, OPENMP/pair_lj_cut_omp.cpp, KOKKOS/...kokkos.cpp)
# share a group so near-copies never straddle the train/eval split.
VARIANT = re.compile(r"_(omp|kokkos|kk|gpu|intel|opt|cuda|hip|dpd|tally)$")


def keep(rel, name, size):
    parts = rel.split(os.sep)
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return "skip_dir"
    if size > MAX_BYTES:
        return "too_big"
    ext = os.path.splitext(name)[1].lower()
    if ext in CODE_EXT or name == "CMakeLists.txt" or name.startswith("in."):
        return "code"
    if ext in DOC_EXT and (parts[0] == "doc" or len(parts) == 1):
        return "doc"
    return "other_ext"


def group_of(rel):
    stem = os.path.splitext(os.path.basename(rel))[0]
    return VARIANT.sub("", stem.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    os.makedirs(a.out, exist_ok=True)

    counts, seen, files = {}, set(), []
    for dp, dn, fn in os.walk(a.repo):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            p = os.path.join(dp, f); rel = os.path.relpath(p, a.repo)
            if os.path.islink(p) or not os.path.isfile(p):
                counts["symlink_or_special"] = counts.get("symlink_or_special", 0) + 1; continue
            kind = keep(rel, f, os.path.getsize(p))
            if kind not in ("code", "doc"):
                counts[kind] = counts.get(kind, 0) + 1; continue
            try:
                text = open(p, encoding="utf-8").read()
            except UnicodeDecodeError:
                counts["not_utf8"] = counts.get("not_utf8", 0) + 1; continue
            h = hashlib.sha1(text.strip().encode()).hexdigest()
            if h in seen or not text.strip():
                counts["duplicate_or_empty"] = counts.get("duplicate_or_empty", 0) + 1; continue
            seen.add(h); files.append((rel, kind, text))
            counts["kept_" + kind] = counts.get("kept_" + kind, 0) + 1

    def is_eval(rel):  # deterministic, group-level split
        return int(hashlib.md5(group_of(rel).encode()).hexdigest(), 16) % 100 < EVAL_PCT

    train, ev, index = [], [], []
    texts = [f"// File: {rel}\n{text}" for rel, _, text in files]
    encs = tok.encode_batch(texts)
    for (rel, kind, _), enc in zip(files, encs):
        ids = enc.ids + [eos]
        if is_eval(rel):
            index.append({"path": rel, "kind": kind, "start": len(ev), "end": len(ev) + len(ids)}); ev.extend(ids)
        else:
            train.extend(ids)

    np.asarray(train, dtype=np.uint32).tofile(os.path.join(a.out, "train.bin"))
    np.asarray(ev, dtype=np.uint32).tofile(os.path.join(a.out, "eval.bin"))
    small, n = [], 0
    for e in index:  # whole files only, in hash order, up to EVAL_SMALL tokens
        if n + (e["end"] - e["start"]) > EVAL_SMALL and small: break
        small.extend(ev[e["start"]:e["end"]]); n += e["end"] - e["start"]
    np.asarray(small, dtype=np.uint32).tofile(os.path.join(a.out, "eval_small.bin"))
    json.dump(index, open(os.path.join(a.out, "eval_index.json"), "w"))

    meta = {"repo": a.repo, "files": counts, "eos_id": eos,
            "tokens": {"train": len(train), "eval": len(ev), "eval_small": len(small)},
            "eval_files": len(index), "eval_groups": len({group_of(e["path"]) for e in index}),
            "split_rule": f"md5(group) % 100 < {EVAL_PCT}; group = basename without ext/accelerator suffix",
            "source": open(os.path.join(a.repo, "SOURCE_INFO")).read().strip()
                      if os.path.exists(os.path.join(a.repo, "SOURCE_INFO"))
                      else open(os.path.join(a.repo, "CRUCIBLE_SOURCE")).read().strip()}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("RESULT " + json.dumps({"kind": "ingest", **meta}), flush=True)


if __name__ == "__main__":
    main()
