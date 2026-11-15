"""Phase B policy: which training optimizations become defaults, decided from measurements.

Order of candidates (the agent always tries them in this order):
  1. torch.compile (per decoder layer)   -> configs/perf-compile.json
  2. experts implementation               -> configs/perf-experts-*.json
  3. custom kernels (fused linear + CE)   -> configs/perf-fused-ce.json
A candidate is adopted only if its median tokens/s beats the baseline by >= MIN_SPEEDUP and its
training loss on the same steps/data stays within LOSS_TOL of the baseline. Adopted flags are
written to results/perf/defaults.json, which the recipe search and the pipeline emitter read.
"""
import json, os, statistics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T = os.path.join(REPO, "results", "trials")
MIN_SPEEDUP, LOSS_TOL = 1.05, 0.01
BASELINE = "s4-perf-experts-grouped"
STACKED_BASELINE = {"no_checkpointing": "s4-perf-compile", "fused_ce": "s4-perf-compile"}  # measured on top of compile
CANDIDATES = [("compile", "s4-perf-compile", {"compile": True}),
              ("no_checkpointing", "s4-perf-compile-nockpt", {"grad_ckpt": False}),
              ("fused_ce", "s4-perf-compile-fused-ce", {"fused_ce": True})]
MEM_SAVER_GIB = 5.0  # a candidate that is not slower and frees >= this much memory is adopted too


def recs(name):
    p = os.path.join(T, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def stats(name):
    tr = [r for r in recs(name) if r.get("kind") == "train"]
    if len(tr) < 3: return None
    steady = [r["tok_s"] for r in tr[2:]] or [r["tok_s"] for r in tr[1:]]  # skip warm-up / compile steps
    return {"tok_s": statistics.median(steady), "loss": {r["step"]: r["loss"] for r in tr},
            "peak": max(r["peak_mem_gib"] for r in tr)}


def decide():
    base0 = stats(BASELINE)
    out = {"baseline": BASELINE, "baseline_tok_s": base0 and base0["tok_s"], "adopted": {}, "decisions": []}
    for name, trial, flags in CANDIDATES:
        bname = STACKED_BASELINE.get(name, BASELINE); base = stats(bname)
        s = stats(trial)
        if not (base and s):
            out["decisions"].append({"candidate": name, "adopted": False, "reason": "not measured"}); continue
        common = sorted(set(base["loss"]) & set(s["loss"]))
        dl = max(abs(s["loss"][k] - base["loss"][k]) / base["loss"][k] for k in common) if common else None
        speed = s["tok_s"] / base["tok_s"]
        mem_saved = base["peak"] - s["peak"]
        same_loss = dl is not None and dl <= LOSS_TOL
        ok = same_loss and (speed >= MIN_SPEEDUP or (speed >= 1.0 and mem_saved >= MEM_SAVER_GIB))
        out["decisions"].append({"candidate": name, "baseline": bname, "adopted": ok, "speedup": round(speed, 3),
                                 "mem_saved_gib": round(mem_saved, 1),
                                 "tok_s": s["tok_s"], "max_rel_loss_diff": dl and round(dl, 4),
                                 "rule": f"loss within {LOSS_TOL:.0%} and (speedup >= {MIN_SPEEDUP} or not slower and frees >= {MEM_SAVER_GIB} GiB)"})
        if ok: out["adopted"].update(flags)
    os.makedirs(os.path.join(REPO, "results", "perf"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "results", "perf", "defaults.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(decide(), indent=1))
