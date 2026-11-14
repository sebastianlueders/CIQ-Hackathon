"""Build the results table from logged measurements only (results/). Missing => "not measured".

Usage: python src/report/make_report.py      -> writes results/summary.md, prints it
"""
import glob, json, math, os, statistics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(REPO, "results")
ACTIVE_PARAMS = 3.3e9   # Qwen3-Coder-30B-A3B active parameters per token (model card)


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def trial(name):
    return jl(os.path.join(R, "trials", f"{name}.jsonl"))


def tok_s(recs):
    t = [r["tok_s"] for r in recs if r.get("kind") == "train" and r.get("step", 0) > 1]  # skip warm-up step
    return statistics.median(t) if t else None


def bf16_peak():
    p = os.path.join(R, "env_probe_ngc.json")
    if not os.path.exists(p): return None
    return max(r["tflops"] for r in json.load(open(p))["records"] if r.get("kind") == "gemm" and r.get("dtype") == "bf16")


def fmt(x, nd=3):
    return "not measured" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    rows, notes = [], []
    # Quality: base = step-0 eval of any trial on eval_small; fine-tuned = recipe winner.
    base = next((r["eval_loss"] for f in sorted(glob.glob(os.path.join(R, "trials", "*.jsonl")))
                 for r in jl(f) if r.get("kind") == "eval" and r.get("tokens") == 0), None)
    recipe = json.load(open(os.path.join(R, "search", "recipe.json"))) if os.path.exists(os.path.join(R, "search", "recipe.json")) else {}
    ft = recipe.get("eval_loss")
    rows.append(("Held-out LAMMPS loss (perplexity), base -> fine-tuned",
                 f"{fmt(base)} ({fmt(base and math.exp(base), 2)}) -> {fmt(ft)} ({fmt(ft and math.exp(ft), 2)})"))
    dec = jl(os.path.join(R, "search", "recipe_decisions.jsonl"))
    rungs = [d for d in dec if d.get("event") == "rung"]
    if rungs:
        n0 = len(rungs[0]["ranking"]); stopped = sum(len(d["stopped"]) for d in rungs)
        rows.append(("Recipe search: configs tried / stopped early / winner",
                     f"{n0} / {stopped} / {recipe.get('config', 'running')}"))
    # Training throughput by experts implementation (phase B).
    perf = {impl: tok_s(trial(f"s4-perf-experts-{impl}")) for impl in ("eager", "grouped", "batched")}
    peak = bf16_peak()
    best_impl = max((k for k in perf if perf[k]), key=lambda k: perf[k], default=None)
    if perf.get("eager") and best_impl:
        rows.append(("Training throughput, eager experts -> best (" + best_impl + ")",
                     f"{perf['eager']:.0f} -> {perf[best_impl]:.0f} tokens/s ({perf[best_impl] / perf['eager']:.2f}x)"))
    else:
        rows.append(("Training throughput by experts implementation",
                     ", ".join(f"{k}: {fmt(v, 0)}" for k, v in perf.items())))
    if best_impl and peak:
        mfu = perf[best_impl] * 4 * ACTIVE_PARAMS / (peak * 1e12)
        rows.append(("Utilization vs measured BF16 peak (4 x active params FLOPs/token)", f"{100 * mfu:.1f}% of {peak} TFLOP/s"))
    # Serving (phase C).
    serve = {}
    for f in glob.glob(os.path.join(R, "serve", "*.json")):
        if f.endswith("best.json"): continue
        s = json.load(open(f)); serve[s["config"]] = s
    for name, s in sorted(serve.items()):
        if not s.get("ok"):
            rows.append((f"Serving {name}", "failed: " + s.get("error", "")[:80])); continue
        sp = {x["concurrency"]: x for x in s["speed"]}
        rows.append((f"Serving {name}: out tok/s at C=1 / C=16; held-out loss",
                     f"{sp.get(1, {}).get('out_tok_s')} / {sp.get(16, {}).get('out_tok_s')}; {s['eval_loss']}"))
    md = ["| Result | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    md += ["", f"Formula: utilization = tokens/s x 4 x {ACTIVE_PARAMS:.1e} FLOPs / measured BF16 GEMM peak "
           "(results/env_probe_ngc.json). Base and fine-tuned loss use the same held-out eval_small tokens."]
    open(os.path.join(R, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
