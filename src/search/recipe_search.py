"""Phase A: LoRA recipe search by successive halving, every trial a Fuzzball workflow.

Rung k trains each surviving config up to BUDGETS[k] total tokens (resuming from its
previous rung's adapter + optimizer state, constant LR after warmup), ranks by held-out
eval loss, and keeps the best 1/ETA. Every decision is appended to
results/search/recipe_decisions.jsonl; the winner goes to results/search/recipe.json.

Usage: python src/search/recipe_search.py [--n 8] [--budgets 500000,1500000,4000000] [--dry-run]
"""
import argparse, itertools, json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner"))
import runner  # noqa: E402

OUT = os.path.join(runner.REPO, "results", "search")
SPACE = {  # initial space; the LLM layer may narrow or extend it between rounds
    "lr": [1e-4, 2e-4, 5e-4, 1e-3],
    "lora_r": [8, 16, 64],
    "targets": [["q_proj", "k_proj", "v_proj", "o_proj"]],
}
FIXED = {"seq_len": 4096, "micro_batch": 1, "grad_accum": 8, "warmup_tokens": 65_536,
         "schedule": "constant", "eval_every_tokens": 250_000, "save_adapter": True, "resume": True}


def sample(n, seed=0):
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    random.Random(seed).shuffle(grid)
    cfgs = []
    for i, g in enumerate(grid[:n]):
        g = {**g, "lora_alpha": 2 * g["lora_r"]}
        name = f"c{i}-lr{g['lr']:.0e}-r{g['lora_r']}".replace("e-0", "e-")
        cfgs.append((name, g))
    return cfgs


def decide(fh, **rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fh.write(json.dumps(rec) + "\n"); fh.flush()
    print("[search] " + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--eta", type=int, default=2)
    ap.add_argument("--budgets", default="500000,1500000,4000000")
    ap.add_argument("--concurrency", type=int, default=2); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-from-rung", type=int, default=None,
                    help="resume: take the kept configs of this logged rung and run the remaining rungs")
    ap.add_argument("--keep", type=int, default=None, help="override how many configs survive a rung")
    a = ap.parse_args()
    budgets = [int(b) for b in a.budgets.split(",")]
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(OUT, "recipe_decisions.jsonl"), "a")
    alive, start, scores = sample(a.n), 0, {}
    if a.continue_from_rung is not None:
        rung = [d for d in map(json.loads, open(fh.name)) if d.get("event") == "rung" and d["rung"] == a.continue_from_rung][-1]
        keep = rung["kept"][:a.keep] if a.keep else rung["kept"]
        alive = [(n, g) for n, g in sample(12) if n in keep] or [(n, g) for n, g in sample(a.n) if n in keep]
        for r in rung["ranking"]: scores[r["config"]] = {"rung": a.continue_from_rung, "eval_loss": r["eval_loss"]}
        start = a.continue_from_rung + 1
        decide(fh, event="continue", from_rung=a.continue_from_rung, configs=[n for n, _ in alive])
    else:
        decide(fh, event="start", space=SPACE, fixed=FIXED, budgets=budgets, eta=a.eta, configs=[n for n, _ in alive])

    for k, budget in enumerate(budgets):
        if k < start: continue
        trials = [runner.Trial(f"s3-recipe-r{k}-{name}",
                               {**FIXED, **g, "train_tokens": budget, "out": f"/runs/trials/s3-recipe-{name}"})
                  for name, g in alive]
        if a.dry_run:
            print(json.dumps(runner.render(trials[0]))[:400]); return
        done = runner.run_all(trials, concurrency=a.concurrency)
        by_id = {t.id: t for t in done}
        ranked = sorted(((runner.final_eval(by_id[f"s3-recipe-r{k}-{n}"]), n, g) for n, g in alive), key=lambda x: x[0])
        for loss, n, g in ranked: scores[n] = {"rung": k, "tokens": budget, "eval_loss": loss, **g}
        ranked = [x for x in ranked if math.isfinite(x[0])] or ranked  # failed / cut trials never survive
        keep = (a.keep or max(1, len(ranked) // a.eta)) if k < len(budgets) - 1 else 1
        decide(fh, event="rung", rung=k, budget_tokens=budget,
               ranking=[{"config": n, "eval_loss": l} for l, n, _ in ranked],
               kept=[n for _, n, _ in ranked[:keep]], stopped=[n for _, n, _ in ranked[keep:]],
               reason=f"keep best 1/{a.eta} by held-out eval loss (failed trials score inf)")
        alive = [(n, g) for _, n, g in ranked[:keep]]

    best_name, best = alive[0]
    recipe = {"config": best_name, **FIXED, **best, "eval_loss": scores[best_name]["eval_loss"],
              "adapter": f"/runs/trials/s3-recipe-{best_name}/adapter", "all_scores": scores}
    json.dump(recipe, open(os.path.join(OUT, "recipe.json"), "w"), indent=1)
    decide(fh, event="done", winner=best_name, eval_loss=recipe["eval_loss"])


if __name__ == "__main__":
    main()
