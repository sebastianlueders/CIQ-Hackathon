"""Adopt already-launched trials of a rung: wait for them, rank by held-out loss, log the rung.

Used when the search driver is restarted while its trials keep running on the cluster.
Usage: python src/search/finish_rung.py --rung 0 --budget 250000 --keep 2 NAME=WORKFLOW_ID ...
Then:  python src/search/recipe_search.py --continue-from-rung 0 --keep 2 ...
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner"))
import runner  # noqa: E402
import recipe_search as rs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, required=True); ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--keep", type=int, default=2); ap.add_argument("trials", nargs="+")
    a = ap.parse_args()
    ts = []
    for spec in a.trials:
        name, wid = spec.split("=")
        t = runner.Trial(f"s3-recipe-r{a.rung}-{name}", {}); t.workflow_id = wid; ts.append((name, t))
    while True:
        for _, t in ts: t.status = runner.status(t.workflow_id)
        if all(t.status in runner.TERMINAL for _, t in ts): break
        time.sleep(20)
    for _, t in ts: runner.collect(t)
    ranked = sorted(((runner.final_eval(t), n) for n, t in ts), key=lambda x: x[0])
    finite = [x for x in ranked if math.isfinite(x[0])]
    kept = [n for _, n in finite[:a.keep]]
    fh = open(os.path.join(rs.OUT, "recipe_decisions.jsonl"), "a")
    rs.decide(fh, event="rung", rung=a.rung, budget_tokens=a.budget,
              ranking=[{"config": n, "eval_loss": l} for l, n in ranked],
              kept=kept, stopped=[n for _, n in ranked if n not in kept],
              reason=f"keep best {a.keep} by held-out eval loss (cluster schedules ~2 of our GPU jobs at once)")


if __name__ == "__main__":
    main()
