"""Phase C: serving search. For each serving config, start vLLM, measure speed and quality, stop.

Usage (inside the vLLM image): python serve_bench.py --configs configs.json --out /runs/serve/<name>

configs.json: [{"name": "bf16", "args": []}, {"name": "fp8", "args": ["--quantization", "fp8"]}, ...]
Each config is measured on the same held-out LAMMPS prompts:
  speed:   output tok/s (aggregate) and per-request latency at each concurrency in --concurrency
  quality: mean next-token loss on held-out files from vLLM's prompt logprobs (same tokens for
           every config), so a faster config is only accepted if quality stays within a tolerance.
Every record prints as "RESULT {...}".
"""
import argparse, json, math, os, statistics, subprocess, threading, time, urllib.request
import numpy as np

URL = "http://127.0.0.1:8000"


def out(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wait_ready(proc, limit=1500):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None: return None
        try:
            urllib.request.urlopen(URL + "/health", timeout=5); return time.time() - t0
        except Exception:
            time.sleep(5)
    return None


def held_out(data, n, plen):
    """n prompts of plen tokens, each from a different held-out file (same for every config)."""
    ev = np.fromfile(os.path.join(data, "eval.bin"), dtype=np.uint32)
    idx = [e for e in json.load(open(os.path.join(data, "eval_index.json"))) if e["end"] - e["start"] >= plen + 1]
    return [ev[e["start"]:e["start"] + plen].tolist() for e in idx[:n]]


def quality(model, prompts):
    """Mean NLL of prompt tokens (excluding the first) from vLLM prompt logprobs."""
    nll, cnt = 0.0, 0
    for p in prompts:
        r = post("/v1/completions", {"model": model, "prompt": p, "max_tokens": 1, "temperature": 0,
                                     "echo": True, "logprobs": 0})
        lps = r["choices"][0]["logprobs"]["token_logprobs"][1:len(p)]
        lps = [x for x in lps if x is not None]
        nll -= sum(lps); cnt += len(lps)
    return nll / max(cnt, 1)


def speed(model, prompts, conc, gen):
    lat, toks, lock = [], [0], threading.Lock()
    work = list(prompts)
    def worker():
        while True:
            with lock:
                if not work: return
                p = work.pop()
            t = time.time()
            r = post("/v1/completions", {"model": model, "prompt": p, "max_tokens": gen, "temperature": 0,
                                         "ignore_eos": True})
            with lock:
                lat.append(time.time() - t); toks[0] += r["usage"]["completion_tokens"]
    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(conc)]
    [x.start() for x in th]; [x.join() for x in th]
    wall = time.time() - t0
    return {"concurrency": conc, "requests": len(prompts), "out_tok_s": round(toks[0] / wall, 1),
            "latency_p50_s": round(statistics.median(lat), 2),
            "latency_p95_s": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 2),
            "per_request_tok_s": round(gen / statistics.median(lat), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/models/Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--data", default="/data/tok/lammps")
    ap.add_argument("--concurrency", default="1,16"); ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--plen", type=int, default=1024); ap.add_argument("--n-quality", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    concs = [int(c) for c in a.concurrency.split(",")]
    qprompts = held_out(a.data, a.n_quality, a.plen)
    sprompts = held_out(a.data, max(concs) * 3, 512)
    for cfg in json.load(open(a.configs)):
        name, served = cfg["name"], cfg.get("served_model", "m")
        cmd = ["vllm", "serve", a.model, "--served-model-name", "m", "--host", "127.0.0.1", "--port", "8000",
               "--max-model-len", "4096", "--gpu-memory-utilization", str(cfg.get("gpu_util", 0.80)),
               *cfg.get("args", [])]
        log = open(os.path.join(a.out, f"{name}.server.log"), "w")
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        ready = wait_ready(proc)
        if ready is None:
            out(kind="serve", config=name, ok=False, error="server did not start",
                log_tail=open(log.name).read()[-1500:]); proc.kill(); continue
        rec = {"kind": "serve", "config": name, "ok": True, "args": cfg.get("args", []), "startup_s": round(ready, 1)}
        try:
            post("/v1/completions", {"model": served, "prompt": "int main() {", "max_tokens": 8})  # warm-up
            e = quality(served, qprompts)
            rec.update(eval_loss=round(e, 5), ppl=round(math.exp(e), 4), quality_tokens=len(qprompts) * (a.plen - 1))
            rec["speed"] = [speed(served, sprompts[:c * 3], c, a.gen) for c in concs]
        except Exception as ex:
            rec.update(ok=False, error=str(ex)[:500])
        out(**rec)
        json.dump(rec, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
        proc.terminate()
        try: proc.wait(60)
        except subprocess.TimeoutExpired: proc.kill()
        time.sleep(10)


if __name__ == "__main__":
    main()
