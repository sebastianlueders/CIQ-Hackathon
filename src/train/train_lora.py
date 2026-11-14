"""Single-node LoRA trainer: one search trial = one run of this script.

Usage: python train_lora.py --config trial.json

Config keys (defaults in DEFAULTS): model, data, out, seq_len, micro_batch, grad_accum,
lr, warmup_tokens, schedule (constant|cosine), train_tokens, lora_r, lora_alpha,
lora_dropout, targets, grad_ckpt, eval_every_tokens, seed, resume, save_adapter.

Writes to <out>/: metrics.jsonl (train + eval records), summary.json, and when
save_adapter or a later rung needs it, adapter/ + state.pt (resume point).
Every record line is also printed as "RESULT {...}" so it shows in the Fuzzball log.
"""
import argparse, json, math, os, time
import numpy as np
import torch

DEFAULTS = dict(
    model="/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-fused", data="/data/tok/lammps", out="/runs/trial",
    seq_len=4096, micro_batch=1, grad_accum=8, lr=2e-4, warmup_tokens=1_000_000, schedule="constant",
    train_tokens=4_000_000, lora_r=16, lora_alpha=32, lora_dropout=0.0,
    targets=["q_proj", "k_proj", "v_proj", "o_proj"], grad_ckpt=True,
    eval_every_tokens=1_000_000, eval_set="eval_small.bin", seed=0, resume=False, save_adapter=True,
)


def log(fh, **rec):
    rec["t"] = time.time()
    line = json.dumps(rec)
    fh.write(line + "\n"); fh.flush(); print("RESULT " + line, flush=True)


def batches(tokens, seq_len, micro_batch, rng):
    """Random contiguous windows from the packed, repository-ordered token stream."""
    n = len(tokens) - seq_len - 1
    while True:
        idx = rng.integers(0, n, size=micro_batch)
        x = np.stack([tokens[i:i + seq_len + 1] for i in idx]).astype(np.int64)
        yield torch.from_numpy(x).cuda(non_blocking=True)


@torch.no_grad()
def evaluate(model, tokens, seq_len):
    """Mean next-token loss over the held-out set, in non-overlapping windows."""
    model.eval()
    tot, cnt = 0.0, 0
    for s in range(0, len(tokens) - 1, seq_len):
        x = torch.from_numpy(tokens[s:s + seq_len + 1].astype(np.int64)).cuda()[None]
        if x.shape[1] < 2: break
        logits = model(input_ids=x[:, :-1], use_cache=False).logits.float()  # loss computed here, not by HF
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        tot += loss.item(); cnt += x.shape[1] - 1
    model.train()
    return tot / max(cnt, 1)


def lr_at(cfg, tokens_seen):
    w = cfg["warmup_tokens"]
    if tokens_seen < w: return cfg["lr"] * tokens_seen / w
    if cfg["schedule"] == "cosine":
        p = min(1.0, (tokens_seen - w) / max(1, cfg["train_tokens"] - w))
        return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    return cfg["lr"]  # constant after warmup: lets successive-halving rungs resume cleanly


def start_mem_monitor(fh, every=20):
    """Background thread: log process RSS, cgroup usage and CUDA memory (load-time OOM diagnosis)."""
    import threading
    def cg():
        for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            try: return round(int(open(f).read()) / 2**30, 1)
            except Exception: pass
    def rss():
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"): return round(int(l.split()[1]) / 2**20, 1)
    def loop():
        while True:
            free, total = torch.cuda.mem_get_info()
            log(fh, kind="mem", rss_gib=rss(), cgroup_gib=cg(),
                cuda_alloc_gib=round(torch.cuda.memory_allocated() / 2**30, 1),
                device_used_gib=round((total - free) / 2**30, 1))
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()


def drop_cache(path):
    """Release page cache for a file (on GB10 page cache and GPU memory share one pool)."""
    try:
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass


def load_direct(path):
    """Load a checkpoint already in transformers' in-memory layout (see src/prep/fuse_experts.py):
    build on meta, read each shard straight to GPU, then drop that shard's page cache."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    shards = sorted(set(json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"].values()))
    for f in shards: drop_cache(os.path.join(path, f))  # stale cache from earlier jobs on this node
    loaded = set()
    for f in shards:
        with safe_open(os.path.join(path, f), framework="pt", device="cuda:0") as h:
            sd = {k: h.get_tensor(k) for k in h.keys()}
        model.load_state_dict(sd, strict=False, assign=True); loaded |= sd.keys(); del sd
        drop_cache(os.path.join(path, f))
    missing = [k for k in model.state_dict() if k not in loaded]
    if missing: raise RuntimeError(f"direct load missing {len(missing)} tensors, e.g. {missing[:3]}")
    # Non-persistent buffers (rotary inv_freq) were created on meta: rebuild them on GPU.
    rot = model.model.rotary_emb
    with torch.device("cuda:0"):
        model.model.rotary_emb = type(rot)(config=cfg)
    for n, b in model.named_buffers():
        if b.is_meta: raise RuntimeError(f"buffer still on meta: {n}")
    return model


class RouterStats:
    """Counts top-k expert selections per MoE layer (router-collapse monitor)."""
    def __init__(self, model):
        self.counts, self.k = {}, getattr(model.config, "num_experts_per_tok", 8)
        for name, m in model.named_modules():
            if name.endswith("mlp.gate") and isinstance(m, torch.nn.Linear):
                m.register_forward_hook(self._hook(name))
    def _hook(self, name):
        def f(mod, inp, out):
            idx = out.topk(self.k, dim=-1).indices.flatten()
            c = torch.bincount(idx, minlength=out.shape[-1])
            self.counts[name] = self.counts.get(name, 0) + c
        return f
    def summary(self):
        if not self.counts: return None
        ent = []
        for c in self.counts.values():
            p = c.float() / c.sum(); ent.append(float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(len(p))))
        self.counts = {}
        return {"router_entropy_min": round(min(ent), 4), "router_entropy_mean": round(sum(ent) / len(ent), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = {**DEFAULTS, **json.load(open(ap.parse_args().config))}
    # Data parallel across nodes (one GB10 GPU each): every rank trains on its own data stream
    # and LoRA gradients are averaged once per step. Rank 0 evaluates, logs, and saves.
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    dist = None
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl"); torch.cuda.set_device(0)
    os.makedirs(cfg["out"], exist_ok=True)
    fh = open(os.path.join(cfg["out"], "metrics.jsonl") if rank == 0 else os.devnull, "a")
    if rank == 0: log(fh, kind="config", world_size=world, **cfg)
    torch.manual_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"] + 1000 * rank)

    if rank == 0: start_mem_monitor(fh)
    if cfg.get("fused_ce"):
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        global fused_linear_ce
        from fused_ce import fused_linear_ce
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    t0 = time.time()
    if cfg.get("loader", "direct") == "direct" and cfg["model"].endswith("-fused"):
        model = load_direct(cfg["model"])
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", device_map={"": 0})
    if cfg.get("experts_impl"):  # "eager" (per-expert loop) | "grouped_mm" | "batched_mm" | registered custom kernel
        model.config._experts_implementation = cfg["experts_impl"]
    log(fh, kind="loaded", seconds=round(time.time() - t0, 1), loader=cfg.get("loader", "direct"),
        experts_impl=model.config._experts_implementation)
    load_s = time.time() - t0
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    adapter_dir = os.path.join(cfg["out"], "adapter"); state_path = os.path.join(cfg["out"], "state.pt")
    resumed = cfg["resume"] and os.path.exists(state_path)
    if resumed:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                                 lora_dropout=cfg["lora_dropout"],
                                                 target_modules=cfg["targets"], task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)
    tokens_seen, step = 0, 0
    if resumed:
        st = torch.load(state_path, weights_only=False)
        opt.load_state_dict(st["opt"]); tokens_seen, step = st["tokens_seen"], st["step"]
        rng = np.random.default_rng(st["rng_seed"])
    router = RouterStats(model)

    train = np.memmap(os.path.join(cfg["data"], "train.bin"), dtype=np.uint32, mode="r")
    ev = np.fromfile(os.path.join(cfg["data"], cfg["eval_set"]), dtype=np.uint32)
    n_trainable = sum(p.numel() for p in params)
    log(fh, kind="setup", load_s=round(load_s, 1), trainable_params=n_trainable, resumed=resumed,
        train_tokens_available=int(len(train)), eval_tokens=int(len(ev)),
        mem_gib=round(torch.cuda.memory_allocated() / 2**30, 1))

    if not resumed and not cfg.get("skip_eval") and rank == 0:
        e = evaluate(model, ev, cfg["seq_len"])
        log(fh, kind="eval", tokens=0, eval_loss=round(e, 5), ppl=round(math.exp(e), 4))
    if dist: dist.barrier()

    per_step = cfg["seq_len"] * cfg["micro_batch"] * cfg["grad_accum"] * world  # cluster-wide tokens/step
    gen = batches(train, cfg["seq_len"], cfg["micro_batch"], rng)
    next_eval = (tokens_seen // cfg["eval_every_tokens"] + 1) * cfg["eval_every_tokens"]
    model.train(); torch.cuda.reset_peak_memory_stats()
    while tokens_seen < cfg["train_tokens"]:
        t = time.time(); tot = 0.0
        for g in opt.param_groups: g["lr"] = lr_at(cfg, tokens_seen)
        for _ in range(cfg["grad_accum"]):
            x = next(gen)
            if cfg.get("fused_ce"):  # src/kernels/fused_ce.py: chunked lm_head + Triton CE, no full FP32 logits
                base = model.get_base_model()
                hid = base.model(input_ids=x[:, :-1], use_cache=False).last_hidden_state
                loss = fused_linear_ce(hid.reshape(-1, hid.shape[-1]), base.lm_head.weight, x[:, 1:].reshape(-1))
            else:
                loss = model(input_ids=x[:, :-1], labels=x[:, :-1], use_cache=False).loss  # HF shifts labels
            (loss / cfg["grad_accum"]).backward(); tot += loss.item() / cfg["grad_accum"]
        comm_s = 0.0
        if dist:  # one flattened all-reduce of the LoRA gradients (~13M params)
            tc = time.time()
            grads = [p.grad for p in params]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat); flat /= world
            for g, f in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)): g.copy_(f)
            torch.cuda.synchronize(); comm_s = time.time() - tc
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - t
        tokens_seen += per_step; step += 1
        rec = dict(kind="train", step=step, tokens=tokens_seen, loss=round(tot, 5), lr=opt.param_groups[0]["lr"],
                   grad_norm=round(float(gn), 4), step_s=round(dt, 3), tok_s=round(per_step / dt, 1),
                   peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                   world_size=world, allreduce_s=round(comm_s, 4))
        if step % 10 == 0 or step == 1:
            rs = router.summary()
            if rs: rec.update(rs)
        log(fh, **rec)
        if not math.isfinite(tot):
            log(fh, kind="abort", reason="non-finite loss"); break
        if not cfg.get("skip_eval") and (tokens_seen >= next_eval or tokens_seen >= cfg["train_tokens"]):
            if rank == 0:
                e = evaluate(model, ev, cfg["seq_len"])
                log(fh, kind="eval", tokens=tokens_seen, eval_loss=round(e, 5), ppl=round(math.exp(e), 4))
            if dist: dist.barrier()
            next_eval += cfg["eval_every_tokens"]

    if rank != 0:
        dist.barrier(); dist.destroy_process_group(); return
    if cfg["save_adapter"]:
        model.save_pretrained(adapter_dir)
        torch.save({"opt": opt.state_dict(), "tokens_seen": tokens_seen, "step": step,
                    "rng_seed": int(rng.integers(1 << 31))}, state_path)
    summary = dict(kind="summary", tokens=tokens_seen, steps=step,
                   final_eval_loss=([json.loads(l)["eval_loss"] for l in open(fh.name) if '"kind": "eval"' in l] or [None])[-1])
    json.dump(summary, open(os.path.join(cfg["out"], "summary.json"), "w"), indent=1)
    log(fh, **summary)
    if dist: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
