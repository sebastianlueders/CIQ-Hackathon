"""One-time checkpoint conversion: per-expert MoE weights -> transformers' fused 3D layout.

The released Qwen3-MoE checkpoint stores 48 layers x 128 experts x {gate,up,down}_proj as
separate tensors. transformers 5 keeps them fused per layer:
    mlp.experts.gate_up_proj  [E, 2*I, H]   = cat(gate, up) along dim 1 (forward chunks gate first)
    mlp.experts.down_proj     [E, H, I]
and converts on every load (~10 min for this model on GB10). Writing the fused layout once
turns each trial's load into a straight read from VAST.

Usage: python fuse_experts.py --src /models/Qwen/Qwen3-Coder-30B-A3B-Instruct \
                              --dst /models/Qwen/Qwen3-Coder-30B-A3B-Instruct-fused
"""
import argparse, json, os, re, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


def drop_cache(path):
    """Release page cache for a file we are done with (cache and GPU share memory on GB10)."""
    try:
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--layers-per-shard", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.dst, exist_ok=True)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    E, L = cfg["num_experts"], cfg["num_hidden_layers"]
    wmap = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]

    by_layer, other = {}, []
    for k in wmap:
        m = EXPERT.match(k)
        if m: by_layer.setdefault(int(m.group(2)), []).append(k)
        else: other.append(k)

    handles = {}
    def get(k):
        f = wmap[k]
        if f not in handles: handles[f] = safe_open(os.path.join(a.src, f), framework="pt", device="cpu")
        return handles[f].get_tensor(k)

    new_map, shard_id, n_shards = {}, 0, (L + a.layers_per_shard - 1) // a.layers_per_shard + 1
    for start in range(0, L, a.layers_per_shard):
        out = {}
        for li in range(start, min(L, start + a.layers_per_shard)):
            pre = f"model.layers.{li}.mlp.experts"
            gate = [get(f"{pre}.{e}.gate_proj.weight") for e in range(E)]
            up = [get(f"{pre}.{e}.up_proj.weight") for e in range(E)]
            down = [get(f"{pre}.{e}.down_proj.weight") for e in range(E)]
            out[f"{pre}.gate_up_proj"] = torch.stack([torch.cat([g, u], 0) for g, u in zip(gate, up)]).contiguous()
            out[f"{pre}.down_proj"] = torch.stack(down).contiguous()
            for k in other:  # attention, norms, router for this layer
                if k.startswith(f"model.layers.{li}."): out[k] = get(k)
        name = f"model-{shard_id + 1:05d}-of-{n_shards:05d}.safetensors"
        save_file(out, os.path.join(a.dst, name), metadata={"format": "pt"})
        for k in out: new_map[k] = name
        shard_id += 1
        print(f"shard {name}: layers {start}..{min(L, start + a.layers_per_shard) - 1} "
              f"({sum(t.numel() * t.element_size() for t in out.values()) / 2**30:.1f} GiB) t={time.time() - t0:.0f}s", flush=True)
        del out
    rest = {k: get(k) for k in other if not k.startswith("model.layers.")}  # embeddings, final norm, lm_head
    name = f"model-{shard_id + 1:05d}-of-{n_shards:05d}.safetensors"
    save_file(rest, os.path.join(a.dst, name), metadata={"format": "pt"})
    for k in rest: new_map[k] = name
    for f in set(wmap.values()): drop_cache(os.path.join(a.src, f))

    # Spot-check: fused tensors must equal the original per-expert weights.
    for li, e in ((0, 5), (L - 1, E - 1)):
        pre = f"model.layers.{li}.mlp.experts"
        with safe_open(os.path.join(a.dst, new_map[f"{pre}.gate_up_proj"]), framework="pt") as f:
            gu, dn = f.get_slice(f"{pre}.gate_up_proj")[e], f.get_slice(f"{pre}.down_proj")[e]
        I = gu.shape[0] // 2
        assert torch.equal(gu[:I], get(f"{pre}.{e}.gate_proj.weight")), (li, e, "gate")
        assert torch.equal(gu[I:], get(f"{pre}.{e}.up_proj.weight")), (li, e, "up")
        assert torch.equal(dn, get(f"{pre}.{e}.down_proj.weight")), (li, e, "down")
    print("spot-check OK: fused == original for (layer, expert) (0,5) and (last,last)", flush=True)

    total = sum(os.path.getsize(os.path.join(a.dst, f)) for f in set(new_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    for f in os.listdir(a.src):  # config, tokenizer, generation config, provenance
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), a.dst)
    open(os.path.join(a.dst, "SOURCE_INFO"), "a").write(f"fused_from={a.src}\n")
    print("RESULT " + json.dumps({"kind": "fuse_experts", "tensors": len(new_map), "gib": round(total / 2**30, 1),
                                  "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
