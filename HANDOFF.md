# Crucible Cluster & Architecture Handoff (`HANDOFF.md`)

**Team 20**: Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania  
**Track**: Multi-Node Training (NVIDIA GB10 · CIQ Fuzzball · VAST Data)

---

## 1. Team 20 Live Cluster Facts (`definitionId: team20`) & VAST Layout

- **Provisioner Definition**: `definitionId: team20` — pinned in Fuzzball `v4` workflows via the auto-exempt platform annotation `fuzzball.io/provisioner_definition_id: team20`.
- **Compute Nodes (2 READY, idle)**:
  - `CIQ-CHI-SPARK39` (`192.168.99.48/7331`) — `1x NVIDIA GB10` (`NVIDIA-GB10`, Blackwell, `sm_121`), `nvidia.com/gpu: 1`, `20` cores (`cpu/arm64/sve: 1`), `127.9 GiB` unified LPDDR5X
  - `CIQ-CHI-SPARK40` (`192.168.99.49/7331`) — `1x NVIDIA GB10` (`NVIDIA-GB10`, Blackwell, `sm_121`), `nvidia.com/gpu: 1`, `20` cores (`cpu/arm64/sve: 1`), `127.9 GiB` unified LPDDR5X
  - **Team 20 Total**: `2x nvidia.com/gpu`, `40` cores, `255.8 GiB`
- **Safe Per-Node Container Envelope (`16 cores / 112GiB`)**:
  - GPU jobs/services request `cores: 16`, `memory.size: 112GiB`, `devices: { nvidia.com/gpu: 1, cpu/arm64/sve: 1 }`.
  - Leaving `4` cores and `15.9 GiB` headroom per node guarantees Fuzzball never rejects for Substrate/host overhead and allows lightweight CPU helper jobs (`bench-*` at `2c / 8GiB`, `crucible-studio` at `2c / 8GiB`) to co-schedule cleanly even during 2-node jobs.
- **Verified `linux/arm64` + `sm_121` NGC Container Images**:
  - Training / Ingest / Eval: `docker://nvcr.io/nvidia/pytorch:26.03-py3` (CUDA 13.2 + PyTorch 2.11 verified clean on DGX Spark GB10; `TORCH_CUDA_ARCH_LIST="12.0;12.1"`).
  - Serving: `docker://nvcr.io/nvidia/vllm:26.04-py3` (multi-arch `arm64` + `sm_121` PyTorch; launched with `--attention-backend TRITON_ATTN` and `FLASHINFER_DISABLE_VERSION_CHECK=1`).
- **Shared Storage (VAST Data)**: Persistent Fuzzball volume `crucible-vast` (`provisioner: vast`) mounted at `/vast` across both Spark nodes, with compact numeric `summary.json` egressed to `fuzzball://user/crucible/<run_id>/<trial_id>/metrics.json` for `object_get` (zero raw dataset/weight egress).

---

## 2. CIQ Fuzzball v4 Rules Enforced by `crucible validate-yaml`

1. **Single Allocation Namespace**: Volumes, jobs, and services never share names.
2. **Strict Annotation Allowlist**: Only `fuzzball.io/provisioner_definition_id` (and platform-enumerated keys) use the reserved `fuzzball.io/*` prefix so scheduler annotation matching never silently rejects `team20` nodes.
3. **Hardware Envelope Gate**: Rejects any job/service requesting `> 1` GPU/node, `> 20` cores/node, `> 127.9 GiB`/node, or `multinode.nodes > 2`.
4. **Data-Isolation Egress Gate**: Blocks any volume `egress:` rule whose source path references raw LAMMPS files, JSONL splits, or adapter/model weights.

---

## 3. Quick Commands (`team20`)

```bash
# One-command launch: discover team20, start Live Studio on http://127.0.0.1:8000, and run
./scripts/run_team20.sh

# Start the Live Web Studio on http://127.0.0.1:8000 and launch fuzzball_mcp runs from the browser
crucible serve --port 8000

# Run via CLI directly against CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve --allow-write --allow-exec`)
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball_mcp --output-dir runs/team20-01

# Validate all Fuzzfiles against Team 20 hardware bounds
crucible validate-yaml workflows/00-fetch.yaml
crucible validate-yaml workflows/crucible-studio.yaml
crucible validate-yaml workflows/examples/final-lammps-qwen3-30b.yaml
```
