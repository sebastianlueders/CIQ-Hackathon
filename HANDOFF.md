# Alchemy Cluster & Architecture Handoff (`HANDOFF.md`)

**Team 3**: Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania  
**Track**: Multi-Node Training (NVIDIA GB10 · CIQ Fuzzball · VAST Data)

---

## 1. Team 3 Live Cluster Facts (`team3@fb-hack.local`) & VAST Layout

- **Fuzzball Endpoint & Context**: `api.fb-hack.civilfritz.net:443` (`team3@fb-hack.local`, `definitionId: team3`, VAST storage provisioner `use: team3`).
- **Compute Nodes (2x NVIDIA GB10 Grace Blackwell)**:
  - `CIQ-CHI-SPARK05` (`192.168.99.14/7331`) — `1x NVIDIA GB10` (`NVIDIA-GB10`, Blackwell, `sm_121`, 48 SMs), `nvidia.com/gpu: 1`, `20` cores (`cpu/arm64/sve: 1`), `127.9 GiB` unified LPDDR5X
  - `CIQ-CHI-SPARK06` (`192.168.99.15/7331`) — `1x NVIDIA GB10` (`NVIDIA-GB10`, Blackwell, `sm_121`, 48 SMs), `nvidia.com/gpu: 1`, `20` cores (`cpu/arm64/sve: 1`), `127.9 GiB` unified LPDDR5X
  - **Team 3 Active Pool**: `2x nvidia.com/gpu`, `40` cores, `255.8 GiB` unified memory, `200GbE RoCE` (`NCCL`)
- **Pre-Cached `linux/arm64` + `sm_121` Container Images**:
  - Training / Ingest / Eval / Live Inference: `docker://nvcr.io/nvidia/pytorch:26.10-py3`
  - Serving: `docker://vllm/vllm-openai:v0.33.0`
  - Lightweight Fetch / Helpers: `docker://python:3.12-slim`
- **Shared Storage (VAST Data)**: Persistent Fuzzball volumes on `use: team3` (`runs`, `models`, `data`, `envs`), mounted at `/vast`, `/models`, `/data`, and `/envs` across both Spark nodes (`1.64 GB/s` single-stream safetensors read).

---

## 2. CIQ Fuzzball v4 Rules Enforced by `crucible validate-yaml`

1. **Single Allocation Namespace**: Files, volumes, jobs, and services never share names.
2. **Strict Annotation Allowlist**: Only `fuzzball.io/provisioner_definition_id` (and platform-enumerated keys) may use the reserved `fuzzball.io/*` prefix.
3. **Hardware Envelope Gate**: Rejects any job/service requesting `> 1` GPU/node, `> 20` cores/node, `> 127.9 GiB`/node, or `multinode.nodes > 2`.
4. **Data-Isolation Egress Gate**: Blocks any volume `egress:` rule whose source path references raw LAMMPS files, JSONL splits, or adapter/model weights.

---

## 3. Quick Commands (`team3`)

```bash
# Start the Enterprise Client Studio on http://127.0.0.1:8000 (auto-detects team3 + cluster Wi-Fi)
./scripts/run_team3.sh --studio-only

# One-command launch + live Fuzzball MCP optimization on Team 3 GPUs
./scripts/run_team3.sh --model Qwen/Qwen2.5-Coder-1.5B-Instruct --backend fuzzball_mcp --time-budget 5 --cost-budget 3.50

# Run via CLI directly against CIQ Fuzzball MCP
alchemy run --config configs/demo_lammps_30b.yaml --backend fuzzball_mcp --output-dir runs/team3-01

# Validate all Fuzzfiles against Team 3 hardware bounds
alchemy validate-yaml workflows/00-fetch.yaml
alchemy validate-yaml workflows/crucible-studio.yaml
alchemy validate-yaml workflows/examples/final-lammps-qwen3-30b.yaml
```
