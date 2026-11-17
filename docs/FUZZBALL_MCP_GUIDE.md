# CIQ Fuzzball 4.2 MCP & Auto-Discovery Guide

Crucible includes a native JSON-RPC 2.0 stdio client (`src/crucible/fuzzball/mcp_client.py`) built specifically for **CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve --allow-write --allow-exec`)**, alongside `fuzzball` CLI and offline simulator backends.

---

## 1. How Crucible Uses CIQ Fuzzball 4.2 MCP

When you run on a machine where `fuzzball` is installed and authenticated (`fuzzball login`), Crucible speaks directly to `fuzzball mcp serve` over stdio:

- **Cluster & Resource Discovery (`crucible discover`)**:
  - `cluster_list` — resolves active cluster context
  - `node_list` (`fields: ["id", "resource"]`) — inspects GPU (`nvidia.com/gpu`) and CPU core topology across GB10 nodes
  - `volume_list` — discovers active persistent VAST volumes and storage provisioners
  - `workflow_list` — checks status of `workflows/00-fetch.yaml` and active trials
- **Autonomous Trial Execution (`crucible run --backend fuzzball_mcp`)**:
  1. `workflow_validate` — validates each generated trial Fuzzfile (`{"valid": bool, "errors": [...]}`)
  2. `workflow_start` — launches the trial workflow (`{"workflowId": "..."}`)
  3. `workflow_wait` — blocks cleanly until terminal status
  4. `object_get` — retrieves compact numeric `metrics.json` (< 4 KB) egressed from the volume to `fuzzball://user/crucible/<run_id>/<trial_id>/metrics.json`

### CIQ Object Cache Egress Rule + Strict Data Isolation

Per CIQ Fuzzball 4.2 agent architecture, agents cannot read arbitrary files directly out of a persistent storage volume — outputs that an agent needs must egress to the Fuzzball object cache (`fuzzball://user/...` or `fuzzball://group/...`) for `object_get`.

Crucible satisfies both CIQ's rule and strict data isolation simultaneously:
- **Heavy & Sensitive Assets Stay on VAST**: Base weights (`/vast/models/...`), LAMMPS C++ raw/tokenized splits (`/vast/data/...`), and LoRA adapter checkpoints remain strictly inside the VAST persistent volume.
- **Compact Numeric Metrics Egress to Object Cache**: Rendered `v4` trial Fuzzfiles attach a volume `egress:` rule exporting *only* `summary.json` -> `fuzzball://user/crucible/<run_id>/<trial_id>/metrics.json`. Crucible's Fuzzfile validator (`validate_fuzzfile`) explicitly blocks any egress rule whose source path references dataset splits or model weights.

---

## 2. Cursor MCP Configuration Reference (Documentation Only)

If configuring Fuzzball 4.2 MCP in your local Cursor client:

```json
{
  "mcpServers": {
    "fuzzball": {
      "command": "fuzzball",
      "args": ["mcp", "serve", "--allow-write", "--allow-exec"]
    }
  }
}
```

---

## 3. Workstation Commands

```bash
# Auto-discover active cluster, VAST volumes, and GB10 nodes (via fuzzball mcp serve / 00-fetch.yaml)
crucible discover

# Run the 4-stage autoresearch loop natively through `fuzzball mcp serve --allow-write --allow-exec`
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball_mcp --output-dir runs/live-mcp

# Or run through the direct `fuzzball workflow` CLI backend
crucible run --config configs/demo_lammps_30b.yaml --backend fuzzball --output-dir /vast/runs/live-cli
```
