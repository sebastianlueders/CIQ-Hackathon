#!/usr/bin/env bash
# scripts/run_team3.sh
# Turnkey launcher for Crucible Studio (Pick Any Model -> Drop Data -> Optimize -> Download Build).
#
# Default: ensures the `team3` VAST volume exists, opens http://127.0.0.1:8000 with the built-in
# Train / Validation / Held-Out Test benchmark pre-loaded so you can pick any model, drop files,
# click Optimize, and download your `.zip` build!
#
# Options:
#   ./scripts/run_team3.sh                         # interactive Studio at http://127.0.0.1:8000
#   ./scripts/run_team3.sh --auto-run              # also auto-launch a run immediately
#   ./scripts/run_team3.sh --backend fuzzball_mcp  # auto-launch live on Team 3 Fuzzball (queues if GPUs busy)
#   ./scripts/run_team3.sh --model Qwen/Qwen2.5-Coder-14B-Instruct --auto-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="${HOME}/bin:${REPO_ROOT}:${PATH}"

MODEL_ID=""
PROFILE="demo_30b"
BACKEND="auto"
AUTO_RUN=0
TEAM_ID="${CRUCIBLE_DEFINITION_ID:-auto}"
PORT="${CRUCIBLE_PORT:-8000}"
HOST="${CRUCIBLE_HOST:-127.0.0.1}"
OPEN_BROWSER=1
KEEP_SERVING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      AUTO_RUN=1
      shift 2
      ;;
    --model)
      MODEL_ID="$2"
      AUTO_RUN=1
      shift 2
      ;;
    --auto-run)
      AUTO_RUN=1
      shift
      ;;
    --studio-only)
      AUTO_RUN=0
      shift
      ;;
    --team|--definition-id)
      TEAM_ID="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=0
      shift
      ;;
    --exit-after-run)
      KEEP_SERVING=0
      AUTO_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# 1. Resolve existing Python virtual environment (zero extra pip packages needed!)
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  VENV_BIN="${REPO_ROOT}/.venv/bin"
elif command -v crucible >/dev/null 2>&1; then
  VENV_BIN="$(dirname "$(command -v crucible)")"
else
  echo "[crucible] Creating local Python virtual environment at ${REPO_ROOT}/.venv ..."
  python3 -m venv "${REPO_ROOT}/.venv"
  VENV_BIN="${REPO_ROOT}/.venv/bin"
  "${VENV_BIN}/pip" install --upgrade pip
  "${VENV_BIN}/pip" install -e ".[ui,test]"
fi

CRUCIBLE="${VENV_BIN}/crucible"

# 2. Discover active Fuzzball team (`team3`) and check whether venue/cluster Wi-Fi is reachable
DISCOVER_JSON="$("$CRUCIBLE" discover --json)"
if [[ "$TEAM_ID" == "auto" ]]; then
  TEAM_ID="$("${VENV_BIN}/python" -c 'import json,sys; print(json.loads(sys.stdin.read())["cluster_config"]["definition_id"])' <<<"$DISCOVER_JSON")"
fi
CLUSTER_REACHABLE="$("${VENV_BIN}/python" -c 'import json,sys; print("1" if json.loads(sys.stdin.read())["discovery_report"].get("cluster_reachable") else "0")' <<<"$DISCOVER_JSON")"
export CRUCIBLE_DEFINITION_ID="$TEAM_ID"

# 3. Auto-select execution backend based on live cluster reachability
if [[ "$BACKEND" == "auto" ]]; then
  if [[ "$CLUSTER_REACHABLE" == "1" ]] && command -v fuzzball >/dev/null 2>&1; then
    if fuzzball mcp serve --help >/dev/null 2>&1; then
      BACKEND="fuzzball_mcp"
    else
      BACKEND="fuzzball"
    fi
  else
    BACKEND="simulator"
  fi
fi

if [[ "$CLUSTER_REACHABLE" == "1" ]] && command -v fuzzball >/dev/null 2>&1; then
  VOL_OUT="$(fuzzball volume list "$TEAM_ID" 2>/dev/null || true)"
  if echo "$VOL_OUT" | grep -qw "runs"; then
    export CRUCIBLE_VAST_VOLUME="runs"
  elif echo "$VOL_OUT" | grep -qw "crucible-vast"; then
    export CRUCIBLE_VAST_VOLUME="crucible-vast"
  else
    echo "[crucible] Ensuring persistent volume 'runs' exists on provisioner '${TEAM_ID}' ..."
    fuzzball volume create "$TEAM_ID" runs >/dev/null 2>&1 || fuzzball volume provisioner scan "$TEAM_ID" >/dev/null 2>&1 || true
    export CRUCIBLE_VAST_VOLUME="runs"
  fi
  echo "[crucible] LIVE CLUSTER CONNECTED -> team='${TEAM_ID}', backend='${BACKEND}', volume='${CRUCIBLE_VAST_VOLUME}'"
  # Pre-stage Qwen/Qwen2.5-Coder-1.5B-Instruct (3.1 GB BF16, 0 GPUs used) in background onto VAST for fast live-demo baseline!
  ACTIVE_WFS="$(fuzzball workflow list 2>/dev/null || true)"
  if ! echo "$ACTIVE_WFS" | grep -q "crucible-stage-qwen2.5-1.5b"; then
    STAGE_YAML="$("${VENV_BIN}/python" -c 'from crucible.config import ClusterConfig; from crucible.fuzzball.discovery import discover_cluster_config; from crucible.fuzzball.renderer import render_model_staging_fuzzfile; c,_=discover_cluster_config(); print(render_model_staging_fuzzfile(c, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))')"
    STAGE_TMP="${REPO_ROOT}/runs/stage-qwen25-1.5b.yaml"
    mkdir -p "${REPO_ROOT}/runs"
    printf "%s\n" "$STAGE_YAML" > "$STAGE_TMP"
    echo "[crucible] Pre-loading fast live-demo baseline model (Qwen/Qwen2.5-Coder-1.5B-Instruct, 0 GPUs used) onto VAST..."
    fuzzball workflow start "$STAGE_TMP" --name crucible-stage-qwen2.5-1.5b >/dev/null 2>&1 || true
  fi
else
  export CRUCIBLE_VAST_VOLUME="runs"
  echo "[crucible] Cluster Wi-Fi not currently connected -> starting Client Studio in GB10 Empirical Twin mode (team='${TEAM_ID}', auto-switches to live Fuzzball MCP when on cluster Wi-Fi)."
fi
"$CRUCIBLE" discover

# 4. Start Crucible Studio on http://127.0.0.1:8000
RUN_DIR="${REPO_ROOT}/runs/${TEAM_ID}-live"
mkdir -p "$RUN_DIR"

echo "[crucible] Starting Crucible Studio at http://${HOST}:${PORT} ..."
"$CRUCIBLE" serve --run-dir "$RUN_DIR" --host "$HOST" --port "$PORT" >"${RUN_DIR}/studio.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "http://${HOST}:${PORT}" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://${HOST}:${PORT}" >/dev/null 2>&1 || true
  fi
fi

# 5. Optional auto-run when `--auto-run` / `--backend` / `--exit-after-run` is passed
if [[ "$AUTO_RUN" -eq 1 ]]; then
  echo "[crucible] Launching optimization (team=${TEAM_ID}, backend=${BACKEND}) ..."
  PAYLOAD="$("${VENV_BIN}/python" -c 'import json,sys; print(json.dumps({"profile":sys.argv[1],"model_id":sys.argv[2] or None,"backend":sys.argv[3],"fuzzfile_version":"v4","definition_id":sys.argv[4],"async_mode":True}))' "$PROFILE" "$MODEL_ID" "$BACKEND" "$TEAM_ID")"
  curl -fsS -X POST "http://${HOST}:${PORT}/api/run" -H "Content-Type: application/json" -d "$PAYLOAD" >/dev/null

  while true; do
    STATUS_JSON="$(curl -fsS "http://${HOST}:${PORT}/api/live-status")"
    STATE="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["state"])' <<<"$STATUS_JSON")"
    LABEL="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["stage_label"])' <<<"$STATUS_JSON")"
    echo "  [${STATE}] ${LABEL}"
    if [[ "$STATE" == "completed" ]]; then
      break
    fi
    if [[ "$STATE" == "failed" ]]; then
      ERR="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("error") or "")' <<<"$STATUS_JSON")"
      echo "[crucible] Run stopped with error: ${ERR}" >&2
      break
    fi
    sleep 1
  done
fi

echo ""
echo "[crucible] Studio ready at: http://${HOST}:${PORT}"
echo "[crucible] Pick any model, drop files (or use the pre-loaded benchmark), click Optimize, and download your .zip build!"
if [[ "$KEEP_SERVING" -eq 1 ]]; then
  echo "[crucible] Press Ctrl+C to stop Crucible Studio."
  wait "$SERVER_PID"
fi
