#!/usr/bin/env bash
# scripts/run_team3.sh
# Turnkey launcher for Crucible on Team 3 GPUs (VAST storage provisioner `use: team3`).
# Auto-detects the logged-in Fuzzball team from `context_show` / `fuzzball context list` (defaulting to `team3`).
#
# Usage:
#   ./scripts/run_team3.sh                        # auto-selects fuzzball_mcp if `fuzzball` is available, else simulator
#   ./scripts/run_team3.sh --backend fuzzball_mcp # force live Fuzzball 4.2 MCP execution
#   ./scripts/run_team3.sh --team team20          # override VAST storage provisioner / team ID
#   ./scripts/run_team3.sh --profile stretch_80b  # run 2-node 80B stretch profile
#   ./scripts/run_team3.sh --no-open              # skip opening browser automatically

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="${REPO_ROOT}:${PATH}"

PROFILE="demo_30b"
BACKEND="auto"
TEAM_ID="${CRUCIBLE_DEFINITION_ID:-auto}"
PORT="${CRUCIBLE_PORT:-8000}"
HOST="${CRUCIBLE_HOST:-127.0.0.1}"
OPEN_BROWSER=1
KEEP_SERVING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
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
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# 1. Resolve Python virtual environment
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

# 2. Auto-select execution backend
if [[ "$BACKEND" == "auto" ]]; then
  if command -v fuzzball >/dev/null 2>&1; then
    if fuzzball mcp serve --help >/dev/null 2>&1; then
      BACKEND="fuzzball_mcp"
      echo "[crucible] Detected 'fuzzball mcp serve' -> selecting live CIQ Fuzzball 4.2 MCP backend"
    else
      BACKEND="fuzzball"
      echo "[crucible] Detected 'fuzzball' CLI -> selecting live Fuzzball CLI backend"
    fi
  else
    BACKEND="simulator"
    echo "[crucible] 'fuzzball' CLI not found -> selecting calibrated GB10 simulator"
  fi
fi

# 3. Discover active Fuzzball team (`team3` from `team3@fb-hack.local`), volumes, and GB10 nodes
if [[ "$TEAM_ID" == "auto" ]]; then
  DISCOVER_JSON="$("$CRUCIBLE" discover --json)"
  TEAM_ID="$("${VENV_BIN}/python" -c 'import json,sys; print(json.loads(sys.stdin.read())["cluster_config"]["definition_id"])' <<<"$DISCOVER_JSON")"
fi
export CRUCIBLE_DEFINITION_ID="$TEAM_ID"

echo "[crucible] Active provisioner/team: ${TEAM_ID}"
"$CRUCIBLE" discover
"$CRUCIBLE" validate-yaml workflows/00-fetch.yaml
"$CRUCIBLE" validate-yaml workflows/crucible-studio.yaml
"$CRUCIBLE" validate-yaml workflows/examples/final-lammps-qwen3-30b.yaml

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

# 5. Trigger autoresearch run via Studio API so browser streams live updates
echo "[crucible] Launching 4-stage autoresearch run (team=${TEAM_ID}, profile=${PROFILE}, backend=${BACKEND}) ..."
curl -fsS -X POST "http://${HOST}:${PORT}/api/run" \
  -H "Content-Type: application/json" \
  -d "{\"profile\":\"${PROFILE}\",\"backend\":\"${BACKEND}\",\"fuzzfile_version\":\"v4\",\"definition_id\":\"${TEAM_ID}\",\"async_mode\":true}" \
  >/dev/null

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

echo ""
echo "[crucible] Live UI running at: http://${HOST}:${PORT}"
if [[ "$KEEP_SERVING" -eq 1 ]]; then
  echo "[crucible] Press Ctrl+C to stop Crucible Studio."
  wait "$SERVER_PID"
fi
