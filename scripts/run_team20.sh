#!/usr/bin/env bash
# scripts/run_team20.sh
# Forwards to scripts/run_team3.sh (which auto-discovers the active Fuzzball context's team,
# e.g. `team3` for `team3@fb-hack.local`, unless `--team team20` is explicitly passed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_team3.sh" "$@"
