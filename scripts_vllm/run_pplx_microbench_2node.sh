#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

export ARTIFACT_BASE="${ARTIFACT_BASE:-${SCRIPT_DIR}/runtime/pplx_micro_ki_auto_${SLURM_JOB_ID:-manual}_${STAMP}}"

exec bash "${SCRIPT_DIR}/run_microbench_2node.sh" --backend pplx "$@"
