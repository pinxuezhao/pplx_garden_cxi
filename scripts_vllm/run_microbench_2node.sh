#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_clic_organize2_ki_auto.sh
source "${SCRIPT_DIR}/env_clic_organize2_ki_auto.sh"

PROFILE="qwen3-30b"
TOKENS_CSV="64,128,256"
WARMUP="10"
ITERS="50"
BACKEND_FILTER="both"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_BASE="${ARTIFACT_BASE:-${SCRIPT_DIR}/runtime/micro_ki_auto_${SLURM_JOB_ID:-manual}_${STAMP}}"

usage() {
  cat <<EOF
Usage: bash ${BASH_SOURCE[0]} [options]

Run CLIC clic_organize2 KI-auto and pplx-garden dispatch/combine microbenchmarks.

Options:
  --profile NAME       qwen3-30b or qwen3-next-80b (default: ${PROFILE})
  --backend NAME       both, clic, or pplx (default: ${BACKEND_FILTER})
  --tokens LIST        Comma-separated token counts (default: ${TOKENS_CSV})
  --warmup N           Warmup iterations (default: ${WARMUP})
  --iters N            Timed iterations (default: ${ITERS})
  --artifact-base PATH Output directory
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#--profile=}"
      shift
      ;;
    --backend)
      BACKEND_FILTER="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND_FILTER="${1#--backend=}"
      shift
      ;;
    --tokens)
      TOKENS_CSV="$2"
      shift 2
      ;;
    --tokens=*)
      TOKENS_CSV="${1#--tokens=}"
      shift
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --warmup=*)
      WARMUP="${1#--warmup=}"
      shift
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --iters=*)
      ITERS="${1#--iters=}"
      shift
      ;;
    --artifact-base)
      ARTIFACT_BASE="$2"
      shift 2
      ;;
    --artifact-base=*)
      ARTIFACT_BASE="${1#--artifact-base=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${BACKEND_FILTER}" in
  both|clic|pplx)
    ;;
  *)
    echo "ERROR: --backend must be one of: both, clic, pplx." >&2
    exit 2
    ;;
esac

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run inside a 2-node Slurm allocation." >&2
  exit 2
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES="${NNODES:-2}"
TASKS_PER_NODE="${TASKS_PER_NODE:-4}"
if [ "${#ALLOC_NODES[@]}" -lt "${NNODES}" ]; then
  echo "ERROR: need ${NNODES} nodes, got ${#ALLOC_NODES[@]}." >&2
  exit 2
fi

NODES=("${ALLOC_NODES[@]:0:${NNODES}}")
NODELIST_CSV="$(IFS=,; echo "${NODES[*]}")"
MASTER_ADDR="${MASTER_ADDR:-${NODES[0]}}"
WORLD_SIZE="$((NNODES * TASKS_PER_NODE))"
LOCAL_WORLD_SIZE="${TASKS_PER_NODE}"
IFS=',' read -r -a TOKEN_VALUES <<< "${TOKENS_CSV}"

mkdir -p "${ARTIFACT_BASE}"
SUMMARY_TSV="${ARTIFACT_BASE}/micro_summary.tsv"
printf "backend\tkind\ttokens\tok\tmedian_us_max_rank\tp99_us_max_rank\tjson\n" > "${SUMMARY_TSV}"

export MASTER_ADDR WORLD_SIZE LOCAL_WORLD_SIZE
export SLURM_NTASKS="${WORLD_SIZE}"
export SLURM_NTASKS_PER_NODE="${TASKS_PER_NODE}"

run_case() {
  local backend="$1"
  local kind="$2"
  local tokens="$3"
  local json_path="${ARTIFACT_BASE}/${backend}_${kind}_${tokens}.json"
  local log_path="${ARTIFACT_BASE}/${backend}_${kind}_${tokens}.log"
  local master_port
  local container_env
  local container_workdir
  local env_script
  local python_script
  local -a extra_args

  master_port="$((29800 + SLURM_JOB_ID % 10000 + tokens % 100))"
  if [ "${kind}" = "combine" ]; then
    master_port="$((master_port + 100))"
  fi
  if [ "${backend}" = "pplx" ]; then
    master_port="$((master_port + 200))"
  fi

  if [ "${backend}" = "clic" ]; then
    container_env="${CLIC_CONTAINER_ENV}"
    container_workdir="${CLIC_PROJECT_DIR}"
    env_script="${CLIC_ENV_SCRIPT}"
    python_script="${SCRIPT_DIR}/minimal_clic_${kind}.py"
    extra_args=(--transport "${CLIC_TRANSPORT}")
    export CLIC_BOOTSTRAP_DIR="${CLIC_BOOTSTRAP_DIR:-${CLIC_PROJECT_DIR}/runtime/clic-bootstrap}"
    export CLIC_BOOTSTRAP_ID="ki_auto_micro_${SLURM_JOB_ID}_${backend}_${kind}_${tokens}_${master_port}"
    rm -rf "${CLIC_BOOTSTRAP_DIR:?}/${CLIC_BOOTSTRAP_ID:?}"
    mkdir -p "${CLIC_BOOTSTRAP_DIR}/${CLIC_BOOTSTRAP_ID}"
  else
    container_env="${PPLX_CONTAINER_ENV}"
    container_workdir="${PPLX_PROJECT_DIR}"
    env_script="${PPLX_ENV_SCRIPT}"
    python_script="${PPLX_PROJECT_DIR}/scripts_vllm_pplx_garden/minimal_pplx_garden_${kind}.py"
    extra_args=(--node-group-size "${TASKS_PER_NODE}" --sync-timing)
  fi

  echo "BENCH backend=${backend} kind=${kind} tokens=${tokens}"
  MASTER_PORT="${master_port}" \
  PYTHON_SCRIPT="${python_script}" \
  CASE_BACKEND="${backend}" \
  CASE_ENV_SCRIPT="${env_script}" \
  srun --overlap \
    --nodes="${NNODES}" \
    --nodelist="${NODELIST_CSV}" \
    --ntasks-per-node="${TASKS_PER_NODE}" \
    --gpus-per-task=1 \
    --export=ALL \
    --mpi=pmix \
    -ul \
    --environment="${container_env}" \
    --container-workdir="${container_workdir}" \
    bash -lc '
      set -euo pipefail
      if [ "${CASE_BACKEND}" = "clic" ]; then
        cd "${CLIC_PROJECT_DIR}"
        source "${CASE_ENV_SCRIPT}"
        export CLIC_CXI_DEV="cxi${SLURM_LOCALID}"
      else
        cd "${PPLX_PROJECT_DIR}"
        source "${CASE_ENV_SCRIPT}"
      fi
      export NCCL_DEBUG="${CLIC_NCCL_DEBUG:-WARN}"
      export RANK="${SLURM_PROCID}"
      export LOCAL_RANK="${SLURM_LOCALID}"
      exec python3 "${PYTHON_SCRIPT}" "$@"
    ' bash \
      --profile "${PROFILE}" \
      --routing random \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --num-tokens "${tokens}" \
      --max-num-tokens "${tokens}" \
      --json "${json_path}" \
      "${extra_args[@]}" \
    > "${log_path}" 2>&1

  python3 - "${SUMMARY_TSV}" "${backend}" "${kind}" "${tokens}" "${json_path}" <<'PY'
import json
import sys

summary, backend, kind, tokens, path = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(summary, "a", encoding="utf-8") as f:
    f.write(
        "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\n".format(
            backend,
            kind,
            tokens,
            int(bool(data.get("ok"))),
            float(data.get("median_us_max_rank", 0.0)),
            float(data.get("p99_us_max_rank", 0.0)),
            path,
        )
    )
PY
}

for tokens in "${TOKEN_VALUES[@]}"; do
  if [ "${BACKEND_FILTER}" = "both" ] || [ "${BACKEND_FILTER}" = "clic" ]; then
    run_case clic dispatch "${tokens}"
    run_case clic combine "${tokens}"
  fi
  if [ "${BACKEND_FILTER}" = "both" ] || [ "${BACKEND_FILTER}" = "pplx" ]; then
    run_case pplx dispatch "${tokens}"
    run_case pplx combine "${tokens}"
  fi
done

echo "MICRO_SUMMARY=${SUMMARY_TSV}"
