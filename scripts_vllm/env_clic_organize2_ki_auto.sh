#!/usr/bin/env bash

# Shared defaults for the clic_organize2 KI-auto validation scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CLIC_PPLX_KI_AUTO_DIR="${CLIC_PPLX_KI_AUTO_DIR:-${SCRIPT_DIR}}"
export CLIC_PROJECT_DIR="${CLIC_PROJECT_DIR:-/capstor/scratch/cscs/pzhao/KI_INFER_clic}"
export CLIC_ROOT="${CLIC_ROOT:-${CLIC_PROJECT_DIR}/clic_organize2}"
export CLIC_VLLM_PROJECT_DIR="${CLIC_VLLM_PROJECT_DIR:-${CLIC_PROJECT_DIR}}"
export CLIC_CONTAINER_ENV="${CLIC_CONTAINER_ENV:-${CLIC_PROJECT_DIR}/DOCKER/clic.toml}"
export CLIC_ENV_SCRIPT="${CLIC_ENV_SCRIPT:-${CLIC_PROJECT_DIR}/scripts_vllm_clic/env_clic_vllm.sh}"
export CLIC_PYTHON_SCRIPT="${CLIC_PYTHON_SCRIPT:-${SCRIPT_DIR}/capture_clic.py}"

export PPLX_PROJECT_DIR="${PPLX_PROJECT_DIR:-/capstor/scratch/cscs/pzhao/KI_INFER_0.16.0_pplxgarden}"
export PPLX_CONTAINER_ENV="${PPLX_CONTAINER_ENV:-${PPLX_PROJECT_DIR}/DOCKER/pplx_garden.toml}"
export PPLX_ENV_SCRIPT="${PPLX_ENV_SCRIPT:-${PPLX_PROJECT_DIR}/scripts_vllm_pplx_garden/env_pplx_garden_vllm.sh}"
export PPLX_PYTHON_SCRIPT="${PPLX_PYTHON_SCRIPT:-${SCRIPT_DIR}/capture_pplx.py}"

export CLIC_TRANSPORT="${CLIC_TRANSPORT:-KI}"
export CLIC_KI_CHAINED="${CLIC_KI_CHAINED:-1}"
export CLIC_KI_CHAIN_SIGNALS="${CLIC_KI_CHAIN_SIGNALS:-1}"
export CLIC_KI_CHAIN_DB_BATCH="${CLIC_KI_CHAIN_DB_BATCH:-64}"
export CLIC_KI_CHAIN_FILL_PUTS="${CLIC_KI_CHAIN_FILL_PUTS:-512}"
export CLIC_KI_CQ_DRAIN="${CLIC_KI_CQ_DRAIN:-1}"
export CLIC_KI_STATS="${CLIC_KI_STATS:-0}"
export CLIC_KI_STATS_PRINT_EVERY="${CLIC_KI_STATS_PRINT_EVERY:-0}"

# Dispatch/combine optimization is selected inside CLIC when the KI backend,
# BF16 tensors, bounded top-k, and batched layout are all present.

export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-ofi}"
export CLIC_NCCL_DEBUG="${CLIC_NCCL_DEBUG:-${NCCL_DEBUG:-WARN}}"
export NCCL_DEBUG="${CLIC_NCCL_DEBUG}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_NET_GDR_READ="${NCCL_NET_GDR_READ:-0}"
export NCCL_IB_GDR_LEVEL="${NCCL_IB_GDR_LEVEL:-0}"
export OFI_NCCL_DISABLE_GDR_REQUIRED_CHECK="${OFI_NCCL_DISABLE_GDR_REQUIRED_CHECK:-1}"
export OFI_NCCL_DISABLE_DMABUF="${OFI_NCCL_DISABLE_DMABUF:-1}"
