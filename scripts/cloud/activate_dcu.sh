#!/usr/bin/env bash
# Usage: source scripts/cloud/activate_dcu.sh

export DCU_VENV_DIR="${DCU_VENV_DIR:-/output/lerobot-dcu/.venv}"
export LEROBOT_SRC_DIR="${LEROBOT_SRC_DIR:-/openbayes/home/lerobot-dcu/lerobot/src}"
export TORCH_HOME="${TORCH_HOME:-/output/torch_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -f "${DCU_VENV_DIR}/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found at: ${DCU_VENV_DIR}" >&2
    return 1 2>/dev/null || exit 1
fi

source "${DCU_VENV_DIR}/bin/activate"

VENV_SITE_PACKAGES=$("${DCU_VENV_DIR}/bin/python" -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="${LEROBOT_SRC_DIR}:${VENV_SITE_PACKAGES}:/usr/local/lib/python3.10/dist-packages"

echo "[INFO] Activated Hygon DCU LeRobot environment: ${DCU_VENV_DIR}"
echo "[INFO] TORCH_HOME=${TORCH_HOME}"