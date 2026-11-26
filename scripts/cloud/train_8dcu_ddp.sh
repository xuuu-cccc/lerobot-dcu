#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_dcu.sh"

# Configurable parameters (aligned with 8x K100_AI verified run)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REPO_ID="${REPO_ID:-local/so101_demo}"
DATASET_ROOT="${DATASET_ROOT:-/output/datasets/so101_demo}"
OUTPUT_BASE="${OUTPUT_BASE:-/output/lerobot-dcu/outputs}"
RUN_NAME="${RUN_NAME:-act_so101_8xk100_$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-8}"       # Global batch size = 8 * 8 = 64
STEPS="${STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-2500}"

# Verify offline assets
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "[ERROR] Dataset meta/info.json not found under: ${DATASET_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth" ]]; then
    echo "[WARN] Offline ResNet18 weights not found at ${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth" >&2
fi

export CUDA_VISIBLE_DEVICES

echo "[INFO] Starting ${NPROC_PER_NODE}-DCU DDP training: ${RUN_NAME}"

"${DCU_VENV_DIR}/bin/python" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_BASE}/${RUN_NAME}" \
    --job_name="${RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --log_freq="${LOG_FREQ}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable=false