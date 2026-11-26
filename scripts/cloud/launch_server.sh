#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_dcu.sh"

GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[INFO] Launching LeRobot gRPC policy_server on DCU ${GPU_ID} (${HOST}:${PORT})..."

"${DCU_VENV_DIR}/bin/python" -m lerobot.async_inference.policy_server \
    --host="${HOST}" \
    --port="${PORT}" \
    --fps="${FPS}" \
    --inference_latency="${INFERENCE_LATENCY}" \
    --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}"