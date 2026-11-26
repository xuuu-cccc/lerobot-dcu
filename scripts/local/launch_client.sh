#!/usr/bin/env bash
set -euo pipefail

: "${PRETRAINED_PATH:?[ERROR] Please set PRETRAINED_PATH (cloud-side path to pretrained_model)}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-my_so101_follower}"
CAM_INDEX="${CAM_INDEX:-0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
FPS="${FPS:-30}"

POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN="${AGGREGATE_FN:-weighted_average}"

echo "[INFO] Connecting SO-101 (${FOLLOWER_PORT}) to policy_server at ${SERVER_ADDRESS}..."
echo "[INFO] Cloud model: ${PRETRAINED_PATH}"

# Pre-import lerobot_record to register so101_follower & opencv in LeRobot v0.4.4
python -c "import lerobot.scripts.lerobot_record, runpy; runpy.run_module('lerobot.async_inference.robot_client', run_name='__main__')" \
    --server_address="${SERVER_ADDRESS}" \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="{ top: {type: opencv, index_or_path: ${CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}} }" \
    --policy_type="${POLICY_TYPE}" \
    --pretrained_name_or_path="${PRETRAINED_PATH}" \
    --policy_device="${POLICY_DEVICE}" \
    --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
    --aggregate_fn_name="${AGGREGATE_FN}" \
    --fps="${FPS}"