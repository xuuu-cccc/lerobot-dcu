#!/usr/bin/env bash
set -euo pipefail

# Hardware & Camera configuration
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-my_so101_follower}"
LEADER_ID="${LEADER_ID:-my_so101_leader}"
CAM_INDEX="${CAM_INDEX:-0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FPS="${CAM_FPS:-30}"

# Dataset configuration (must be an absolute path)
REPO_ID="${REPO_ID:-local/so101_demo}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/lerobot/data/so101_demo}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
RESET_TIME_S="${RESET_TIME_S:-10}"
SINGLE_TASK="${SINGLE_TASK:-pick and place}"

if [[ "${DATASET_ROOT}" == ~* ]]; then
    echo "[ERROR] DATASET_ROOT must be an absolute path (do not use '~'): ${DATASET_ROOT}" >&2
    exit 1
fi

echo "[INFO] Recording ${NUM_EPISODES} episodes to ${DATASET_ROOT}..."

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="{ top: {type: opencv, index_or_path: ${CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}} }" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.single_task="${SINGLE_TASK}" \
    --dataset.push_to_hub=false