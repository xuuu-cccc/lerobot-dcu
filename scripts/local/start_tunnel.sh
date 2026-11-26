#!/usr/bin/env bash
set -euo pipefail

: "${CLOUD_HOST:?[ERROR] Please set CLOUD_HOST (e.g., export CLOUD_HOST=your.cloud.host)}"
: "${CLOUD_PORT:?[ERROR] Please set CLOUD_PORT (e.g., export CLOUD_PORT=31941)}"

CLOUD_USER="${CLOUD_USER:-root}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
REMOTE_PORT="${REMOTE_PORT:-8080}"

echo "[INFO] Opening SSH tunnel: 127.0.0.1:${LOCAL_PORT} -> ${CLOUD_HOST}:${REMOTE_PORT} (SSH port ${CLOUD_PORT})"
echo "[INFO] Keep this terminal open. Press Ctrl+C to close the tunnel."

ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${CLOUD_USER}@${CLOUD_HOST}" \
    -p "${CLOUD_PORT}"