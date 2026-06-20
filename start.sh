#!/usr/bin/env bash
# Start EEG + Chest inference with one command (Linux / VPS)
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created Ai/.env — set EEG_SERVICE_API_KEY and CHEST_SERVICE_API_KEY, then run again."
  exit 1
fi

docker compose up -d --build

EEG_PORT="${EEG_HOST_PORT:-8000}"
CHEST_PORT="${CHEST_HOST_PORT:-5000}"

echo ""
echo "Medical Scans AI stack is starting."
echo "  EEG:   http://127.0.0.1:${EEG_PORT}/health"
echo "  Chest: http://127.0.0.1:${CHEST_PORT}/health"
echo ""
echo "Logs:  docker compose logs -f"
echo "Stop:  docker compose down"
