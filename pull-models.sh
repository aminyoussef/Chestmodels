#!/usr/bin/env bash
# Download model weights from GitHub into the paths expected by docker-compose.yml
# Run on VPS:  cd /docker/chestmodels && bash pull-models.sh
set -euo pipefail

REPO="${MODEL_REPO:-https://raw.githubusercontent.com/aminyoussef/Chestmodels/main}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT/EEG/eeg_web_output" "$ROOT/Chest/models"

download() {
  local url="$1"
  local dest="$2"
  echo ">> $dest"
  curl -fsSL --retry 3 --retry-delay 5 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  ls -lh "$dest"
}

download "$REPO/EEG/eeg_web_output/eeg_model.pt" "$ROOT/EEG/eeg_web_output/eeg_model.pt"
download "$REPO/EEG/eeg_web_output/scaler.pkl" "$ROOT/EEG/eeg_web_output/scaler.pkl"
download "$REPO/Chest/models/best_model.pth" "$ROOT/Chest/models/best_model.pth"

echo ""
echo "Models ready. Restarting containers..."
cd "$ROOT"
docker compose restart

echo ""
echo "Health checks:"
curl -fsS "http://127.0.0.1:${EEG_HOST_PORT:-8000}/health" && echo ""
curl -fsS "http://127.0.0.1:${CHEST_HOST_PORT:-5000}/health" && echo ""
