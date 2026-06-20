#!/usr/bin/env bash
# Sync EEG + Chest source files from GitHub (when Hostinger deploy is incomplete)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="${GITHUB_RAW:-https://raw.githubusercontent.com/aminyoussef/Chestmodels/main}"

fetch() {
  local rel="$1"
  local dest="$ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  echo ">> $rel"
  curl -fsSL -o "$dest" "$REPO/$rel"
}

fetch "EEG/api.py"
fetch "EEG/Dockerfile"
fetch "EEG/requirements-docker.txt"
fetch "EEG/requirements.txt"
fetch "Chest/app.py"
fetch "Chest/Dockerfile"
fetch "Chest/requirements.txt"

echo "Source sync complete."
