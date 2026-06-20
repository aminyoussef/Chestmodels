# Start EEG + Chest inference with one command (Windows)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created Ai\.env — set API keys, then run again."
    exit 1
}

docker compose up -d --build

$eegPort = if ($env:EEG_HOST_PORT) { $env:EEG_HOST_PORT } else { "8000" }
$chestPort = if ($env:CHEST_HOST_PORT) { $env:CHEST_HOST_PORT } else { "5000" }

Write-Host ""
Write-Host "Medical Scans AI stack is starting."
Write-Host "  EEG:   http://127.0.0.1:${eegPort}/health"
Write-Host "  Chest: http://127.0.0.1:${chestPort}/health"
Write-Host ""
Write-Host "Logs:  docker compose logs -f"
Write-Host "Stop:  docker compose down"
