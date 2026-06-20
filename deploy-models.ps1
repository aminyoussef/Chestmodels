# Upload local model files to VPS via SCP (run on Windows from Ai/ folder)
param(
    [string]$Server = "root@187.127.71.11",
    [string]$RemoteRoot = "/docker/chestmodels"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$files = @(
    @{ Local = "$root\EEG\eeg_web_output\eeg_model.pt"; Remote = "$RemoteRoot/EEG/eeg_web_output/eeg_model.pt" },
    @{ Local = "$root\EEG\eeg_web_output\scaler.pkl"; Remote = "$RemoteRoot/EEG/eeg_web_output/scaler.pkl" },
    @{ Local = "$root\Chest\models\best_model.pth"; Remote = "$RemoteRoot/Chest/models/best_model.pth" }
)

foreach ($f in $files) {
    if (-not (Test-Path $f.Local)) {
        throw "Missing file: $($f.Local)"
    }
    Write-Host "Uploading $($f.Local) ..."
    scp $f.Local "${Server}:$($f.Remote)"
}

Write-Host ""
Write-Host "Done. On VPS run:"
Write-Host "  cd $RemoteRoot && docker compose restart"
