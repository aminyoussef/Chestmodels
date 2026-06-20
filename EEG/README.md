# EEG Dementia Classifier (FastAPI)

PyTorch + MNE service for **AD / FTD / CN** classification from EEG files.

## Model artifacts

Place (or keep) in `eeg_web_output/`:

- `eeg_model.pt`
- `scaler.pkl`
- `reshape_params.json` (reference only)

Default `EEG_MODEL_DIR` points to `eeg_web_output/` next to `api.py`.

## Local run

```bash
cd Ai/EEG
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set EEG_SERVICE_API_KEY=dev-secret
python api.py
```

Open http://localhost:8000/docs

## Environment

| Variable | Description |
|----------|-------------|
| `EEG_MODEL_DIR` | Folder containing `eeg_model.pt` and `scaler.pkl` |
| `EEG_SERVICE_API_KEY` | If set, `POST /predict` requires header `X-EEG-Service-Key` |
| `PORT` | HTTP port (default 8000) |

## Supported uploads

`.edf`, `.set`, `.bdf`, `.fif`, `.cnt`

## Deploy (Docker on VPS) — recommended

Upload the **`Ai/` folder** to your server and run **one command**:

```bash
cd Ai
cp .env.example .env
# edit API keys
./start.sh
```

See [`../README.md`](../README.md) for the full deploy guide.

**EEG only:**

```bash
cd Ai
docker compose up -d --build eeg-inference
```

Requires **Docker** + **Docker Compose** on the VPS (≥ **2 GB RAM** for EEG alone; ≥ **6 GB** for both services).

### 1. Copy project folder to VPS

Upload model files:

- `Ai/EEG/eeg_web_output/eeg_model.pt`
- `Ai/EEG/eeg_web_output/scaler.pkl`

### 2. Configure env

Use the **project root** `.env` (same file PHP reads):

```bash
cp .env.example .env
# Set EEG_SERVICE_API_KEY (same value as Ai/.env on VPS)
```

### 3. Build and run

```bash
docker compose up -d --build eeg-inference
docker compose logs -f eeg-inference
```

Health check: `curl http://YOUR_VPS_IP:8000/health` → `"model_ready": true`

### 4. Point PHP to the VPS

On Hostinger `.env`:

```env
EEG_SERVICE_URL=http://YOUR_VPS_IP:8000
EEG_SERVICE_API_KEY=same-secret-as-docker-.env
EEG_SERVICE_TIMEOUT=120
```

Use `https://` if you put **nginx** + SSL in front of port 8000.

### Optional: mount models without rebuilding

The root `docker-compose.yml` mounts `./Ai/EEG/eeg_web_output` read-only from the host.

### Commands

| Command | Purpose |
|---------|---------|
| `docker compose up -d --build` | From `Ai/` — start EEG + Chest |
| `docker compose up -d --build eeg-inference` | EEG only |
| `docker compose logs -f eeg-inference` | Follow EEG logs |
| `docker compose restart eeg-inference` | Restart after env change |
| `docker compose down` | Stop containers |

---

## Deploy (Render)

1. Connect repo; set **Root Directory** to `Ai/EEG` (or use repo-root `render.yaml`).
2. Set env `EEG_SERVICE_API_KEY` to a long random secret.
3. Ensure `eeg_web_output/` is deployed (not gitignored on your deploy branch).
4. Use **≥ 2 GB RAM** plan (PyTorch + MNE + ICA).
5. Copy service URL to PHP `.env` as `EEG_SERVICE_URL`.

## PHP integration

Medical Scans PHP calls `{EEG_SERVICE_URL}/predict` with the same API key header. Mobile/web use `POST /api/v1/scans.php?action=upload_eeg` only.
