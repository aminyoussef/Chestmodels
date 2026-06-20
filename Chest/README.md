# ChestScan API — Setup & Deployment Guide

## What's included

| File | Purpose |
|------|---------|
| `app.py` | Flask REST API server |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container for cloud/server deployment |
| `index.html` | Full-featured web & mobile test client |

---

## 1 · Quick local start

### 1.1 Install dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Place your trained model
Create a `models/` folder next to `app.py` and copy your `.pth` checkpoint there:
```
api/
  models/
    best_model.pth          ← from CFG['save_path'] in the notebook
```

The API **automatically picks the checkpoint with the highest `val_auc`** — no config needed. Drop multiple `.pth` files and it will always use the best one.

### 1.3 Start the server
```bash
python app.py
# Server starts on http://0.0.0.0:5000
```

### 1.4 Open the test client
Open `index.html` in any browser. Set the API URL to `http://localhost:5000`, click **Check API**, then upload a scan.

---

## 2 · API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status + loaded model info |
| GET | `/classes` | List of 14 disease class names |
| POST | `/analyze` | Upload a scan, get predictions |

### POST /analyze
Send a `multipart/form-data` request with field name `image`:

```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@/path/to/xray.jpg"
```

#### Response
```json
{
  "image_hash": "sha256-hex",
  "threshold": 0.5,
  "no_finding": false,
  "detected": [
    { "condition": "Effusion", "probability": 0.832, "detected": true }
  ],
  "predictions": [
    { "condition": "Effusion",     "probability": 0.832, "detected": true  },
    { "condition": "Atelectasis",  "probability": 0.411, "detected": false },
    ...
  ],
  "model": "efficientnet_b0",
  "checkpoint": "best_model.pth"
}
```

---

## 3 · Deterministic / consistent results

The same scan **always returns the same result**.

Two mechanisms ensure this:

1. **No randomness at inference**: `model.eval()` + `torch.no_grad()` disable dropout and gradient tracking.
2. **SHA-256 hash cache**: Every result is stored in memory keyed by the SHA-256 of the raw image bytes. Re-uploading the same file returns the cached result instantly without re-running the model. The client displays a ⚡ "Cached result" notice when this happens.

---

## 4 · Docker deployment

Upload the **`Ai/` folder** to your server and run:

```bash
cd Ai
mkdir -p Chest/models
# Copy best_model.pth into Chest/models/
./start.sh
```

See [`../README.md`](../README.md) for the full deploy guide.

**Chest only:**

```bash
cd Ai
docker compose up -d --build chest-inference
```

When `/health` returns `"status": "ready"`, the model is loaded.

### Standalone container (without compose)

```bash
# Build
docker build -t chestscan-api .

# Run (mount your models folder)
docker run -d \
  -p 5000:5000 \
  -v /absolute/path/to/models:/app/models \
  --name chestscan \
  chestscan-api
```

For GPU support add `--gpus all` and use a CUDA base image.

---

## 5 · Medical Scans PHP integration

The main PHP app calls the Chest service through `ChestInferenceClient.php` when users upload X-rays via `POST /api/v1/imaging.php?action=upload`.

Add to the project root `.env`:

```env
CHEST_SERVICE_URL=http://YOUR_VPS_IP:5000
CHEST_SERVICE_API_KEY=same-secret-as-docker-.env
CHEST_SERVICE_TIMEOUT=120
```

| Endpoint (PHP) | Description |
|----------------|-------------|
| `POST /api/v1/imaging.php?action=upload` | Multipart field `image` → AI analysis + DB record |
| `GET /api/v1/imaging.php?action=chest_service_health` | Proxy health check for admins/clients |
| `GET /api/v1/imaging.php?id={id}` | Fetch saved imaging report |

PHP sends `POST {CHEST_SERVICE_URL}/analyze` with header `X-Chest-Service-Key` when the key is configured. Mobile/web clients should use the PHP API only, not the Flask service directly.

In development (`APP_DEBUG=true`), if `CHEST_SERVICE_URL` is empty, PHP returns a mock report instead of failing.

---

## 6 · Mobile usage

The API is fully CORS-enabled. Any mobile web app or React Native / Flutter app can call:

```
POST http://your-server:5000/analyze
Content-Type: multipart/form-data; boundary=...
Body: image=<file bytes>
```

Serve `index.html` from any static host (Nginx, S3, Netlify) — it works on phones out of the box.

---

## 7 · Supported model architectures

The API auto-detects the architecture from the checkpoint's `cfg.model_name` key:

| Key | Architecture |
|-----|-------------|
| `efficientnet_b0` | EfficientNet-B0 (timm) |
| `efficientnet_b3` | EfficientNet-B3 (timm) |
| `resnet50` | ResNet-50 (torchvision) |
| `densenet121` | DenseNet-121 (torchvision) |

---

## 8 · Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models` | Folder scanned for `.pth` checkpoints |
| `THRESHOLD` | `0.5` | Probability threshold for "detected" flag |
| `CHEST_SERVICE_API_KEY` | *(empty)* | If set, `POST /analyze` requires header `X-Chest-Service-Key` |
