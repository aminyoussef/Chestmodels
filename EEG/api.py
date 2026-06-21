"""
EEG Dementia Classifier — FastAPI Backend
==========================================
Make sure you run this with the SAME Python/env where packages are installed:

    E:\plant-disease-project\tfenv\Scripts\python.exe app.py

Or activate that environment first:
    E:\plant-disease-project\tfenv\Scripts\activate
    python app.py

Then open: http://localhost:8000/docs
"""

import os, json, pickle, tempfile, traceback, pathlib, sys

try:
    from dotenv import load_dotenv
    _env_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_env_dir, '.env'))
    load_dotenv(os.path.join(_env_dir, '..', '..', '.env'))
except ImportError:
    pass

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import mne; mne.set_log_level('ERROR')
from mne.preprocessing import ICA
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

print(f"Python  : {sys.executable}")
print(f"PyTorch : {torch.__version__}")

# ── Config ────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    'EEG_MODEL_DIR',
    os.path.join(_BASE_DIR, 'eeg_web_output'),
)
EEG_SERVICE_API_KEY = os.environ.get('EEG_SERVICE_API_KEY', '').strip()
TARGET_SFREQ    = 128
EPOCH_LENGTH    = 4.0
EPOCH_OVERLAP   = 0.5
CLASS_NAMES     = ['AD', 'FTD', 'CN']
SUPPORTED_EXTS  = {'.set', '.edf', '.bdf', '.fif', '.cnt'}
EMOTIV_CHANNELS = ['AF3','AF4','F3','F4','FC5','FC6',
                    'F7','F8','T7','T8','P7','P8','O1','O2']
CHANNEL_RENAME  = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
INTERPOLATION_RULES = {
    'AF3':[('Fp1',.5),('F3',.3),('F7',.2)],
    'AF4':[('Fp2',.5),('F4',.3),('F8',.2)],
    'FC5':[('F7',.3),('C3',.4),('T3',.3)],
    'FC6':[('F8',.3),('C4',.4),('T4',.3)],
}
BANDS      = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}
HEMI_PAIRS = [('F3','F4'),('T7','T8'),('P7','P8'),('O1','O2')]

# ── Model definition (must match saved eeg_model.pt exactly) ──
class AttnPool(nn.Module):
    """Learned attention pooling — checkpoint keys: attn.attn.weight/bias"""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class EEG_CNN_LSTM(nn.Module):
    def __init__(self, n_feats, n_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(256),
        )
        self.lstm = nn.LSTM(
            256, 128, num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True,
        )
        self.attn = AttnPool(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.head(x)

# ── Load artefacts ────────────────────────────────────────────
print(f"[startup] Loading model from: {os.path.abspath(MODEL_DIR)}")
clf, scaler, n_feats = None, None, None
_model_load_error = None
try:
    ckpt    = torch.load(
        os.path.join(MODEL_DIR, 'eeg_model.pt'),
        map_location='cpu',
        weights_only=False,
    )
    n_feats = ckpt['n_feats']
    clf     = EEG_CNN_LSTM(n_feats, ckpt['n_classes'])
    clf.load_state_dict(ckpt['model_state'])
    clf.eval()
    with open(os.path.join(MODEL_DIR,'scaler.pkl'),'rb') as f:
        scaler = pickle.load(f)
    print(f"[startup] ✓ Model ready.  n_feats={n_feats}")
except FileNotFoundError:
    _model_load_error = f"No trained model found in '{MODEL_DIR}'."
    print(f"[startup] ⚠  {_model_load_error}")
    print( "[startup]    Run the notebook (Cells 1-10) to train and save the model first.")
except Exception as e:
    _model_load_error = str(e)
    print(f"[startup] ⚠  Error loading model: {e}")
    traceback.print_exc()

# ── Preprocessing ─────────────────────────────────────────────
def spectral_entropy(psd):
    p = psd / (psd.sum() + 1e-10)
    return -np.sum(p * np.log2(p + 1e-10))

def hjorth(x):
    d1 = np.diff(x); d2 = np.diff(d1)
    mob = np.std(d1) / (np.std(x) + 1e-10)
    com = (np.std(d2) / (np.std(d1) + 1e-10)) / (mob + 1e-10)
    return mob, com

def extract_features(epochs, sfreq=TARGET_SFREQ):
    rows = []
    for ep in epochs:
        row = []; bp_all = []
        for ci in range(ep.shape[0]):
            x = ep[ci]
            f, psd = signal.welch(x, fs=sfreq, nperseg=min(256, len(x)))
            bp = {}
            for name,(lo,hi) in BANDS.items():
                bp[name] = np.mean(psd[(f>=lo)&(f<=hi)]) + 1e-10
                row.append(np.log10(bp[name]))
            bp_all.append(bp)
            row.append(np.log10(bp['theta'] / bp['alpha']))
            row.append(np.log10(bp['delta'] / bp['alpha']))
            row.append(np.log10((bp['delta']+bp['theta']) / (bp['alpha']+bp['beta'])))
            row.append(spectral_entropy(psd))
            row += [np.mean(x), np.std(x), float(skew(x)), float(kurtosis(x))]
            mob, com = hjorth(x); row += [mob, com]
        for l,r in HEMI_PAIRS:
            if l in EMOTIV_CHANNELS and r in EMOTIV_CHANNELS:
                li = EMOTIV_CHANNELS.index(l)
                ri = EMOTIV_CHANNELS.index(r)
                for name in BANDS:
                    row.append(np.log10(bp_all[li][name] / (bp_all[ri][name]+1e-10)))
        rows.append(row)
    return np.array(rows, dtype=np.float32)

def _read_magic(path, n=16):
    with open(path, 'rb') as f:
        return f.read(n)


def _friendly_load_error(path, exc):
    """Turn low-level scipy/MNE errors into actionable upload hints."""
    msg = str(exc)
    low = msg.lower()
    name = pathlib.Path(path).name
    stem = pathlib.Path(path).stem

    if 'unknown mat file type' in low:
        return (
            f"'{name}' is not a valid EEGLAB .set (MATLAB) file. "
            'Use .edf if possible, or re-export from EEGLAB with "File → Save current dataset" '
            'using MATLAB v6/v7 format (not v7.3). If the recording is EDF/BDF, upload with the correct extension.'
        )
    if 'hdf' in low or 'v7.3' in low or 'please use hdf reader' in low:
        return (
            f"'{name}' uses MATLAB v7.3 (.set). Re-save in EEGLAB as MATLAB v6/v7, or export as .edf."
        )
    if 'fdt' in low and ('not find' in low or 'could not' in low or 'no such file' in low):
        return (
            f"Missing companion data file '{stem}.fdt' for '{name}'. "
            f'Upload both {stem}.set and {stem}.fdt together, or export a single-file .set/.edf from EEGLAB.'
        )
    if 'buffer is too small' in low or 'uint16' in low:
        return f"Could not decode EEGLAB metadata in '{name}'. Re-export as .edf or re-save the .set from EEGLAB."
    return msg


def load_raw_eeg(path):
    """Load supported EEG formats with EEGLAB/.set fallbacks."""
    ext = pathlib.Path(path).suffix.lower()
    p = pathlib.Path(path)

    if ext == '.set':
        magic = _read_magic(path, 8)
        if magic.startswith(b'\x89HDF'):
            pass  # MATLAB v7.3 — MNE uses h5py when available
        elif not magic.startswith(b'MATLAB'):
            # Misnamed clinical file (common: EDF uploaded as .set)
            if len(magic) >= 8 and magic[0:1] in (b'0', b'1') and magic[1:8] == b'       ':
                return mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
            if magic.startswith(b'BIOSEMI'):
                return mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
            raise ValueError(_friendly_load_error(path, Exception('Unknown mat file type')))

        last_err = None
        for codec in ('latin-1', 'utf-8', 'ascii'):
            try:
                return mne.io.read_raw_eeglab(
                    path, preload=True, verbose='ERROR', uint16_codec=codec,
                )
            except Exception as e:
                last_err = e
        raise ValueError(_friendly_load_error(path, last_err or Exception('read failed')))

    loaders = {
        '.edf': mne.io.read_raw_edf,
        '.bdf': mne.io.read_raw_bdf,
        '.fif': mne.io.read_raw_fif,
        '.cnt': mne.io.read_raw_cnt,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported format '{ext}'. Accepted: {sorted(SUPPORTED_EXTS)}")
    try:
        return loaders[ext](path, preload=True, verbose='ERROR')
    except Exception as e:
        raise ValueError(_friendly_load_error(path, e)) from e


def preprocess(path):
    raw = load_raw_eeg(path)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 45., method='fir', fir_design='firwin', verbose='ERROR')
    data, chs = raw.get_data(), list(raw.ch_names)
    new_d, new_n = [], []
    for ch, wts in INTERPOLATION_RULES.items():
        if ch in chs: continue
        vec = np.zeros(data.shape[1])
        for src, w in wts:
            if src in chs: vec += w * data[chs.index(src)]
        new_d.append(vec); new_n.append(ch)
    if new_d:
        raw.add_channels([mne.io.RawArray(
            np.stack(new_d),
            mne.create_info(new_n, raw.info['sfreq'], 'eeg'),
            verbose='ERROR')], force_update_info=True)
    raw.rename_channels({k:v for k,v in CHANNEL_RENAME.items() if k in raw.ch_names})
    keep = [c for c in EMOTIV_CHANNELS if c in raw.ch_names]
    raw.pick_channels(keep, ordered=True)
    try:
        ica = ICA(n_components=min(15,len(keep)-1), random_state=42, verbose='ERROR')
        ica.fit(raw); bads,_ = ica.find_bads_eog(raw)
        ica.exclude = bads; ica.apply(raw)
    except Exception: pass
    if raw.info['sfreq'] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad='auto', verbose='ERROR')
    d   = raw.get_data()
    win = int(EPOCH_LENGTH * TARGET_SFREQ)
    stp = int(win * (1 - EPOCH_OVERLAP))
    eps = [d[:, s:s+win] for s in range(0, d.shape[1]-win+1, stp)]
    return np.stack(eps) if eps else np.empty((0, d.shape[0], win))

def deterministic_predict(model, X):
    """
    Fully deterministic: model.eval() disables dropout,
    no random noise, torch.no_grad() for speed.
    Same file always returns same result.
    """
    model.eval()
    Xt = torch.tensor(X)
    with torch.no_grad():
        proba = F.softmax(model(Xt), dim=1).numpy()
    return proba

def run_inference(path):
    epochs = preprocess(path)
    if len(epochs) == 0:
        raise ValueError('No valid epochs could be extracted from the file.')
    X  = extract_features(epochs)
    Xs = scaler.transform(X).astype(np.float32).reshape(-1, 1, n_feats)
    # Average probabilities across all epochs from the file (deterministic)
    p   = deterministic_predict(clf, Xs).mean(axis=0)
    idx = int(p.argmax())
    return {
        'prediction':    CLASS_NAMES[idx],
        'confidence':    round(float(p[idx]), 4),
        'probabilities': {c: round(float(v), 4) for c,v in zip(CLASS_NAMES, p)},
        'n_epochs':      len(epochs),
    }

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title='EEG Dementia Classifier',
    version='2.2.0',
    description='Upload .edf / .set / .bdf / .fif / .cnt → AD / FTD / CN')

app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class PredictionResponse(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: Dict[str, float]
    n_epochs:      int
    filename:      str

def verify_service_key(x_eeg_service_key: str = Header(default=None, alias='X-EEG-Service-Key')):
    if not EEG_SERVICE_API_KEY:
        return
    if not x_eeg_service_key or x_eeg_service_key != EEG_SERVICE_API_KEY:
        raise HTTPException(401, 'Invalid or missing X-EEG-Service-Key')

@app.get('/')
@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_ready': clf is not None and scaler is not None,
        'model_dir': MODEL_DIR,
        'load_error': _model_load_error,
        'service': 'eeg-dementia-classifier',
        'classes': CLASS_NAMES,
        'supported_exts': sorted(SUPPORTED_EXTS),
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    x_eeg_service_key: str = Header(default=None, alias='X-EEG-Service-Key'),
):
    verify_service_key(x_eeg_service_key)
    if clf is None or scaler is None:
        raise HTTPException(503,
            'Model not loaded. Run the notebook (Cells 1-10) first to train.')
    ext = pathlib.Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400,
            f"Unsupported format '{ext}'. Accepted: {sorted(SUPPORTED_EXTS)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = run_inference(tmp_path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, _friendly_load_error(tmp_path, e))
    finally:
        os.unlink(tmp_path)
    return PredictionResponse(filename=file.filename, **result)

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', '8000'))
    uvicorn.run('api:app', host='0.0.0.0', port=port, reload=False)
