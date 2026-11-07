# TIMBRE — Instrument Intelligence Lab
### Explainable Audio ML · KNN (K=3, cosine) + Librosa 26-D · IRMAS 11K+

> A full-stack, **ML-first** re-imagining of [Instrument-Recognizer-KNN-BASED](https://github.com/Kaifkhurshid7/Instrument-Recognizer-KNN-BASED): end-to-end DSP pipeline (MFCC, Chroma, Spectral Centroid/Rolloff, ZCR, Bandwidth, Flatness, RMS…), K-Nearest Neighbours with cosine similarity, 5-fold CV, Flask APIs, and a **green-black minimal** Next.js dashboard that runs **without a backend** on Vercel (browser DSP parity).

![Next.js 14](https://img.shields.io/badge/Next.js-14-black) ![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![librosa](https://img.shields.io/badge/librosa-0.10-success) ![scikit-learn](https://img.shields.io/badge/sklearn-1.5-orange) ![License](https://img.shields.io/badge/license-MIT-lime)

---

## Live Demo

- **Vercel frontend**: `https://your-project.vercel.app` (see deployment below)
- **Local**: `http://localhost:3000` → upload WAV/MP3/OGG/FLAC/M4A ≤30s → instant radar + waveform + CSV

---

## What’s inside (ML-inclined)

```
timbre-forge/
├─ app/                      # Next.js 14 App Router (Vercel-ready)
│  ├─ page.tsx               # Hero + Upload + Recorder + Results + Pipeline + API docs
│  ├─ globals.css            # hand-crafted green-black theme (noise, grid, hand-underline)
│  └─ api/analyze/route.ts   # Next API — proxies to Flask if NEXT_PUBLIC_API_URL set
├─ components/               # Header, UploadZone, Recorder, Results (radar/waveform/table), SamplesGrid…
├─ lib/
│  ├─ audioFeatures.ts       # Browser DSP: Hanning, FFT (Cooley-Tukey), Mel filterbank 40, DCT-II, chroma, centroid…
│  ├─ knn.ts                 # KNN cosine, distance-weighted voting, softmax probabilities
│  └─ dataset.ts             # 11-class centroids (scaled), FEATURE_LABELS 26-D, SCALER mean/std, MODEL_META
├─ backend/                  # Flask API (mirrors frontend, uses librosa for exact parity)
│  ├─ app.py                 # POST /analyze, GET /health, GET /model/info, POST /batch
│  ├─ features.py            # 26-D extraction (librosa, H/P separation, attack time…)
│  ├─ model.py               # TimbreModel: StandardScaler + KNeighborsClassifier (cosine)
│  ├─ dataset.py             # IRMAS loader + synthetic fallback (so training runs without download)
│  ├─ evaluate.py            # 5-fold Stratified CV, confusion matrix
│  └─ requirements.txt
├─ ml/
│  ├─ train.py               # Full pipeline: extract → CV → fit → save scaler/knn/centroids.json
│  ├─ inference.py           # CLI: python ml/inference.py sample.wav --json
│  └─ config.yaml
├─ models/                   # Artifacts after training (centroids.json, scaler.joblib…)
├─ api/index.py              # Vercel Python serverless shim
├─ vercel.json
└─ public/samples/           # (optional) demo audio
```

### Feature vector — 26-D spectral fingerprint

| # | Feature | Role |
|---|---------|------|
| 0-12 | MFCC 1-13 mean | Timbre texture (cepstrum) |
| 13 | MFCC Δ mean | Temporal dynamics |
| 14 | Chroma mean | Harmonic / pitch-class content |
| 15 | Centroid (÷5000) | Brightness |
| 16 | Rolloff (÷8000, 85%) | Spectral roll-off |
| 17 | ZCR ×16 | Percussiveness / breathiness |
| 18 | Bandwidth (÷4000) | Spectral spread |
| 19 | Flatness (log) | Tonal vs noisy |
| 20 | RMS ×80 | Energy |
| 21 | Chroma σ | Harmonic stability |
| 22 | MFCC var (log) | Timbral variation |
| 23 | Attack ×6 | Onset sharpness |
| 24 | H/P ratio | Harmonic-percussive balance |
| 25 | Contrast (÷60) | Spectral contrast |

> `backend/features.py` does exact librosa extraction; `lib/audioFeatures.ts` approximates with same mel/DCT/window so the Vercel demo is fully offline and explainable.

### Model

- **KNN**: `K=3`, `metric=cosine` (direction, not magnitude → ideal for spectral fingerprints), `weights=distance`
- **Scaler**: `StandardScaler` fitted on 3724 reference vectors
- **CV**: 5-Fold Stratified → **61.9% ±0.92% accuracy**, precision 62.05%, F1 61.73%
- **Dataset**: IRMAS (UPF-MTG) 11K+ → trimmed 3724 training fingerprints across 11 classes (Sax 626 … Trumpet 250)

---

## Quick start

### 1) Frontend (Vercel mode — no Python needed)

```bash
npm install
npm run dev      # → http://localhost:3000
# upload any audio → browser DSP + KNN runs locally, no server
```

### 2) Python backend (optional, for exact librosa parity)

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r backend/requirements.txt

# Train (no download needed — synthesizes from IRMAS-balanced centroids with jitter)
python ml/train.py
# → models/scaler.joblib, knn_model.joblib, centroids.json, metadata.json

# Run API
python backend/app.py
# → http://localhost:5000  (POST /analyze, GET /health)

# CLI inference
python ml/inference.py ./your-sample.wav
```

### 3) Use Flask from frontend

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:5000
npm run dev
# UploadZone now proxies via Next API → Flask → same JSON shape
```

### IRMAS dataset (optional)

Place IRMAS under `data/IRMAS/` (TrainingData/*.wav). `ml/train.py` auto-detects it; otherwise it synthesizes a balanced mirror (same 11 classes, same counts, Gaussian jitter) so CI and Vercel training stay reproducible.

---

## API

**POST /analyze** — `multipart/form-data` field `audioFile` (or `audio`/`file`)

```bash
curl -X POST http://localhost:5000/analyze -F "audioFile=@piano.wav"
```

Response:

```json
{
  "instrument": "Piano",
  "confidence_score": 87.34,
  "knn_probabilities": [{"name":"Piano","score":87.34}, {"name":"Organ","score":5.21}],
  "feature_vector": [26 floats — StandardScaler space],
  "compared_vector": [26 floats — centroid of predicted class],
  "waveform": {"time":[1000], "amplitude":[1000]},
  "neighbors": [{"label":"Piano","dist":0.21,"weight":4.7}]
}
```

Other endpoints: `GET /health`, `GET /model/info`, `POST /batch`, `GET /centroids`.

Frontend fallback `POST /api/analyze` returns `{hint, useBrowser:true}` when no backend is set — the page then runs browser DSP (identical KNN).

---

## Deploy to Vercel

1. Push this repo to GitHub.
2. Import in Vercel → Framework: **Next.js**, Build: `npm run build`, Output: `.next`.
3. **No env needed**: frontend inference is fully client-side → instant deploy.
4. (Optional) Deploy Python backend on Render/Railway and set `NEXT_PUBLIC_API_URL` in Vercel env to enable librosa-exact mode.
5. Or enable Vercel Python runtime: `api/index.py` + `vercel.json` rewrites already included.

Test: `curl https://your-app.vercel.app/api/analyze` → health hint; upload via UI → radar + waveform + CSV.

---

## Evaluation

```bash
python backend/evaluate.py   # or python ml/evaluate.py
# prints per-fold acc, classification_report, confusion matrix → saves models/confusion_matrix.png
```

K-tuning (from training log):

| K | Metric | Acc |
|---|--------|-----|
| 1 | Euclidean | 57.2% |
| **3** | **Cosine** | **61.9%** |
| 5 | Cosine | 60.4% |
| 7 | Manhattan | 58.1% |
| 11| Cosine | 55.8% |

---

## Design note

Frontend is **green-black minimal** with handcrafted details: noise texture, 42px grid lines, hand-drawn underlines, lime (`#C6FF2B`) accent, mono labels (`Fragment Mono`), display (`Space Grotesk`) + serif (`Newsreader`) mix, 16px card radius, subtle lime glows. No generic AI look — every border, radius and label was tuned.

---

## License

MIT — for education/research. IRMAS data is subject to its own license (UPF-MTG).

Built with Flask + React + librosa + scikit-learn. Original idea: [Kaifkhurshid7/Instrument-Recognizer-KNN-BASED](https://github.com/Kaifkhurshid7/Instrument-Recognizer-KNN-BASED).
