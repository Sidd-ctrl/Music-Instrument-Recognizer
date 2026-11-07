"""
Flask API — Vercel-ready + local dev.

Endpoints:
  GET  /health        → liveness + model status
  GET  /model/info    → K, metric, CV metrics, classes
  POST /analyze       → multipart/form-data audioFile → JSON with instrument + explainability
  POST /batch         → batch analyze (zip or multi-file)
  GET  /centroids     → per-class centroids for radar

Audio conversion is handled via pydub + librosa (mp3/ogg/flac/m4a → wav mono 22.05k).
"""
import os, tempfile, traceback, json
from pathlib import Path
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import PORT, DEBUG, CORS_ORIGINS, CLASSES, MAX_DURATION, SAMPLE_RATE
from features import extract_26d
from model import TimbreModel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS if CORS_ORIGINS!="*" else "*"}})

# Lazy-load model (don’t crash on import if not trained)
_model = None
def get_model():
    global _model
    if _model is None:
        try:
            _model = TimbreModel.load()
            print("[api] model loaded")
        except Exception as e:
            print(f"[api] model not ready: {e}")
            _model = None
    return _model

ALLOWED_EXTS = {".wav",".mp3",".ogg",".flac",".m4a",".aac",".wma",".aiff",".webm"}

def allowed_file(fname):
    return Path(fname).suffix.lower() in ALLOWED_EXTS

def convert_to_wav(src_path: str, dst_path: str):
    """Use pydub if needed; fallback to librosa direct load."""
    ext = Path(src_path).suffix.lower()
    if ext == ".wav":
        # just copy/ensure mono via librosa load later
        import shutil; shutil.copy(src_path, dst_path); return
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        # trim to MAX_DURATION
        max_ms = int(MAX_DURATION*1000)
        if len(audio) > max_ms:
            audio = audio[:max_ms]
        audio.export(dst_path, format="wav")
    except Exception as e:
        # fallback: librosa will handle most formats via audioread/ffmpeg
        print(f"[convert] pydub failed ({e}), falling back to librosa passthrough")
        import shutil; shutil.copy(src_path, dst_path)

@app.get("/health")
def health():
    m = get_model()
    return jsonify({"status":"ok","model_ready": m is not None and m.is_fitted})

@app.get("/model/info")
def model_info():
    m = get_model()
    if m is None:
        return jsonify({"error":"model not trained. Run ml/train.py","model_ready":False}), 503
    return jsonify({
        "k": m.k, "metric": m.metric, "weights": m.weights,
        "scaler": "StandardScaler", "classes": CLASSES,
        "samples": 3724, "cv_accuracy": 61.9, "cv_std": 0.92,
        "hop_length": 1024, "sr": SAMPLE_RATE,
        "model_ready": True
    })

@app.get("/centroids")
def centroids():
    m = get_model()
    if m is None:
        return jsonify({"error":"model not ready"}), 503
    # return per-class centroids in scaled space
    from config import ARTIFACTS
    if ARTIFACTS["centroids"].exists():
        with open(ARTIFACTS["centroids"]) as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"hint":"no centroids file, inference still works"})

@app.post("/analyze")
def analyze():
    m = get_model()
    if m is None:
        return jsonify({"error":"Model not initialized. Train first: python ml/train.py"}), 503

    if "audioFile" not in request.files and "audio" not in request.files and "file" not in request.files:
        return jsonify({"error":"No audio file provided. Field must be 'audioFile' (or 'audio'/'file')"}), 400

    # resolve field name
    file = request.files.get("audioFile") or request.files.get("audio") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error":"Empty filename"}), 400

    filename = secure_filename(file.filename) or "upload.bin"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"}), 400

    tmpdir = tempfile.mkdtemp(prefix="timbre_")
    src = os.path.join(tmpdir, filename)
    dst = os.path.join(tmpdir, "converted.wav")
    file.save(src)

    try:
        convert_to_wav(src, dst)
        # librosa load (ensures mono, 22.05k, trim)
        y, sr = librosa.load(dst, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        if len(y) == 0:
            return jsonify({"error":"Decoded audio is empty"}), 400

        raw_vec, meta = extract_26d(y, sr)
        pred = m.predict_with_proba(raw_vec)

        # full response shape matching frontend contract
        return jsonify({
            "instrument": pred["instrument"],
            "confidence_score": round(float(pred["confidence"]), 2),
            "knn_probabilities": [{"name": p["name"], "score": round(p["score"],2)} for p in pred["probabilities"]],
            "feature_vector": pred["feature_vector"],
            "compared_vector": pred["compared_vector"],
            "waveform": meta["waveform"],
            "neighbors": pred["neighbors"],
            "meta": {"filename": filename, "duration": meta["duration"], "sr": meta["sr"], "centroid": meta["centroid"], "rolloff": meta["rolloff"], "zcr": meta["zcr"]},
            "model": {"k": m.k, "metric": m.metric, "scaler": "StandardScaler"}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500
    finally:
        # cleanup
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/batch")
def batch():
    """Accept multiple files: field 'files' (multi) or zip"""
    m = get_model()
    if m is None:
        return jsonify({"error":"model not ready"}), 503
    files = request.files.getlist("files") or request.files.getlist("audioFile")
    if not files:
        return jsonify({"error":"No files provided. Use multipart field 'files' (multiple)"}),400
    results=[]
    for f in files[:20]: # cap
        with tempfile.TemporaryDirectory() as td:
            src=os.path.join(td, secure_filename(f.filename)); dst=os.path.join(td,"c.wav")
            f.save(src)
            try:
                convert_to_wav(src, dst)
                y,sr=librosa.load(dst, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
                raw,_=extract_26d(y,sr)
                pred=m.predict_with_proba(raw)
                results.append({"file": f.filename, "instrument": pred["instrument"], "confidence": round(pred["confidence"],2)})
            except Exception as e:
                results.append({"file": f.filename, "error": str(e)})
    return jsonify({"results": results})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error":"File too large"}), 413

if __name__ == "__main__":
    # local dev
    print(f"[api] starting on http://0.0.0.0:{PORT}  debug={DEBUG}")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
