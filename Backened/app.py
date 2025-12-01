from flask import Flask, jsonify, request
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import uuid
import os

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "model/knn_instrument_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    model = None
    print("Model not found.")

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=22050, duration=5)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        feat_dict = {
            "mfcc_mean": float(np.mean(mfcc)),
            "mfcc_std": float(np.std(mfcc)),
            "chroma_mean": float(np.mean(chroma)),
            "chroma_std": float(np.std(chroma)),
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff))
        }

        ordered = np.array(list(feat_dict.values()))
        return feat_dict, ordered
    except:
        return None, None

@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_id = str(uuid.uuid4()) + ".wav"
    path = os.path.join(UPLOAD_DIR, file_id)
    f.save(path)

    feat_dict, arr = extract_features(path)
    if arr is None:
        return jsonify({"error": "Feature extraction failed"}), 400

    arr = arr.reshape(1, -1)

    pred = model.predict(arr)[0]

    try:
        prob = model.predict_proba(arr)[0]
        classes = model.classes_
        similarity = {classes[i]: float(prob[i]) for i in range(len(classes))}
        acc = max(prob) * 100
    except:
        similarity = {pred: 1.0}
        acc = 100.0

    return jsonify({
        "predicted_instrument": pred,
        "accuracy_percent": acc,
        "similarity_scores": similarity,
        "features": feat_dict
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
