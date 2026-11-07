"""
KNN model wrapper — StandardScaler + KNeighborsClassifier (cosine, distance-weighted)
Parallels frontend lib/knn.ts: same metric, same K.

Artifacts are saved via joblib for Flask inference.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from config import ARTIFACTS, K, METRIC, WEIGHTS, CLASSES

class TimbreModel:
    def __init__(self, k=K, metric=METRIC, weights=WEIGHTS):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
        self.le = LabelEncoder()
        self.le.fit(CLASSES)
        self.is_fitted = False
        self.cv_metrics = {}

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        """X_raw: (n,26) raw features, y: string labels"""
        X_scaled = self.scaler.fit_transform(X_raw)
        y_enc = self.le.transform(y)
        self.knn.fit(X_scaled, y_enc)
        self.is_fitted = True
        return self

    def predict_with_proba(self, x_raw: np.ndarray):
        """
        x_raw: (26,) or (1,26)
        Returns: {
          instrument, confidence, probabilities (list), neighbors, scaler vector, comparedVector (centroid)
        }
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Run train.py first or load artifacts.")
        if x_raw.ndim == 1:
            x_raw = x_raw.reshape(1, -1)
        x_scaled = self.scaler.transform(x_raw)

        # k neighbors
        dists, idxs = self.knn.kneighbors(x_scaled, n_neighbors=self.k)
        # predict
        probs = self.knn.predict_proba(x_scaled)[0]  # (n_classes,)
        pred_enc = int(np.argmax(probs))
        pred_label = self.le.inverse_transform([pred_enc])[0]
        confidence = float(probs[pred_enc]*100)

        # all class probs sorted
        order = np.argsort(probs)[::-1]
        probabilities = [
            {"name": self.le.inverse_transform([int(i)])[0], "score": float(probs[int(i)]*100), "id": str(int(i))}
            for i in order
        ]

        # neighbors info
        train_labels_enc = self.knn._y if hasattr(self.knn, "_y") else self.knn._fit_X  # fallback
        neighbors = []
        for d, idx in zip(dists[0], idxs[0]):
            # try to get label of neighbor from training set if available
            try:
                # sklearn stores _y
                lab_enc = int(self.knn._y[idx]) if hasattr(self.knn, "_y") else pred_enc
                lab = self.le.inverse_transform([lab_enc])[0]
            except Exception:
                lab = f"#{idx}"
            neighbors.append({"label": lab, "dist": float(d), "weight": float(1/(d+1e-6))})

        # centroids (mean scaled vector per class) — computed on the fly from training data if we have it
        # If not, fallback to nearest-neighbor centroid approximation (stored centroids.json)
        compared = None
        try:
            import json as _json
            from config import ARTIFACTS as _ART
            if _ART["centroids"].exists():
                with open(_ART["centroids"]) as f:
                    cents = _json.load(f)
                    # centroids are stored as scaled vectors per label
                    compared = np.array(cents.get(pred_label, cents.get(str(pred_enc), [0]*26)))
                # if stored centroids are raw, scale them
                if compared is not None and len(compared)==26:
                    # heuristic: if values look raw (|mean|>5) scale
                    if np.abs(compared).mean() > 2:
                        compared = self.scaler.transform(compared.reshape(1,-1))[0]
            if compared is None:
                compared = x_scaled[0] # fallback
        except Exception:
            compared = x_scaled[0]

        return {
            "instrument": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "neighbors": neighbors,
            "feature_vector": x_scaled[0].tolist(),  # scaled space, matches frontend
            "compared_vector": compared.tolist() if hasattr(compared, "tolist") else list(compared),
            "raw_scaled": x_scaled[0].tolist(),
        }

    def save(self, models_dir: Path = None):
        from config import MODELS_DIR as _MD
        md = models_dir or _MD
        md.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, ARTIFACTS["scaler"])
        joblib.dump(self.knn, ARTIFACTS["knn"])
        joblib.dump(self.le, ARTIFACTS["label_encoder"])
        with open(ARTIFACTS["metadata"], "w") as f:
            json.dump({
                "k": self.k, "metric": self.metric, "weights": self.weights,
                "classes": CLASSES, "cv_metrics": self.cv_metrics
            }, f, indent=2)
        print(f"[model] artifacts saved to {md}")

    @classmethod
    def load(cls):
        m = cls()
        if not ARTIFACTS["scaler"].exists():
            raise FileNotFoundError("No trained artifacts found. Run `python ml/train.py` first.")
        m.scaler = joblib.load(ARTIFACTS["scaler"])
        m.knn = joblib.load(ARTIFACTS["knn"])
        m.le = joblib.load(ARTIFACTS["label_encoder"])
        m.is_fitted = True
        if ARTIFACTS["metadata"].exists():
            with open(ARTIFACTS["metadata"]) as f:
                m.cv_metrics = json.load(f).get("cv_metrics", {})
        return m
