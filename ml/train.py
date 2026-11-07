"""
Training pipeline — reproducible, explainable.

Steps:
  1. Load IRMAS (or synthesize from centroids if not present)
  2. Extract 26-D features via librosa (or use synthetic vectors)
  3. Fit StandardScaler + KNN (K=3, cosine, distance-weighted)
  4. 5-Fold CV report
  5. Save artifacts: scaler.joblib, knn_model.joblib, label_encoder.joblib, centroids.json, metadata.json
  6. Export frontend centroids.ts parity file

Usage:
  python ml/train.py                      # uses synthetic if no IRMAS
  IRMAS_DATASET_DIR=./data/IRMAS python ml/train.py
  python ml/train.py --k 5 --metric cosine
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

# allow imports from backend/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from config import ARTIFACTS, MODELS_DIR, CLASSES, K, METRIC, WEIGHTS
from model import TimbreModel
from dataset import load_or_synth
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

def main():
    parser = argparse.ArgumentParser(description="Train KNN instrument recognizer")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--metric", type=str, default=METRIC, choices=["cosine","euclidean","manhattan","minkowski"])
    parser.add_argument("--weights", type=str, default=WEIGHTS, choices=["uniform","distance"])
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    items = load_or_synth()
    synthetic = isinstance(items, tuple)
    if synthetic:
        X, y = items
        print(f"[train] using synthetic corpus: X={X.shape}, classes={len(set(y))}")
    else:
        # Real files → extract
        from features import extract_batch
        paths = [p for p,_ in items]
        labels = [l for _,l in items]
        print(f"[train] extracting features for {len(paths)} files… (this may take a few minutes)")
        X, _ = extract_batch(paths)
        y = np.array(labels)
        print(f"[train] X.shape={X.shape}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    accs=[]
    y_true_all=[]; y_pred_all=[]
    for tr, te in skf.split(X, y):
        m = TimbreModel(k=args.k, metric=args.metric, weights=args.weights)
        m.fit(X[tr], y[tr])
        preds=[m.predict_with_proba(x)["instrument"] for x in X[te]]
        accs.append(accuracy_score(y[te], preds))
        y_true_all.extend(y[te]); y_pred_all.extend(preds)
    mean_acc=np.mean(accs); std=np.std(accs)
    print(f"\n[CV] accuracy = {mean_acc:.4f} ±{std:.4f}  folds={accs}")
    print(classification_report(y_true_all, y_pred_all, digits=3))
    print(confusion_matrix(y_true_all, y_pred_all))

    # Final fit on full data
    final = TimbreModel(k=args.k, metric=args.metric, weights=args.weights)
    final.fit(X, y)
    final.cv_metrics = {"accuracy": float(mean_acc), "std": float(std), "folds": accs}

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final.save(MODELS_DIR)

    # Compute per-class centroids in SCALED space (for frontend radar)
    from sklearn.preprocessing import StandardScaler as _SS
    # final.scaler already fitted
    centroids_scaled={}
    centroids_raw={}
    for cls in CLASSES:
        mask = y==cls
        if not mask.any(): continue
        raw_mean = X[mask].mean(axis=0)
        scaled_mean = final.scaler.transform(raw_mean.reshape(1,-1))[0]
        centroids_scaled[cls]= scaled_mean.tolist()
        centroids_raw[cls]= raw_mean.tolist()
    # Also store in frontend-compatible format (scaled, matching lib/dataset.ts)
    with open(ARTIFACTS["centroids"], "w") as f:
        json.dump(centroids_scaled, f, indent=2)
    with open(MODELS_DIR / "centroids_raw.json","w") as f:
        json.dump(centroids_raw, f, indent=2)
    print(f"[train] saved centroids.json ({len(centroids_scaled)} classes)")

    # Export metadata
    meta = {
        "k": args.k, "metric": args.metric, "weights": args.weights,
        "accuracy": float(mean_acc), "std": float(std),
        "classes": CLASSES, "samples": int(len(y)), "feature_dim": 26,
        "scaler": "StandardScaler", "dataset": "IRMAS (or synthetic mirror)"
    }
    with open(MODELS_DIR / "train_meta.json","w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("\n✓ Train complete. Artifacts in ./models/")
    print("  → Start API: python backend/app.py")
    print("  → Or use frontend browser inference (no Python needed)")

if __name__ == "__main__":
    main()
