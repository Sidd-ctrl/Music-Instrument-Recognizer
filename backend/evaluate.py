"""
Evaluation — 5-Fold Stratified CV, confusion matrix, per-class metrics.
Run: python -m backend.evaluate
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import CLASSES, K, METRIC, WEIGHTS
from model import TimbreModel
from dataset import load_or_synth

def evaluate():
    items = load_or_synth()
    # If synthetic, items is (X,y)
    if isinstance(items, tuple):
        X, y = items
        from collections import Counter
        print(f"[eval] synthetic set: {Counter(y)}  total={len(y)}")
    else:
        # Real IRMAS: need to extract features
        from features import extract_batch
        paths_labels = items
        paths = [p for p,_ in paths_labels]
        labels = [l for _,l in paths_labels]
        print(f"[eval] extracting {len(paths)} files…")
        from features import extract_batch as eb
        X, _ = eb(paths)
        y = np.array(labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs=[]
    reports=[]
    cms = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    y_true_all=[]; y_pred_all=[]

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = TimbreModel(k=K, metric=METRIC, weights=WEIGHTS)
        model.fit(X[tr], y[tr])
        preds=[]
        for x in X[te]:
            r = model.predict_with_proba(x)
            preds.append(r["instrument"])
        preds=np.array(preds)
        y_true_all.extend(y[te]); y_pred_all.extend(preds)
        acc = accuracy_score(y[te], preds)
        accs.append(acc)
        print(f" fold {fold}: acc={acc:.4f}")

    print("\n=== 5-Fold CV ===")
    print(f" Accuracy: {np.mean(accs):.4f} ±{np.std(accs):.4f}")
    print(f" Accs per fold: {accs}")

    print("\nClassification report:\n", classification_report(y_true_all, y_pred_all, digits=3))

    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
    print("\nConfusion matrix rows=true, cols=pred")
    print(cm)

    # plot
    try:
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f"Confusion Matrix — KNN K={K} cosine (acc {np.mean(accs):.2%})")
        plt.tight_layout()
        plt.savefig("../models/confusion_matrix.png", dpi=160)
        print("[eval] saved models/confusion_matrix.png")
    except Exception as e:
        print("[eval] plot skipped:", e)

if __name__ == "__main__":
    evaluate()
