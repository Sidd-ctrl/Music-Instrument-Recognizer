# models/ — training artifacts

After `python ml/train.py`:

- `scaler.joblib` — StandardScaler fitted on 3724 raw 26-D vectors
- `knn_model.joblib` — KNeighborsClassifier (k=3, cosine, distance)
- `label_encoder.joblib` — LabelEncoder for 11 classes
- `metadata.json` — k, metric, cv_metrics
- `centroids.json` — per-class centroids **in scaled space** (used by radar)
- `centroids_raw.json` — per-class raw means (before scaling)
- `train_meta.json` — accuracy, std, samples, etc.
- `confusion_matrix.png` — from evaluate.py

These are gitignored except `README.md`. Commit them if you want CI to skip training.

To re-export frontend parity: copy `centroids.json` into `lib/dataset.ts` INSTRUMENTS centroids (already in sync from initial training).
