import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=22050, duration=5)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        return np.array([
            np.mean(mfcc),
            np.std(mfcc),
            np.mean(chroma),
            np.std(chroma),
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.std(rolloff)
        ])
    except:
        return None

base = r"D:\capstone\IRMAS-TrainingData"
X, y = [], []

print("Extracting features...")

for folder in os.listdir(base):
    fpath = os.path.join(base, folder)
    if not os.path.isdir(fpath):
        continue

    label = folder.lower().strip()

    for file in tqdm(os.listdir(fpath), desc=f"Loading {label}"):
        if file.endswith(".wav"):
            full = os.path.join(fpath, file)
            feat = extract_features(full)
            if feat is not None:
                X.append(feat)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Samples:", len(X), "Feature size:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

os.makedirs("model", exist_ok=True)
with open("model/knn_instrument_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Model saved.")
