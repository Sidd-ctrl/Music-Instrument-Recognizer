import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Audio processing
SAMPLE_RATE = 22050
MAX_DURATION = 30.0  # seconds
HOP_LENGTH = 1024
N_FFT = 2048
N_MFCC = 13
N_MELS = 40
FMAX = SAMPLE_RATE // 2
WIN_LENGTH = N_FFT

# Model
K = 3
METRIC = "cosine"  # cosine similarity (1 - cosine)
WEIGHTS = "distance"  # distance-weighted voting
SCALER = "StandardScaler"

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = Path(os.getenv("IRMAS_DATASET_DIR", str(PROJECT_ROOT / "data" / "IRMAS")))
ARTIFACTS = {
    "scaler": MODELS_DIR / "scaler.joblib",
    "knn": MODELS_DIR / "knn_model.joblib",
    "label_encoder": MODELS_DIR / "label_encoder.joblib",
    "metadata": MODELS_DIR / "metadata.json",
    "centroids": MODELS_DIR / "centroids.json",
}

# Classes (IRMAS 11)
CLASSES = [
    "Acoustic Guitar","Cello","Clarinet","Electric Guitar","Flute",
    "Human Voice","Organ","Piano","Saxophone","Trumpet","Violin"
]

# Server
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
