"""
IRMAS dataset loader.

Expected layout (IRMAS v2):
  data/IRMAS/TrainingData/{class}/  — .wav files
  data/IRMAS/TestingData/           — not used for training

Classes: Acoustic Guitar, Cello, Clarinet, Electric Guitar, Flute, Human Voice,
         Organ, Piano, Saxophone, Trumpet, Violin

If dataset not present, this module falls back to synthetic data derived from
frontend centroids (so training can run without downloading 11K files).
"""
import os, random
from pathlib import Path
import numpy as np

from config import DATASET_DIR, CLASSES

CLASS_DIR_MAP = {
    "Acoustic Guitar": ["acoustic_guitar","acoustic","gac"],
    "Cello": ["cello","cel"],
    "Clarinet": ["clarinet","cla"],
    "Electric Guitar": ["electric_guitar","electric","gel"],
    "Flute": ["flute","flu"],
    "Human Voice": ["human_voice","voice","voi","singing"],
    "Organ": ["organ","org"],
    "Piano": ["piano","pia"],
    "Saxophone": ["saxophone","sax"],
    "Trumpet": ["trumpet","tru"],
    "Violin": ["violin","vio"],
}

def scan_irmas(root: Path = DATASET_DIR):
    """Return list of (path, label) tuples if IRMAS found, else []"""
    if not root.exists():
        return []
    out=[]
    # Walk root for known class folders
    for label in CLASSES:
        candidates=[]
        # try exact & variations
        for cand in [root / label, root / "TrainingData" / label, root / "train" / label]:
            if cand.exists(): candidates.append(cand)
        # also fuzzy scan
        if not candidates:
            for p in root.rglob("*"):
                if p.is_dir() and p.name.lower().replace(" ","_") in [x.lower() for xs in CLASS_DIR_MAP.values() for x in xs]:
                    # map back to label
                    for lbl, aliases in CLASS_DIR_MAP.items():
                        if p.name.lower() in [a.lower() for a in aliases]:
                            candidates.append(p)
                            break
        for cand in candidates:
            for wav in cand.rglob("*.wav"):
                out.append((str(wav), label))
    # generic fallback: any wav under root → try to infer label from parent folder name
    if not out and root.exists():
        for wav in root.rglob("*.wav"):
            parent = wav.parent.name
            # guess label
            guess = None
            for lbl, aliases in CLASS_DIR_MAP.items():
                if parent.lower() in [a.lower() for a in aliases] or parent.lower().replace(" ","_") in aliases:
                    guess=lbl; break
            if guess: out.append((str(wav), guess))
    return out

def load_or_synth(n_per_class=250, jitter=0.35, seed=0):
    """
    If IRMAS not found, synthesize X,y from frontend centroids with Gaussian jitter.
    This keeps the ML pipeline runnable for evaluation and Vercel demo parity.
    """
    items = scan_irmas()
    if items:
        print(f"[dataset] found {len(items)} IRMAS files at {DATASET_DIR}")
        return items

    print("[dataset] IRMAS not found — synthesizing from centroids (no download needed).")
    # Import centroids
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from lib.dataset import INSTRUMENTS
        # use Python-side stored centroids (mirrors frontend)
        centroids = {ins["label"] if isinstance(ins, dict) else ins.label: np.array(ins["centroid"] if isinstance(ins, dict) else ins.centroid) for ins in INSTRUMENTS}
    except Exception:
        # hard fallback: load from models/centroids.json or random
        centroids = {lbl: np.random.randn(26)*0.4 for lbl in CLASSES}
        print("[dataset] centroids fallback to random")

    rng = np.random.default_rng(seed)
    synth_paths = []  # virtual paths
    # We return raw X,y directly if synthesized
    X, y = [], []
    target_counts = {
        "Acoustic Guitar":250,"Cello":388,"Clarinet":269,"Electric Guitar":308,"Flute":447,
        "Human Voice":250,"Organ":356,"Piano":250,"Saxophone":626,"Trumpet":250,"Violin":330
    }
    for lbl in CLASSES:
        n = target_counts.get(lbl, n_per_class)
        base = centroids.get(lbl, np.zeros(26))
        for _ in range(n):
            jitter_vec = rng.normal(0, jitter, size=26) * (0.6 + np.abs(base)*0.12)
            X.append(base + jitter_vec)
            y.append(lbl)
    X = np.vstack(X); y = np.array(y)
    # Return as synthetic flag + arrays
    return (X, y)

def split_paths(paths_with_labels, test_size=0.2, seed=42):
    """Stratified split for real file lists"""
    from sklearn.model_selection import train_test_split
    paths = np.array([p for p,_ in paths_with_labels])
    labels = np.array([l for _,l in paths_with_labels])
    return train_test_split(paths, labels, test_size=test_size, random_state=seed, stratify=labels)
