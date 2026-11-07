"""
CLI inference for single file — uses trained model or falls back to synthetic centroids.

Usage:
  python ml/inference.py ./sample.wav
  python ml/inference.py ./sample.mp3 --json
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import librosa
from config import SAMPLE_RATE, MAX_DURATION
from features import extract_26d
from model import TimbreModel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("audio", help="path to audio file")
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--k", type=int, default=None)
    args=ap.parse_args()

    y,sr=librosa.load(args.audio, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
    raw, meta = extract_26d(y, sr)

    try:
        m=TimbreModel.load()
        if args.k: m.k=args.k
    except Exception as e:
        print(f"[inference] no trained model, using synthetic fallback: {e}")
        # fallback: build synthetic model on the fly
        from dataset import load_or_synth
        X,y_lab = load_or_synth()
        m=TimbreModel(k=args.k or 3)
        m.fit(X, y_lab)

    pred=m.predict_with_proba(raw)
    if args.json:
        print(json.dumps({**pred, "meta": meta}, indent=2))
    else:
        print(f" Instrument : {pred['instrument']}")
        print(f" Confidence : {pred['confidence']:.2f}%")
        print(" Probabilities:")
        for p in pred["probabilities"]:
            print(f"   {p['name']:17s} {p['score']:5.1f}%")
        print(" Neighbors:")
        for n in pred["neighbors"]:
            print(f"   {n['label']}  dist={n['dist']:.4f}  w={n['weight']:.2f}")
        print(f" Duration : {meta['duration']:.2f}s  centroid={meta['centroid']:.0f}Hz")

if __name__=="__main__":
    main()
