"""Thin wrapper to call backend/evaluate.py from ml/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import evaluate as ev
if __name__ == "__main__":
    ev.evaluate()
