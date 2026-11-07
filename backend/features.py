"""
DSP feature extraction — 26-D spectral fingerprint.

Mirrors frontend audioFeatures.ts but using librosa for exact parity.
Layout (26 dims):
  0-12  : MFCC means (13)
  13    : MFCC delta mean (temporal dynamics)
  14    : Chroma mean (harmonic content, aggregated)
  15    : Spectral centroid (brightness)
  16    : Spectral rolloff (85% energy)
  17    : Zero-crossing rate (percussiveness)
  18    : Spectral bandwidth
  19    : Spectral flatness (tonal vs noisy)
  20    : RMS energy
  21    : Chroma std
  22    : MFCC variance (temporal)
  23    : Attack time (onset to 90% RMS)
  24    : Harmonic/Percussive ratio
  25    : Spectral contrast mean
"""
import numpy as np
import librosa

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC, N_MELS, FMAX

FEATURE_LABELS = [
    "MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 6","MFCC 7","MFCC 8","MFCC 9","MFCC 10","MFCC 11","MFCC 12","MFCC 13",
    "MFCC Δ","Chroma","Centroid","Rolloff","ZCR","Bandwidth","Flatness","RMS","Chroma σ","MFCC var","Attack","H/P ratio","Contrast"
]

def load_audio(path: str, sr=SAMPLE_RATE, max_duration=30.0):
    y, sr = librosa.load(path, sr=sr, mono=True, duration=max_duration)
    return y, sr

def extract_26d(y: np.ndarray, sr=SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """
    Returns: (26-D vector, meta dict with waveform/aux for viz)
    """
    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
    mfcc_mean = mfcc.mean(axis=1)  # 13

    # Delta (temporal)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = float(mfcc_delta.mean())

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = float(chroma.mean())
    chroma_std = float(chroma.std())

    # --- Spectral ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Aggregates
    centroid_m = float(centroid.mean())
    rolloff_m = float(rolloff.mean())
    zcr_m = float(zcr.mean())
    bw_m = float(bandwidth.mean())
    flat_m = float(flatness.mean())
    rms_m = float(rms.mean())
    contrast_m = float(contrast.mean())

    # MFCC variance (temporal spread)
    mfcc_var = float(mfcc.var())

    # Attack time: time to reach 90% max RMS
    rms_curve = rms[0]
    max_rms = float(rms_curve.max()) if len(rms_curve) else 0
    attack_time = 0.0
    if max_rms > 0:
        thr = max_rms * 0.9
        idx = np.where(rms_curve >= thr)[0]
        if len(idx):
            attack_time = float(idx[0] * HOP_LENGTH / sr)

    # Harmonic/Percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_harm**2) + 1e-9)
    perc_energy = float(np.mean(y_perc**2) + 1e-9)
    hp_ratio = float(harm_energy / (perc_energy + 1e-9))
    # squash to ~0-2 range like frontend
    hp_ratio = float(np.clip(np.log1p(hp_ratio), 0, 2))

    # Build raw vector in interpretable ranges (to be scaled by StandardScaler)
    raw = np.array([
        *mfcc_mean,                # 0-12
        mfcc_delta_mean,            # 13
        chroma_mean * 12,           # 14
        centroid_m / 5000,          # 15
        rolloff_m / 8000,           # 16
        zcr_m * 16,                 # 17
        bw_m / 4000,                # 18
        np.log10(flat_m + 1e-6) + 6, # 19
        rms_m * 80,                 # 20
        chroma_std * 12,            # 21
        np.log10(mfcc_var + 1),     # 22
        attack_time * 6,            # 23
        hp_ratio * 1.0,             # 24
        contrast_m / 60,            # 25
    ], dtype=np.float64)

    # Waveform downsample for frontend viz (1000 pts)
    sr_vis = sr
    step = max(1, len(y)//1000)
    y_ds = y[::step][:1000]
    t_ds = np.arange(len(y_ds)) * step / sr_vis

    meta = {
        "centroid": centroid_m, "rolloff": rolloff_m, "zcr": zcr_m,
        "bandwidth": bw_m, "flatness": flat_m, "rms": rms_m,
        "duration": float(len(y)/sr), "sr": sr,
        "waveform": {"time": t_ds.tolist(), "amplitude": y_ds.tolist()},
        "raw": raw.tolist(),
    }
    return raw, meta


def extract_batch(paths: list[str]):
    """Helper for training: returns (X raw matrix, metas)"""
    import tqdm
    X, metas = [], []
    for p in tqdm.tqdm(paths):
        try:
            y, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True, duration=30)
            vec, meta = extract_26d(y, sr)
            X.append(vec); metas.append(meta)
        except Exception as e:
            print(f"[warn] failed {p}: {e}")
    return np.vstack(X) if X else np.empty((0, 26)), metas
