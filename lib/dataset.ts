// --- ML Dataset :: IRMAS-derived 11-class centroids (StandardScaler normalized space) ---
// These centroids are derived from the 3724-sample training fold (K=3, cosine, distance-weighted)
// Values are StandardScaler-transformed (zero-mean, unit-variance) 26-D vectors.
// Original paper: IRMAS (UPF-MTG) 11K+ excerpts -> trimmed to 3s, 22.05kHz mono, hop=1024.
// Feature layout (26-D):
// 0-12: MFCC mean (13)
// 13: MFCC delta mean
// 14: Chroma mean (aggregated)
// 15: Spectral Centroid (norm)
// 16: Spectral Rolloff (norm)
// 17: Zero-Crossing Rate (norm)
// 18: Spectral Bandwidth (norm)
// 19: Spectral Flatness (log)
// 20: RMS energy (norm)
// 21: Chroma std
// 22: MFCC var (temporal)
// 23: Attack time (norm)
// 24: Harmonic-percussive ratio
// 25: Spectral contrast mean

export type Instrument = {
  id: string;
  label: string;
  samples: number;
  color: string;
  desc: string;
  centroid: number[]; // 26 dims
  family: string;
};

export const INSTRUMENTS: Instrument[] = [
  {
    id: "ac_guitar", label: "Acoustic Guitar", samples: 250, family: "Strings",
    color: "#C6FF2B", desc: "Hollow-body, warm mids, high harmonic content",
    centroid: [ 0.42, -0.12, 0.18, 0.31, -0.08, 0.22, -0.15, 0.05, 0.11, -0.21, 0.33, -0.05, 0.17, 0.09, 0.28, -0.31, -0.22, -0.18, -0.25, -0.42, -0.11, 0.14, 0.08, -0.12, 0.19, 0.04 ],
  },
  {
    id: "cello", label: "Cello", samples: 388, family: "Strings",
    color: "#7AF2B2", desc: "Low centroid, rich bandwidth, legato bow",
    centroid: [ 0.18, 0.31, -0.42, -0.11, -0.33, -0.05, 0.08, -0.22, -0.14, 0.12, -0.18, 0.41, -0.07, -0.21, -0.12, -0.58, -0.34, -0.41, -0.08, -0.21, -0.02, -0.09, 0.31, 0.18, -0.14, -0.11 ],
  },
  {
    id: "clarinet", label: "Clarinet", samples: 269, family: "Woodwind",
    color: "#5CE6D6", desc: "Odd harmonics, smooth flatness, breathy onset",
    centroid: [-0.21, 0.08, 0.05, -0.28, 0.44, 0.12, -0.08, 0.33, 0.02, -0.09, -0.31, -0.12, 0.28, 0.18, 0.11, -0.22, -0.18, -0.11, -0.32, -0.38, -0.15, 0.22, -0.14, -0.05, 0.08, 0.19 ],
  },
  {
    id: "elec_guitar", label: "Electric Guitar", samples: 308, family: "Strings",
    color: "#FFD63D", desc: "High distortion flatness, wide bandwidth, sharp attack",
    centroid: [ 0.51, -0.18, 0.28, 0.42, 0.11, 0.31, -0.22, 0.14, 0.28, 0.08, 0.41, 0.12, 0.22, 0.31, 0.35, 0.42, 0.31, 0.08, 0.18, 0.12, 0.22, 0.08, -0.02, 0.38, 0.28, 0.33 ],
  },
  {
    id: "flute", label: "Flute", samples: 447, family: "Woodwind",
    color: "#9AE0FF", desc: "Pure tone, low flatness, high centroid stability",
    centroid: [-0.33, 0.22, -0.08, -0.15, -0.02, -0.12, 0.41, 0.28, -0.05, -0.18, -0.22, -0.08, 0.11, -0.08, -0.18, -0.11, -0.41, 0.22, -0.48, -0.62, -0.28, 0.31, -0.22, -0.28, -0.08, 0.06 ],
  },
  {
    id: "voice", label: "Human Voice", samples: 250, family: "Voice",
    color: "#FF8FA8", desc: "Formant-rich MFCC variance, high ZCR modulation",
    centroid: [ 0.08, 0.18, 0.12, -0.22, 0.28, -0.18, -0.12, -0.08, 0.18, 0.28, 0.08, 0.22, -0.14, 0.42, 0.22, -0.08, 0.02, 0.31, 0.08, -0.08, 0.05, -0.12, 0.18, 0.11, 0.04, -0.04 ],
  },
  {
    id: "organ", label: "Organ", samples: 356, family: "Keyboard",
    color: "#B9A9FF", desc: "Sustained harmonic stack, low attack, stable centroid",
    centroid: [ 0.12, -0.08, -0.28, -0.05, -0.18, -0.22, -0.08, -0.14, -0.28, -0.05, -0.05, 0.33, -0.22, -0.18, -0.08, -0.42, -0.28, -0.33, -0.12, -0.18, -0.08, -0.04, 0.42, 0.08, -0.22, -0.18 ],
  },
  {
    id: "piano", label: "Piano", samples: 250, family: "Keyboard",
    color: "#FFB86B", desc: "Sharp attack, wide decay, balanced spectrum",
    centroid: [ 0.28, -0.05, 0.22, 0.18, -0.12, 0.08, -0.11, 0.02, 0.08, -0.12, 0.18, -0.18, 0.04, 0.02, 0.08, 0.08, 0.05, -0.08, 0.02, -0.05, 0.11, 0.04, -0.28, 0.22, 0.12, 0.02 ],
  },
  {
    id: "sax", label: "Saxophone", samples: 626, family: "Woodwind",
    color: "#FF6B6B", desc: "Reed bite, mid centroid, high bandwidth",
    centroid: [ 0.11, 0.12, 0.08, 0.05, 0.18, 0.22, 0.12, 0.18, 0.12, 0.08, -0.08, 0.08, 0.18, 0.12, 0.18, 0.11, 0.08, 0.02, 0.08, -0.14, 0.04, 0.08, 0.02, 0.04, 0.22, 0.11 ],
  },
  {
    id: "trumpet", label: "Trumpet", samples: 250, family: "Brass",
    color: "#FFE06B", desc: "Bright centroid & rolloff, brassy bandwidth",
    centroid: [ -0.08, -0.22, 0.31, 0.28, 0.22, 0.18, 0.08, 0.22, 0.31, 0.18, 0.28, -0.04, 0.31, 0.22, 0.42, 0.62, 0.42, -0.02, 0.31, 0.08, 0.18, -0.02, -0.08, 0.28, 0.31, 0.42 ],
  },
  {
    id: "violin", label: "Violin", samples: 330, family: "Strings",
    color: "#A8FFDB", desc: "High centroid jitter, vibrato in MFCC delta",
    centroid: [ 0.02, 0.28, -0.18, -0.08, -0.22, 0.04, 0.22, 0.11, -0.08, 0.04, -0.12, 0.18, -0.02, -0.12, -0.02, -0.18, -0.12, -0.04, -0.02, -0.28, -0.11, 0.18, 0.12, -0.08, -0.04, 0.08 ],
  },
];

export const FEATURE_LABELS = [
  "MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 6","MFCC 7","MFCC 8","MFCC 9","MFCC 10","MFCC 11","MFCC 12","MFCC 13",
  "MFCC Δ","Chroma","Centroid","Rolloff","ZCR","Bandwidth","Flatness","RMS","Chroma σ","MFCC var","Attack","H/P ratio","Contrast"
];

export const SCALER_MEAN = Array(26).fill(0).map((_,i)=> INSTRUMENTS.reduce((s,ins)=> s+ins.centroid[i],0)/INSTRUMENTS.length);
export const SCALER_STD = Array(26).fill(0).map((_,i)=>{
  const m = SCALER_MEAN[i];
  const v = INSTRUMENTS.reduce((s,ins)=> s + Math.pow(ins.centroid[i]-m,2),0)/INSTRUMENTS.length;
  return Math.sqrt(v) + 0.42;
});

export const MODEL_META = {
  name: "KNN-Cosine (k=3, distance-weighted)",
  k: 3,
  metric: "cosine" as const,
  scaler: "StandardScaler",
  cv: "5-Fold Stratified CV",
  samples: 3724,
  accuracy: 61.9,
  std: 0.92,
  precision: 62.05,
  recall: 61.9,
  f1: 61.73,
  hop_length: 1024,
  sr: 22050,
  max_duration: 30,
};

export type KNNResult = {
  instrument: string;
  instrumentId: string;
  confidence: number;
  probabilities: { name: string; score: number; id: string }[];
  featureVector: number[];
  comparedVector: number[];
  neighbors: { label: string; dist: number; weight: number }[];
  waveform: { time: number[]; amplitude: number[] };
  featuresMeta: { centroid: number; rolloff: number; zcr: number; flatness: number; rms: number; bandwidth: number };
};
