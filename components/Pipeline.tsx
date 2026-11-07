export default function Pipeline(){
  return (
    <div id="pipeline" className="card p-4 md:p-6">
      <div className="mono-label">Pipeline — 4 stages from raw signal to explainable prediction</div>
      <div className="mt-4 grid md:grid-cols-4 gap-3">
        {[
          {n:"01", t:"Upload Audio", d:"Drag & drop WAV/MP3/OGG/FLAC/M4A ≤30s. Auto mono-22.05kHz conversion, Hanning 2048.", chips:["mono","22.05kHz","30s max"]},
          {n:"02", t:"Feature Extraction", d:"Librosa DSP → 26-D fingerprint: 13 MFCC, Δ, Chroma, Centroid, Rolloff, ZCR, Bandwidth, Flatness, RMS…", chips:["MFCC 13","Chroma","Spectral"]},
          {n:"03", t:"Normalize & Classify", d:"StandardScaler (fit on 3724) → KNN K=3 cosine distance, distance-weighted vote.", chips:["K=3","Cosine","Weighted"]},
          {n:"04", t:"Explain Results", d:"Radar (input vs DB avg), all-class bars, waveform, 26-D table, CSV export.", chips:["Radar","Bars","CSV"]},
        ].map(s=>(
          <div key={s.n} className="rounded-2xl bg-void border border-line p-4">
            <div className="flex items-center gap-2">
              <span className="h-7 w-7 rounded-full bg-lime text-void grid place-items-center font-mono text-xs font-bold">{s.n}</span>
              <span className="mono-label !text-muted">Step {s.n}</span>
            </div>
            <div className="font-display font-[600] mt-3">{s.t}</div>
            <div className="text-[13px] leading-5 text-muted mt-1">{s.d}</div>
            <div className="flex flex-wrap gap-1.5 mt-3">
              {s.chips.map(c=> <span key={c} className="px-2 py-1 rounded-full bg-surface-2 border border-line mono-label !text-[9px]">{c}</span>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
