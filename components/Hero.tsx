"use client";
import { MODEL_META } from "@/lib/dataset";

export default function Hero(){
  return (
    <div className="relative">
      <div className="absolute inset-0 grid-lines opacity-[0.55] pointer-events-none" />
      <div className="relative mx-auto max-w-[1240px] px-5 md:px-6 pt-10 md:pt-14 pb-8">
        <div className="grid md:grid-cols-[1.35fr_0.85fr] gap-6 items-start">
          {/* left */}
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-surface border border-line mono-label !text-[10px]">
              <span className="px-1.5 py-0.5 rounded-full bg-lime text-void font-[700]">NEW</span>
              EXPLAINABLE AUDIO INTELLIGENCE • IRMAS 11K+
            </div>
            <h1 className="mt-5 font-display font-[600] tracking-[-0.03em] leading-[0.9] text-[42px] md:text-[56px]">
              Identify <span className="hand-underline">instruments</span><br />
              <span className="font-serif font-[300] italic tracking-[-0.02em] text-paper">in any audio —</span><br />
              with proof.
            </h1>
            <p className="mt-4 max-w-[62ch] text-[14.5px] leading-6 text-muted">
              Upload a clip & get a <span className="text-paper">26-dimensional spectral fingerprint</span> (13 MFCCs + Δ, Chroma, Centroid, Rolloff, ZCR, Bandwidth, Flatness, RMS…) classified by <span className="text-paper">K-Nearest Neighbours</span> with cosine similarity. Every prediction is explainable: radar, probabilities, waveform & CSV export.
            </p>

            <div className="mt-6 flex flex-wrap gap-2">
              <a href="#analyze" className="btn-lime px-6 py-3 text-sm">Try live analysis</a>
              <a href="#api-docs" className="btn-ghost px-6 py-3">Read API docs</a>
              <span className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-surface-2 border border-line mono-label">
                <span className="h-2 w-2 rounded-full bg-lime" /> 11 instruments • 3724 training vectors
              </span>
            </div>

            <div className="mt-8 grid grid-cols-4 gap-3 max-w-[520px]">
              {[
                {k:"Accuracy", v:`${MODEL_META.accuracy}%`, s:"±0.92% 5-fold CV"},
                {k:"Precision", v:`${MODEL_META.precision}%`, s:"macro avg"},
                {k:"Classes", v:"11", s:"IRMAS family"},
                {k:"Features", v:"26-D", s:"DSP vector"},
              ].map(c=>(
                <div key={c.k} className="card p-3">
                  <div className="mono-label !text-[9px]">{c.k}</div>
                  <div className="font-display font-[700] text-[18px] tracking-[-0.02em]">{c.v}</div>
                  <div className="mono-label !text-[9px] !tracking-[0.08em] opacity-70">{c.s}</div>
                </div>
              ))}
            </div>
          </div>

          {/* right — model card */}
          <div className="card p-4 md:p-5">
            <div className="flex items-center justify-between">
              <div className="mono-label">Model card — KNN-Cosine</div>
              <span className="px-2 py-1 rounded-full bg-lime text-void font-mono text-[10px] font-bold">K=3 • DIST-WEIGHTED</span>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2">
              {[
                {l:"Distance", v:"Cosine"},
                {l:"Scaler", v:"StandardScaler"},
                {l:"Hop", v:"1024"},
                {l:"SR", v:"22.05kHz"},
                {l:"Window", v:"Hanning 2048"},
                {l:"Max", v:"30s clip"},
              ].map(i=>(
                <div key={i.l} className="rounded-xl bg-void border border-line p-2.5">
                  <div className="mono-label !text-[8px]">{i.l}</div>
                  <div className="font-mono text-xs text-paper">{i.v}</div>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-xl overflow-hidden border border-line">
              <div className="px-3 py-2 mono-label bg-surface-2 border-b border-line flex justify-between">
                <span>Class balance</span><span className="text-paper">3724 samples</span>
              </div>
              <div className="p-3 space-y-2 bg-void">
                {[
                  ["Saxophone",626],["Flute",447],["Cello",388],["Organ",356],["Violin",330],
                  ["Elec. Guitar",308],["Clarinet",269],["Piano",250],["Acoustic Gtr",250],["Voice",250],["Trumpet",250]
                ].map(([name, n])=>(
                  <div key={name} className="flex items-center gap-3">
                    <div className="w-[92px] mono-label !text-[10px] !tracking-[0.06em] truncate">{name}</div>
                    <div className="flex-1 h-2 rounded-full bg-surface-3 overflow-hidden">
                      <div className="h-full bg-lime" style={{width:`${(n as number)/626*100}%`}} />
                    </div>
                    <div className="mono-num text-[11px] text-muted w-8 text-right">{n as number}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-3 mono-label !text-[10px] leading-4">
              Trained on IRMAS • Evaluated with <span className="text-paper">5-Fold Stratified CV</span> — no black box. See <a href="#pipeline" className="text-lime underline">pipeline</a>.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
