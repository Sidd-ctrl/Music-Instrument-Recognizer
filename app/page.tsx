"use client";
import { useEffect, useState } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import UploadZone from "@/components/UploadZone";
import Recorder from "@/components/Recorder";
import Results from "@/components/Results";
import SamplesGrid from "@/components/SamplesGrid";
import Pipeline from "@/components/Pipeline";
import ApiDocs from "@/components/ApiDocs";
import { MODEL_META } from "@/lib/dataset";
import type { KNNResult } from "@/lib/dataset";

export default function Page(){
  const [result, setResult]=useState<(KNNResult & {fileName:string; duration:number})|null>(null);
  const [analyzing, setAnalyzing]=useState(false);
  const [history, setHistory]=useState<(KNNResult & {fileName:string; ts:number})[]>([]);

  useEffect(()=>{
    const h = localStorage.getItem("timbre_history");
    if(h) try{ setHistory(JSON.parse(h)); }catch{}
  },[]);
  useEffect(()=>{
    if(result){
      const nh=[{...result, ts:Date.now()}, ...history].slice(0,12);
      setHistory(nh as any);
      localStorage.setItem("timbre_history", JSON.stringify(nh));
    }
  // eslint-disable-next-line
  },[result]);

  return (
    <main className="min-h-screen">
      <Header />
      <Hero />

      {/* analyze */}
      <section className="mx-auto max-w-[1240px] px-5 md:px-6 py-6 space-y-4">
        <UploadZone onResult={setResult} onAnalyzing={setAnalyzing} />
        <Recorder onResult={setResult} onAnalyzing={setAnalyzing} />

        {analyzing && (
          <div className="card p-4 flex items-center gap-3">
            <div className="h-8 w-8 rounded-full border-2 border-line border-t-lime animate-spin" />
            <div>
              <div className="font-display font-[600]">Analysing… extracting 26-D fingerprint</div>
              <div className="mono-label">MFCC • Chroma • Centroid • Rolloff • ZCR • Bandwidth • Flatness • RMS — StandardScaler → KNN (cosine, k=3)</div>
            </div>
            <span className="ml-auto hidden sm:inline px-3 py-1 rounded-full bg-lime text-void font-mono text-xs font-bold animate-pulse">DSP running</span>
          </div>
        )}

        {result && <Results data={result} />}

        {!result && !analyzing && (
          <div className="card p-6 text-center">
            <div className="mx-auto max-w-[66ch]">
              <div className="font-display font-[600] text-[16px]">No file yet — try the demo board or drop an audio file above</div>
              <div className="mono-label mt-1">Uploads never leave your browser in Vercel mode. Python backend is optional for exact librosa parity.</div>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                <span className="px-3 py-1.5 rounded-full bg-void border border-line mono-label">Flask API optional</span>
                <span className="px-3 py-1.5 rounded-full bg-void border border-line mono-label">Client-side KNN identical</span>
                <span className="px-3 py-1.5 rounded-full bg-lime text-void font-mono text-xs font-bold">Explainable by design</span>
              </div>
            </div>
          </div>
        )}

        <SamplesGrid />
        <Pipeline />

        {/* metrics */}
        <div id="dataset" className="grid md:grid-cols-3 gap-4">
          <div className="card p-5">
            <div className="mono-label">Model performance — {MODEL_META.cv}</div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {[
                ["Accuracy", `${MODEL_META.accuracy}% ±${MODEL_META.std}%`],
                ["Precision", `${MODEL_META.precision}%`],
                ["Recall", `${MODEL_META.recall}%`],
                ["F1", `${MODEL_META.f1}%`],
              ].map(([k,v])=>(
                <div key={k} className="rounded-xl bg-void border border-line p-3">
                  <div className="mono-label !text-[9px]">{k}</div><div className="font-display font-[700]">{v}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 rounded-xl bg-void border border-line p-3">
              <div className="mono-label">K-tuning (grid search)</div>
              <div className="mt-2 space-y-1.5">
                {[
                  ["K=1 Euclidean", 57.2],["K=3 Cosine ★",61.9],["K=5 Cosine",60.4],["K=7 Manhattan",58.1],["K=11 Cosine",55.8]
                ].map(([lb,val])=>(
                  <div key={lb} className="flex items-center gap-2">
                    <div className="w-28 mono-label !text-[10px]">{lb as string}</div>
                    <div className="flex-1 h-2 rounded-full bg-surface-3 overflow-hidden"><div className="h-full bg-lime" style={{width:`${(val as number)}%`}} /></div>
                    <div className="mono-num text-xs text-paper w-12 text-right">{val as number}%</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card p-5">
            <div className="mono-label">Confusion drivers (from evaluation)</div>
            <div className="mt-3 text-[13px] leading-5 text-muted">
              Best: <span className="text-paper">Saxophone (626 samples)</span> — distinctive reed harmonics.<br/>
              Hardest: <span className="text-paper">Violin vs Cello</span> and <span className="text-paper">Acoustic vs Electric Guitar</span> — overlapping MFCC space.<br/>
              Voice is well-separated via formant variance (MFCC var).
            </div>
            <div className="mt-4 rounded-xl bg-void border border-line p-3">
              <div className="mono-label">Files & features</div>
              <div className="mt-2 grid grid-cols-2 gap-2 mono-label">
                <div className="rounded-lg bg-surface-2 border border-line px-3 py-2">SR 22050 Hz • mono</div>
                <div className="rounded-lg bg-surface-2 border border-line px-3 py-2">Hop 1024 • Win 2048</div>
                <div className="rounded-lg bg-surface-2 border border-line px-3 py-2">Hanning • 40 mel</div>
                <div className="rounded-lg bg-surface-2 border border-line px-3 py-2">DCT-II → 13 MFCC</div>
              </div>
            </div>
            <div className="mt-3 mono-label !text-[10px]">Backend training script: <code className="px-1 py-0.5 rounded bg-surface-2 border border-line">ml/train.py</code> reproduces this table.</div>
          </div>

          <div className="card p-5">
            <div className="mono-label flex justify-between items-center"><span>Recent history (local)</span><button onClick={()=>{localStorage.removeItem("timbre_history"); setHistory([]);}} className="mono-label !text-[9px] px-2 py-1 rounded-full bg-void border border-line">Clear</button></div>
            <div className="mt-3 space-y-2 max-h-[320px] overflow-auto pr-1">
              {history.length===0 && <div className="mono-label">No analyses yet — your runs will appear here.</div>}
              {history.map((h,i)=>(
                <div key={i} className="rounded-xl bg-void border border-line p-3 flex items-center justify-between">
                  <div>
                    <div className="font-display font-[600] text-sm">{h.instrument}</div>
                    <div className="mono-label !text-[10px] truncate max-w-[18ch]">{h.fileName} • {h.confidence.toFixed(1)}%</div>
                  </div>
                  <span className="mono-num text-xs text-muted">{new Date(h.ts).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 mono-label !text-[10px]">Stored in localStorage • 12 max • CSV export per run above.</div>
          </div>
        </div>

        <ApiDocs />

        <footer className="pt-4 pb-10">
          <div className="rounded-2xl border border-line bg-surface-2 p-4 flex flex-col md:flex-row items-center justify-between gap-3">
            <div className="mono-label">TIMBRE — Musical Instrument Recognition • Built with Next.js + Flask • Dataset: IRMAS (UPF-MTG) • Educational & Research Use</div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1.5 rounded-full bg-void border border-line mono-label">Python 3.11 • librosa • scikit-learn</span>
              <span className="px-3 py-1.5 rounded-full bg-lime text-void font-mono text-xs font-bold">Vercel-ready</span>
            </div>
          </div>
        </footer>
      </section>
    </main>
  );
}
