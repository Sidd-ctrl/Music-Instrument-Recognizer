"use client";
import { useRef, useState } from "react";
import { extractFeatures } from "@/lib/audioFeatures";
import { classify } from "@/lib/knn";
import type { KNNResult } from "@/lib/dataset";

type Props = {
  onResult: (r: KNNResult & { fileName:string; duration:number })=>void;
  onAnalyzing: (b:boolean)=>void;
};

export default function UploadZone({ onResult, onAnalyzing }: Props){
  const inputRef=useRef<HTMLInputElement>(null);
  const [dragOver,setDragOver]=useState(false);
  const [fileName,setFileName]=useState<string>("");
  const [audioUrl,setAudioUrl]=useState<string>("");
  const [k,setK]=useState(3);

  async function handleFile(file: File){
    if(!file) return;
    setFileName(file.name);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    onAnalyzing(true);
    try{
      const arrayBuf = await file.arrayBuffer();
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuf = await ctx.decodeAudioData(arrayBuf.slice(0));
      const { vector, meta, waveform } = extractFeatures(audioBuf);
      const cls = classify(vector, k);
      const duration = audioBuf.duration;
      onResult({
        fileName: file.name,
        duration,
        instrument: cls.instrument,
        instrumentId: cls.instrumentId,
        confidence: cls.confidence,
        probabilities: cls.probabilities,
        featureVector: vector,
        comparedVector: cls.comparedVector,
        neighbors: cls.neighbors,
        waveform,
        featuresMeta: meta,
      });
    } catch(e){
      console.error(e);
      alert("Could not decode audio. Try WAV/MP3/OGG/FLAC/M4A (≤30s).");
    } finally {
      onAnalyzing(false);
    }
  }

  return (
    <div className="card p-3 md:p-4" id="analyze">
      <div className="flex items-center justify-between gap-4">
        <div className="mono-label">Analyze — drop audio here or click to browse</div>
        <div className="hidden md:flex items-center gap-2">
          <label className="mono-label flex items-center gap-2">K
            <input type="range" min={1} max={11} step={2} value={k} onChange={e=>setK(parseInt(e.target.value))} className="accent-lime w-20" />
            <span className="px-2 py-1 rounded-full bg-void border border-line mono-num text-paper">{k}</span>
          </label>
          <span className="px-2 py-1 rounded-full bg-surface-2 border border-line mono-label !text-[10px]">WAV • MP3 • OGG • FLAC • M4A</span>
        </div>
      </div>

      <div
        onDragOver={e=>{e.preventDefault(); setDragOver(true);}}
        onDragLeave={()=>setDragOver(false)}
        onDrop={e=>{e.preventDefault(); setDragOver(false); const f=e.dataTransfer.files?.[0]; if(f) handleFile(f);}}
        onClick={()=>inputRef.current?.click()}
        className={`mt-3 rounded-2xl border-2 border-dashed p-6 md:p-8 text-center cursor-pointer transition relative overflow-hidden
        ${dragOver ? "border-lime bg-[rgba(198,255,43,0.06)]" : "border-line bg-void hover:border-line-strong"}`}
      >
        <div className="absolute inset-0 noise pointer-events-none" />
        <div className="relative">
          <div className="mx-auto h-12 w-12 rounded-full bg-surface-3 border border-line flex items-center justify-center">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M14 11V7a4 4 0 0 0-8 0v4" stroke="#C6FF2B" strokeWidth="1.7" strokeLinecap="round"/><path d="M6 15a6 6 0 0 0 12 0" stroke="#E8F0E6" strokeWidth="1.5" strokeLinecap="round"/><path d="M12 17v4m-3-2h6" stroke="#E8F0E6" strokeWidth="1.5" strokeLinecap="round"/></svg>
          </div>
          <div className="mt-3 font-display font-[600] text-[16px]">Click to browse or drag & drop</div>
          <div className="mono-label !text-[11px] mt-1">Auto-converts to mono 22.05kHz • Max 30s • Hanning 2048 • hop 1024</div>
          <div className="mt-4 inline-flex gap-2">
            <span className="btn-lime px-5 py-2.5 text-sm pointer-events-none">Choose file</span>
            <span className="btn-ghost px-4 py-2.5 pointer-events-none hidden sm:inline-flex">Record mic ↘</span>
          </div>
          {fileName && <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-2 border border-line mono-label !text-[11px]"><span className="h-2 w-2 rounded-full bg-mint animate-pulse" />{fileName}</div>}
        </div>
      </div>
      <input ref={inputRef} type="file" accept=".wav,.mp3,.ogg,.flac,.m4a,.aac,.wma,audio/*" className="hidden" onChange={e=>{const f=e.target.files?.[0]; if(f) handleFile(f);}} />

      {audioUrl && (
        <div className="mt-3 rounded-xl border border-line bg-void p-3 flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-lime flex items-center justify-center">▶</div>
          <audio controls src={audioUrl} className="flex-1 h-9" />
          <span className="mono-label hidden sm:inline">Preview • browser-decoded</span>
        </div>
      )}

      <div className="mt-3 grid grid-cols-3 gap-2">
        {[
          ["MFCC 1-13","timbre texture"],
          ["Chroma + Centroid","harmonic & brightness"],
          ["ZCR • Flatness • RMS","percussive & energy"],
        ].map(([a,b])=>(
          <div key={a} className="rounded-xl bg-surface-2 border border-line px-3 py-2.5">
            <div className="mono-label !text-[9px]">{a}</div>
            <div className="mono-label !text-[10px] !tracking-[0.06em] !text-paper">{b}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
