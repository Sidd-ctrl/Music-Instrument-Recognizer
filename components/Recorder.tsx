"use client";
import { useState, useRef } from "react";
import { extractFeatures } from "@/lib/audioFeatures";
import { classify } from "@/lib/knn";
import type { KNNResult } from "@/lib/dataset";

export default function Recorder({ onResult, onAnalyzing }:{ onResult:(r:KNNResult & {fileName:string; duration:number})=>void; onAnalyzing:(b:boolean)=>void; }){
  const [rec, setRec]=useState(false);
  const [secs,setSecs]=useState(0);
  const mrRef=useRef<MediaRecorder|null>(null);
  const chunksRef=useRef<Blob[]>([]);
  const timerRef=useRef<any>(null);

  async function toggle(){
    if(rec){
      mrRef.current?.stop(); setRec(false); clearInterval(timerRef.current);
      return;
    }
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    const mr=new MediaRecorder(stream, {mimeType: "audio/webm"});
    chunksRef.current=[];
    mr.ondataavailable=e=>{ if(e.data.size>0) chunksRef.current.push(e.data); };
    mr.onstop=async ()=>{
      onAnalyzing(true);
      const blob=new Blob(chunksRef.current, {type:"audio/webm"});
      const ab=await blob.arrayBuffer();
      const ctx=new (window.AudioContext||(window as any).webkitAudioContext)();
      const buf=await ctx.decodeAudioData(ab.slice(0));
      const {vector, meta, waveform}=extractFeatures(buf);
      const cls=classify(vector, 3);
      onResult({ fileName:`mic-${new Date().toISOString().slice(11,19)}.webm`, duration: buf.duration, instrument:cls.instrument, instrumentId:cls.instrumentId, confidence:cls.confidence, probabilities:cls.probabilities, featureVector:vector, comparedVector:cls.comparedVector, neighbors:cls.neighbors, waveform, featuresMeta:meta });
      onAnalyzing(false);
      stream.getTracks().forEach(t=>t.stop());
    };
    mr.start(); mrRef.current=mr; setRec(true); setSecs(0);
    timerRef.current=setInterval(()=> setSecs(s=>s+1),1000);
  }

  return (
    <div className="rounded-2xl bg-void border border-line p-3 flex items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        <button onClick={toggle} className={`h-9 px-4 rounded-full font-mono text-sm font-bold flex items-center gap-2 ${rec?"bg-red-500 text-white":"bg-lime text-void"}`}>
          <span className={`h-2 w-2 rounded-full ${rec?"bg-white animate-pulse":"bg-void"}`} />{rec? "Stop":"● Record mic"}
        </button>
        <span className="mono-label">{rec? `Recording… ${secs}s / 30s` : "Browser mic → direct 26-D extraction (no upload)"}</span>
      </div>
      <span className="hidden sm:inline mono-label px-2.5 py-1 rounded-full bg-surface-2 border border-line">Mic uses WebAudio decode • same pipeline</span>
    </div>
  );
}
