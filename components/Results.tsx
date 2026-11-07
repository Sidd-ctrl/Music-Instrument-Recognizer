"use client";
import { useEffect, useRef } from "react";
import { FEATURE_LABELS, INSTRUMENTS } from "@/lib/dataset";
import type { KNNResult } from "@/lib/dataset";
import { downloadCSV } from "@/lib/utils";

function Radar({ a, b, labels }:{ a:number[]; b:number[]; labels:string[] }){
  // 8 axes radar for readability — pick representative dims
  const idx = [0,1,4,14,15,16,17,19]; // MFCC1,2,5, chroma, centroid, rolloff, zcr, flatness
  const picks = idx.map(i=> labels[i]);
  const size=200, cx=100, cy=100, r=78;
  const angle = (i:number)=> (Math.PI*2*i/picks.length) - Math.PI/2;
  // normalize a,b to -1..1 already; map to 0..r with offset 50
  const toXY = (vals:number[])=> vals.map((v,i)=>{
    const norm = Math.max(-1.2, Math.min(1.2, v)); // clamp
    const rad = ((norm+1.2)/2.4)*r; // 0..r
    return [cx + rad*Math.cos(angle(i)), cy + rad*Math.sin(angle(i))];
  });
  const av = idx.map(i=> a[i]); const bv = idx.map(i=> b[i]);
  const aPts = toXY(av); const bPts = toXY(bv);
  const poly = (pts:number[][])=> pts.map(p=> p.join(",")).join(" ");
  return (
    <svg viewBox="0 0 200 200" className="w-full h-auto">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#19351D" strokeWidth="1" />
      <circle cx={cx} cy={cy} r={r*0.66} fill="none" stroke="#19351D" strokeDasharray="4 4" />
      <circle cx={cx} cy={cy} r={r*0.33} fill="none" stroke="#19351D" strokeWidth="0.8" />
      {picks.map((_,i)=>{
        const x=cx + r*Math.cos(angle(i)), y=cy + r*Math.sin(angle(i));
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#1E3520" strokeWidth="0.9" />;
      })}
      <polygon points={poly(bPts)} fill="rgba(198,255,43,0.18)" stroke="#C6FF2B" strokeWidth="1.6" />
      <polygon points={poly(aPts)} fill="rgba(0,229,153,0.18)" stroke="#00E599" strokeWidth="1.8" />
      {picks.map((lb,i)=>{
        const x=cx + (r+14)*Math.cos(angle(i)), y=cy + (r+14)*Math.sin(angle(i));
        return <text key={lb} x={x} y={y} textAnchor="middle" dominantBaseline="middle" className="fill-muted" style={{fontSize:"6px", fontFamily:"Fragment Mono"}}>{lb}</text>;
      })}
    </svg>
  );
}

function Waveform({ time, amplitude }:{ time:number[]; amplitude:number[] }){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{
    const canvas=ref.current; if(!canvas) return;
    const dpr=window.devicePixelRatio||1;
    const rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
    const ctx=canvas.getContext("2d")!; ctx.scale(dpr,dpr);
    const W=rect.width, H=rect.height;
    ctx.clearRect(0,0,W,H);
    // grid
    ctx.strokeStyle="rgba(30,53,32,0.45)"; ctx.lineWidth=1;
    for(let x=0;x<W;x+=42){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    ctx.strokeStyle="rgba(30,53,32,0.9)"; ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
    // waveform
    ctx.strokeStyle="#C6FF2B"; ctx.lineWidth=1.4; ctx.beginPath();
    amplitude.forEach((v,i)=>{
      const x=(i/amplitude.length)*W;
      const y=H/2 - v* (H*0.42);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    // fill under
    ctx.lineTo(W,H/2); ctx.lineTo(0,H/2); ctx.closePath();
    ctx.fillStyle="rgba(198,255,43,0.08)"; ctx.fill();
    // time labels
    ctx.fillStyle="#6D8B6E"; ctx.font="10px Fragment Mono";
    ctx.fillText(`${time[0]?.toFixed(2) ?? 0}s`, 6, H-6);
    ctx.fillText(`${time[time.length-1]?.toFixed(2) ?? 0}s`, W-48, H-6);
  },[time, amplitude]);
  return <canvas ref={ref} className="w-full h-[132px] rounded-xl wave-wrap" />;
}

export default function Results({ data }:{ data: KNNResult & {fileName:string; duration:number} }){
  const main = INSTRUMENTS.find(i=> i.label===data.instrument)!;
  return (
    <div className="space-y-4">
      {/* predicted hero */}
      <div className="card p-4 md:p-5 flex flex-col md:flex-row gap-4 items-stretch">
        <div className="flex-1">
          <div className="mono-label flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-mint" /> Predicted instrument
          </div>
          <div className="mt-2 flex items-baseline gap-3">
            <h2 className="font-display font-[700] text-[28px] tracking-[-0.02em]">{data.instrument}</h2>
            <span className="px-2.5 py-1 rounded-full bg-lime text-void font-mono text-xs font-bold">{data.confidence.toFixed(2)}% confidence</span>
          </div>
          <div className="mono-label !text-[11px] mt-1">{main.desc} • {main.family} • {main.samples} training samples</div>

          <div className="mt-4 grid grid-cols-3 gap-2">
            <div className="rounded-xl bg-void border border-line p-3">
              <div className="mono-label !text-[9px]">Duration</div><div className="font-mono text-sm text-paper">{data.duration.toFixed(2)}s</div>
            </div>
            <div className="rounded-xl bg-void border border-line p-3">
              <div className="mono-label !text-[9px]">KNN votes (k=3)</div><div className="font-mono text-xs text-paper">{data.neighbors.map(n=> `${n.label} ${(1/(n.dist+1e-6)).toFixed(1)}`).join(" • ")}</div>
            </div>
            <div className="rounded-xl bg-void border border-line p-3">
              <div className="mono-label !text-[9px]">Cosine distance</div><div className="font-mono text-sm text-paper">{data.neighbors[0].dist.toFixed(4)}</div>
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <button onClick={()=>{
              const rows = [
                ["field","value"],
                ["file", data.fileName],
                ["instrument", data.instrument],
                ["confidence", data.confidence.toFixed(2)],
                ...data.probabilities.map(p=> [p.name, p.score.toFixed(2)]),
                ...data.featureVector.map((v,i)=> [FEATURE_LABELS[i], v.toFixed(4)]),
              ];
              downloadCSV(`timbre-${data.instrument}-${Date.now()}.csv`, rows as any);
            }} className="btn-lime px-4 py-2 text-sm">Download CSV report</button>
            <button onClick={()=> navigator.clipboard.writeText(JSON.stringify(data, null, 2))} className="btn-ghost px-4 py-2">Copy JSON</button>
          </div>
        </div>

        <div className="md:w-[320px] rounded-2xl bg-void border border-line p-3 flex flex-col gap-3">
          <div className="mono-label">All class probabilities</div>
          <div className="space-y-1.5">
            {data.probabilities.map(p=>{
              const isTop=p.name===data.instrument;
              return (
                <div key={p.id} className={`flex items-center gap-2 rounded-full px-2.5 py-1.5 border ${isTop? "bg-lime text-void border-lime font-bold":"bg-surface-2 border-line text-paper"}`}>
                  <div className="w-28 mono-label !text-[10px] !tracking-[0.06em] truncate">{p.name}</div>
                  <div className="flex-1 h-2 rounded-full bg-void/50 overflow-hidden border border-line/40">
                    <div className={`h-full ${isTop? "bg-void":"bg-lime"}`} style={{width:`${p.score}%`}} />
                  </div>
                  <div className={`mono-num text-xs ${isTop? "text-void":"text-paper"}`}>{p.score.toFixed(1)}%</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* explainability */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="mono-label">Explainability — Radar: input vs DB avg</div>
          <div className="mt-3 grid grid-cols-[1.1fr_0.9fr] gap-4 items-center">
            <Radar a={data.featureVector} b={data.comparedVector} labels={FEATURE_LABELS} />
            <div className="space-y-2">
              <div className="flex items-center gap-2 mono-label"><span className="h-2.5 w-2.5 rounded-full bg-mint" /> Your audio</div>
              <div className="flex items-center gap-2 mono-label"><span className="h-2.5 w-2.5 rounded-full bg-lime" /> DB avg — {data.instrument}</div>
              <div className="rounded-xl bg-void border border-line p-3 mono-label !text-[10px] leading-4">
                The shape shows where your clip sits in the <span className="text-paper">26-D fingerprint</span>. Overlap = match; spikes = divergence. This is why KNN chose <span className="text-lime">{data.instrument}</span> (cosine measures angle, not magnitude).
              </div>
              <div className="grid grid-cols-2 gap-2 mono-label !text-[9px]">
                <div className="rounded-lg bg-surface-2 border border-line px-2 py-2">Centroid {data.featuresMeta.centroid.toFixed(0)} Hz</div>
                <div className="rounded-lg bg-surface-2 border border-line px-2 py-2">Rolloff {data.featuresMeta.rolloff.toFixed(0)} Hz</div>
                <div className="rounded-lg bg-surface-2 border border-line px-2 py-2">ZCR {(data.featuresMeta.zcr*100).toFixed(2)}%</div>
                <div className="rounded-lg bg-surface-2 border border-line px-2 py-2">RMS {(data.featuresMeta.rms).toFixed(3)}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="mono-label">Waveform — time-domain signal (downsampled 1000 pts)</div>
          <div className="mt-3"><Waveform time={data.waveform.time} amplitude={data.waveform.amplitude} /></div>
          <div className="mt-3 grid grid-cols-3 gap-2 mono-label">
            <div className="rounded-lg bg-void border border-line px-3 py-2"><div className="mono-label !text-[9px]">Bandwidth</div><div className="text-paper">{data.featuresMeta.bandwidth.toFixed(0)} Hz</div></div>
            <div className="rounded-lg bg-void border border-line px-3 py-2"><div className="mono-label !text-[9px]">Flatness</div><div className="text-paper">{data.featuresMeta.flatness.toExponential(2)}</div></div>
            <div className="rounded-lg bg-void border border-line px-3 py-2"><div className="mono-label !text-[9px]">H/P ratio</div><div className="text-paper">{(data.featureVector[24]).toFixed(2)}</div></div>
          </div>
        </div>
      </div>

      {/* feature table */}
      <div className="card p-4">
        <div className="flex items-center justify-between">
          <div className="mono-label">26-D Feature table — input vs DB average</div>
          <span className="mono-label !text-[10px] px-2 py-1 rounded-full bg-surface-3 border border-line">StandardScaler normalized</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead><tr className="mono-label !text-[10px] border-b border-line">
              <th className="py-2 pr-3">#</th><th className="py-2 pr-3">Feature</th><th className="py-2 pr-3">Your clip</th><th className="py-2 pr-3">DB avg ({data.instrument})</th><th className="py-2">Δ (abs)</th>
            </tr></thead>
            <tbody>
              {FEATURE_LABELS.map((lb,i)=>{
                const a=data.featureVector[i], b=data.comparedVector[i], d=Math.abs(a-b);
                const hot=d>0.6;
                return (
                  <tr key={lb} className="border-b border-line/60 hover:bg-surface-2/40">
                    <td className="py-1.5 pr-3 mono-num text-xs text-muted">{String(i+1).padStart(2,"0")}</td>
                    <td className="py-1.5 pr-3 mono-label !text-[11px] !text-paper">{lb}</td>
                    <td className="py-1.5 pr-3 mono-num text-xs" style={{color: hot? "#C6FF2B": "#EAF2E6"}}>{a.toFixed(3)}</td>
                    <td className="py-1.5 pr-3 mono-num text-xs text-muted">{b.toFixed(3)}</td>
                    <td className="py-1.5 mono-num text-xs"><span className={`px-2 py-0.5 rounded-full border ${hot? "bg-lime text-void border-lime":"bg-void text-muted border-line"}`}>{d.toFixed(3)}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
