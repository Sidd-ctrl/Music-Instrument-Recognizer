"use client";
import { INSTRUMENTS } from "@/lib/dataset";
import { useRef, useState } from "react";

export default function SamplesGrid(){
  const ctxRef = useRef<AudioContext|null>(null);
  const [playing,setPlaying]=useState<string|null>(null);

  function toneFor(id:string){
    // map instrument to base freq & timbre
    const map: any = {
      ac_guitar: {f:196, type:"triangle", detune:2},
      cello:{f:130, type:"sawtooth", detune:1},
      clarinet:{f:262, type:"sine", detune:0},
      elec_guitar:{f:220, type:"square", detune:6},
      flute:{f:392, type:"sine", detune:0},
      voice:{f:247, type:"triangle", detune:3},
      organ:{f:165, type:"sawtooth", detune:0},
      piano:{f:261, type:"triangle", detune:1},
      sax:{f:208, type:"sawtooth", detune:4},
      trumpet:{f:330, type:"square", detune:2},
      violin:{f:294, type:"sawtooth", detune:5},
    };
    return map[id]||{f:220,type:"sine",detune:0};
  }

  async function play(id:string){
    if(!ctxRef.current) ctxRef.current = new (window.AudioContext||(window as any).webkitAudioContext)();
    const ctx=ctxRef.current!;
    if(playing){ ctx.close(); ctxRef.current=new (window.AudioContext||(window as any).webkitAudioContext)(); setPlaying(null); return; }
    const {f, type, detune} = toneFor(id);
    const o1=ctx.createOscillator(); o1.type=type; o1.frequency.value=f;
    const o2=ctx.createOscillator(); o2.type="sine"; o2.frequency.value=f*2; o2.detune.value=detune;
    const gain=ctx.createGain(); gain.gain.value=0.0;
    const filter=ctx.createBiquadFilter(); filter.type="lowpass"; filter.frequency.value= 1800 + Math.random()*1200;
    o1.connect(filter); o2.connect(filter); filter.connect(gain); gain.connect(ctx.destination);
    gain.gain.linearRampToValueAtTime(0.18, ctx.currentTime+0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime+1.1);
    o1.start(); o2.start(); o1.stop(ctx.currentTime+1.15); o2.stop(ctx.currentTime+1.15);
    setPlaying(id); setTimeout(()=> setPlaying(null), 1200);
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div className="mono-label">Demo soundboard — tap to audition synthetic timbres</div>
        <span className="mono-label !text-[10px] px-2 py-1 rounded-full bg-lime text-void font-bold">11 classes • WebAudio</span>
      </div>
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {INSTRUMENTS.map(ins=>(
          <button key={ins.id} onClick={()=>play(ins.id)} className={`text-left rounded-2xl border p-3 transition relative overflow-hidden ${playing===ins.id? "bg-lime border-lime text-void":"bg-void border-line hover:border-line-strong text-paper"}`}>
            <div className="mono-label !text-[9px]" style={{color: playing===ins.id? "#0A0F0A": ins.color}}>{ins.family.toUpperCase()}</div>
            <div className="font-display font-[600] leading-none mt-1 text-[14px]">{ins.label}</div>
            <div className="mono-label !text-[9px] mt-1">{ins.samples} samples</div>
            <div className={`mt-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-mono ${playing===ins.id? "bg-void text-lime border-void":"bg-surface-2 border-line text-paper"}`}>
              {playing===ins.id? "■ Stop":"▶ Play"}
            </div>
          </button>
        ))}
      </div>
      <div className="mt-3 mono-label !text-[10px]">These are procedurally synthesized previews (not IRMAS). Upload real audio for true inference.</div>
    </div>
  );
}
