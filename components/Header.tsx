"use client";
export default function Header(){
  return (
    <header className="sticky top-0 z-40 backdrop-blur-xl bg-void/70 border-b border-line">
      <div className="mx-auto max-w-[1240px] px-5 md:px-6 h-[56px] flex items-center justify-between gap-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="h-7 w-7 rounded-[8px] bg-lime flex items-center justify-center">
              <div className="h-3 w-3 rounded-[3px] bg-void" />
            </div>
            <div className="leading-none">
              <div className="font-display font-[700] tracking-[-0.02em] text-[15px]">TIMBRE</div>
              <div className="mono-label !text-[9px] !tracking-[0.18em] -mt-[2px]">INSTRUMENT LAB v2.1</div>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-1 ml-2">
            {["Working"," "," ","Dataset"].map(l=>(
              <a key={l} href={`#${l.toLowerCase().replace(/ /g,'-')}`} className="px-3 py-1.5 rounded-full mono-label !text-muted hover:!text-paper hover:bg-surface-2 transition">{l}</a>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden sm:inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-2 border border-line text-[11px] font-mono">
            <span className="h-2 w-2 rounded-full bg-mint animate-pulse shadow-[0_0_8px_rgba(0,229,153,0.7)]" />
            KNN • cosine • k=3 — <span className="text-mint">ready</span>
          </span>
          <a href="https://github.com/Sidd-ctrl/Music-Instrument-Recognizer" target="_blank" className="btn-ghost px-4 py-2 hidden sm:inline-flex">GitHub</a>
          <a href="#analyze" className="btn-lime px-4 py-2 text-sm">Analyze →</a>
        </div>
      </div>
    </header>
  );
}
