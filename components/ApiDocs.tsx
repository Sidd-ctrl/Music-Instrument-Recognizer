export default function ApiDocs(){
  return (
    <div id="api-docs" className="card p-4 md:p-5">
      <div className="mono-label">API documentation — Flask + Next.js dual stack</div>
      <div className="mt-4 grid md:grid-cols-2 gap-4">
        <div className="rounded-2xl bg-void border border-line overflow-hidden">
          <div className="px-3 py-2 bg-surface-2 border-b border-line mono-label flex justify-between"><span>POST /api/analyze</span><span className="text-mint">200 JSON</span></div>
          <div className="p-3">
            <div className="mono-label !text-muted">Request — multipart/form-data</div>
            <pre className="mt-2 p-3 rounded-xl bg-surface border border-line text-xs font-mono overflow-auto">curl -X POST http://localhost:5000/analyze \
  -F "audioFile=@sample.wav"

# Vercel (Next API, no Python needed)
curl -X POST /api/analyze \
  -F "audio=@sample.mp3"</pre>
            <div className="mono-label !text-muted mt-3">Response</div>
            <pre className="mt-2 p-3 rounded-xl bg-surface border border-line text-xs font-mono overflow-auto">{`{
  "instrument": "Piano",
  "confidence_score": 87.34,
  "knn_probabilities": [
    {"name":"Piano","score":87.34}, {"name":"Organ","score":5.21}
  ],
  "feature_vector": [26 floats],
  "compared_vector": [26 floats],
  "waveform": {"time":[1000], "amplitude":[1000]}
}`}</pre>
          </div>
        </div>
        <div className="space-y-4">
          <div className="rounded-2xl bg-void border border-line overflow-hidden">
            <div className="px-3 py-2 bg-surface-2 border-b border-line mono-label">GET /health &amp; /model/info</div>
            <pre className="m-3 p-3 rounded-xl bg-surface border border-line text-xs font-mono">GET /health → {"{"}"status":"ok","model_ready":true{"}"}
GET /model/info → {"{"}"k":3,"metric":"cosine","scaler":"StandardScaler",
  "accuracy":61.9,"samples":3724{"}"}</pre>
          </div>
          <div className="rounded-2xl bg-void border border-line p-4">
            <div className="font-display font-[600]">Vercel deployment</div>
            <ol className="mt-2 list-decimal list-inside text-[13px] text-muted space-y-1">
              <li><span className="text-paper">Frontend</span> auto-deploys (Next.js). No Python required — inference runs in browser via <span className="text-paper">audioFeatures.ts</span>.</li>
              <li>Optional <span className="text-paper">Python backend</span> on Render/Railway: set <code className="px-1 py-0.5 rounded bg-surface-2 border border-line font-mono text-xs">NEXT_PUBLIC_API_URL</code>.</li>
              <li>Or use Vercel Python functions: <code className="font-mono text-xs">api/index.py</code> (FLASK) is included.</li>
            </ol>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {[
                ["PORT","5000"],
                ["FLASK_DEBUG","true"],
                ["CORS","*"],
              ].map(([k,v])=>(
                <div key={k} className="rounded-lg bg-surface-2 border border-line px-3 py-2"><div className="mono-label !text-[9px]">{k}</div><div className="font-mono text-xs">{v}</div></div>
              ))}
            </div>
          </div>
          <div className="rounded-2xl border border-lime bg-[rgba(198,255,43,0.06)] p-3 mono-label !text-[11px] leading-5">
            ML note: Backend uses <span className="text-void bg-lime px-1 rounded font-bold">librosa 0.10</span> for exact 26-D extraction; frontend approximates with a hand-rolled FFT + mel-filterbank so the demo works offline. Cosine KNN is identical.
          </div>
        </div>
      </div>
    </div>
  );
}
