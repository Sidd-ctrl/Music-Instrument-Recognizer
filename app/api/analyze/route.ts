import { NextRequest, NextResponse } from "next/server";

// This route proxies to Python backend if NEXT_PUBLIC_API_URL is set,
// otherwise returns a hint that browser-side inference is used.
// Keeping it here makes Vercel deployment complete without a Python server.

export async function POST(req: NextRequest){
  const backend = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL;
  if(!backend){
    return NextResponse.json({
      hint: "No Python backend configured. Frontend runs KNN fully in-browser (audioFeatures.ts + knn.ts) with identical centroids & cosine metric. Set NEXT_PUBLIC_API_URL to your Flask URL to proxy.",
      useBrowser: true
    }, { status: 200 });
  }
  try{
    const form = await req.formData();
    const file = form.get("audio") || form.get("audioFile") || form.get("file");
    if(!file) return NextResponse.json({ error: "No audio file field found (expected 'audio' or 'audioFile')" }, { status: 400 });
    const fd = new FormData();
    fd.append("audioFile", file as Blob);
    const r = await fetch(`${backend.replace(/\/$/,"")}/analyze`, { method: "POST", body: fd as any });
    const data = await r.json();
    return NextResponse.json(data, { status: r.status });
  } catch(e:any){
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function GET(){
  return NextResponse.json({ status:"ok", mode: process.env.NEXT_PUBLIC_API_URL? "proxy":"browser", model_ready:true });
}
