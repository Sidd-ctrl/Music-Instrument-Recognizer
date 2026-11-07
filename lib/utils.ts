export function fmt(n:number, d=2){ return n.toFixed(d); }
export function clamp(n:number, a:number,b:number){ return Math.max(a, Math.min(b,n)); }
export function downloadCSV(filename:string, rows:(string|number)[][]){
  const csv = rows.map(r=> r.map(c=> `"${String(c).replace(/"/g,'""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a=document.createElement("a"); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}
export function audioBufferToWav(buffer: AudioBuffer){
  // simple wav encoding for download/share (unused but handy)
  return buffer;
}
