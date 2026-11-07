import { INSTRUMENTS, SCALER_STD, SCALER_MEAN } from "./dataset";

function cosineDist(a:number[], b:number[]){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  na=Math.sqrt(na); nb=Math.sqrt(nb);
  if(na===0||nb===0) return 1;
  return 1 - dot/(na*nb);
}

export function classify(vector:number[], k=3){
  const dists = INSTRUMENTS.map(ins=>{
    const d = cosineDist(vector, ins.centroid);
    return { ins, d };
  }).sort((a,b)=> a.d - b.d);

  const neighbors = dists.slice(0,k);
  // distance-weighted voting (w = 1/(d+eps))
  const eps=1e-6;
  const votes: Record<string, number> = {};
  neighbors.forEach(n=>{
    const w = 1/(n.d+eps);
    votes[n.ins.id]=(votes[n.ins.id]||0)+w;
  });
  // also compute scores for all classes via softmax over inverse distances
  const inv = dists.map(d=> 1/(d.d+eps));
  const exp = inv.map(v=> Math.exp(v - Math.max(...inv))); // stable
  // actually weights already; use votes normalized
  const total = Object.values(votes).reduce((a,b)=>a+b,0);
  // build full distribution: for each instrument, weight = 1/(dist+eps) normalized
  const allWeights = dists.map(d=> 1/(d.d+eps));
  const sumW = allWeights.reduce((a,b)=>a+b,0);
  const probs = dists.map((d,i)=> ({
    id: d.ins.id, name: d.ins.label, score: (allWeights[i]/sumW)*100, dist: d.d
  })).sort((a,b)=> b.score - a.score);

  const winnerId = Object.entries(votes).sort((a,b)=> b[1]-a[1])[0][0];
  const winner = INSTRUMENTS.find(i=> i.id===winnerId)!;
  const confidence = probs.find(p=> p.id===winnerId)!.score * (0.92 + Math.min(0.08, (1-neighbors[0].d)*0.1)); // small boost if very close

  const normalizedConf = Math.min(97.4, Math.max(38.2, confidence));

  return {
    instrument: winner.label,
    instrumentId: winner.id,
    confidence: normalizedConf,
    probabilities: probs,
    comparedVector: winner.centroid,
    neighbors: neighbors.map(n=> ({ label: n.ins.label, dist: n.d, weight: 1/(n.d+eps)})),
    winner,
  };
}

export function denormalizeForViz(normVec:number[]){
  // convert normalized vector back to ~readable range for table
  return normVec.map((v,i)=> v * SCALER_STD[i] + SCALER_MEAN[i]);
}
