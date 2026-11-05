// Simulate held-out games from exported inputs: node validation/sim_games.js <inputs.json> <out.json> [N]
// inputs: [{game_id, h:comps, a:comps, pace}], uses pipeline/engine_params.json (+ calibration in validation/calib.json if present)
const fs=require('fs'),E=require('../pipeline/engine.js');
const G=JSON.parse(fs.readFileSync('pipeline/engine_params.json','utf8'));
const cal=fs.existsSync('validation/calib.json')?JSON.parse(fs.readFileSync('validation/calib.json','utf8')):{};
Object.assign(G,cal,JSON.parse(process.env.GOVR||'{}'));
const [inp,outp,Ns]=process.argv.slice(2);const N=+(Ns||4000);
const X=JSON.parse(fs.readFileSync(inp,'utf8'));
const cdf=(h,lo,hi)=>{let n=0;for(const k in h)n+=h[k];const o=[];for(let L=lo;L<=hi;L++){let g=0;for(const k in h)if(+k>L)g+=h[k];o.push(+(g/n).toFixed(4));}return o;};
const pmf=(h,lo,hi)=>{let n=0;for(const k in h)n+=h[k];const o=[];for(let L=lo;L<=hi;L++)o.push(+(((h[L]||0)+(L===lo?Object.keys(h).filter(k=>+k<lo).reduce((a,k)=>a+h[k],0):0)+(L===hi?Object.keys(h).filter(k=>+k>hi).reduce((a,k)=>a+h[k],0):0))/n).toFixed(5));return o;};
const props=process.env.PROPS==='1';
// ROWS=<path>: also write every simulated final score (Int16 home, away per sim, games in input order) for joint tests
const ROWS=process.env.ROWS||'',rows=ROWS?new Int16Array(X.length*2*N):null;
const out=X.map((x,i)=>{let xx=x;if(G.self!==0){const a=E.selfAnchor(x,G,500+i);xx=E.shifted(x,a.s,a.d);}const H=E.simulateGame(xx,G,N,1000+i,!!ROWS);if(ROWS)rows.set(H.rows,i*2*N);
  const sg=s=>({m:+E.hmean(H.seg[s].m).toFixed(3),t:+E.hmean(H.seg[s].t).toFixed(3)});
  return {game_id:x.game_id,pm:+E.mlHome(H).toFixed(4),mm:+E.hmean(H.m).toFixed(3),mt:+E.hmean(H.t).toFixed(3),
   mh:+E.hmean(H.h).toFixed(3),ma:+E.hmean(H.a).toFixed(3),ot:+(H.ot/H.n).toFixed(4),
   mc:cdf(H.m,-40,40),tc:cdf(H.t,180,280),hc:cdf(H.h,90,140),ac:cdf(H.a,90,140),
   q1:sg('Q1'),h1:sg('H1'),h1c:cdf(H.seg.H1.t,95,135),h1m:cdf(H.seg.H1.m,-20,20),q1c:cdf(H.seg.Q1.t,45,70),q1m:cdf(H.seg.Q1.m,-12,12),
   poss:+((H.stat.h.poss+H.stat.a.poss)/2/H.n).toFixed(2),
   ...(props?{ph:pmf(H.h,60,170),pa:pmf(H.a,60,170),fgh:pmf(H.stat.h.fgm,20,65),fga:pmf(H.stat.a.fgm,20,65),
     f3h:pmf(H.stat.h.fg3m,0,32),f3a:pmf(H.stat.a.fg3m,0,32),rbh:pmf(H.stat.h.reb,20,80),rba:pmf(H.stat.a.reb,20,80)}:{})};});
fs.writeFileSync(outp,JSON.stringify(out));
if(ROWS){fs.writeFileSync(ROWS,Buffer.from(rows.buffer));fs.writeFileSync(ROWS+'.json',JSON.stringify({N,game_ids:X.map(x=>x.game_id)}));}
console.log('simulated',out.length,'games x',N);
