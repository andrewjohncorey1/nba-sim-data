// NBA possession engine, shared by the app (inlined at build time), pipeline/predict.js and validation.
// Plays a game possession by possession with a running clock. Each possession: turnover, or a free-throw trip, or a
// field-goal attempt (rim / mid-range / three, each with its own make rate, and-ones), offensive rebounds keep it
// alive. Possession length is drawn from measured lengths for how the previous possession ended, scaled to the
// matchup's pace. Score effects (by margin), the last 150 seconds of the 4th quarter and overtime (intentional fouls,
// running the clock out, hurried threes) and end-of-period heaves use measured league rates
// (pipeline/engine_params.py). Ties after 48 minutes go to 5-minute overtimes.
//
// T = {h: comps, a: comps, pace}  comps = {tov, trip, s3, s1, m1, m2, m3, oreb, ft} probabilities (pipeline/ratings.py)
//   pace = possessions per team per 48 minutes.
// G = engine_params.json (+ calibration: kappa scales every make probability's odds).

function mulberry32(seed){return function(){seed|=0;seed=(seed+0x6D2B79F5)|0;let t=Math.imul(seed^(seed>>>15),1|seed);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const lg=p=>Math.log(p/(1-p)), ex=x=>1/(1+Math.exp(-x));
const SEG=['Q1','Q2','Q3','Q4','H1','H2'];

function newHist(){const seg={};for(const s of SEG)seg[s]={m:{},t:{},h:{},a:{}};
  return {n:0,m:{},t:{},h:{},a:{},seg,ot:0,ot2:0,regTie:0,
    stat:{h:{fgm:{},fg3m:{},reb:{},poss:0},a:{fgm:{},fg3m:{},reb:{},poss:0}},sum:{h:{fgm:0,fg3m:0,reb:0},a:{fgm:0,fg3m:0,reb:0}}};}
const inc=(o,k)=>{o[k]=(o[k]||0)+1;};
function bin(edges,v){for(let i=1;i<edges.length;i++)if(v<=edges[i])return i-1;return edges.length-2;}

// Precompute per-side logits so the inner loop is cheap.
function prepSide(c,G){const k=G.kappa||1;return {tov:c.tov,trip:c.trip,ls3:lg(c.s3),s1:c.s1,lm:[lg(c.m1)+Math.log(k),lg(c.m2)+Math.log(k),lg(c.m3)+Math.log(k)],oreb:c.oreb,ft:c.ft};}

// One possession for side S (prepared). shift: logit shift on makes; d3: logit shift on three share.
// Returns [points, end ('make'|'miss'|'tov'), fgm, fg3m, missesForRebounds, offensiveRebounds]
function playPoss(S,G,r,shift,d3){
  let pts=0,fgm=0,f3m=0,miss=0,orb=0;
  const s3=ex(S.ls3+d3);
  for(let k=0;k<8;k++){
    if(r()<S.tov)return [pts,'tov',fgm,f3m,miss,orb];
    if(r()<S.trip){
      const n=r()<G.trip3?3:2;let last=false;
      for(let i=0;i<n;i++){last=r()<S.ft;if(last)pts++;}
      if(last)return [pts,'make',fgm,f3m,miss,orb];
      miss++;
      if(r()<G.oreb_ft){orb++;continue;}
      return [pts,'miss',fgm,f3m,miss,orb];
    }
    const z=r()<s3?2:(r()<S.s1?0:1);
    if(r()<ex(S.lm[z]+shift)){
      pts+=z===2?3:2;fgm++;if(z===2)f3m++;
      if(r()<G.and1[z]&&r()<S.ft)pts++;
      return [pts,'make',fgm,f3m,miss,orb];
    }
    miss++;
    if(r()<S.oreb){orb++;continue;}
    return [pts,'miss',fgm,f3m,miss,orb];
  }
  return [pts,'miss',fgm,f3m,miss,orb];
}

function simulateGame(T,G,N,seed,keepRows){
  const r=mulberry32(seed),H=newHist();
  // keepRows: also save every simulated final score as H.rows = [home, away, home, away, ...] (for joint prices such
  // as same-game parlays). Draws no extra random numbers, so every other price is unchanged.
  if(keepRows)H.rows=new Int16Array(2*N);
  const side=[prepSide(T.h,G),prepSide(T.a,G)];
  const durMul=(G.durK||1)*(2880/(2*T.pace))/G.base_dur;      // scale measured lengths to this matchup's pace (durK: calibration)
  const D=G.dur,SC=G.score,LT=G.late,HV=G.heave,base3=lg(G.base_f3);
  const scF3=SC.f3.map(f=>lg(f)-base3);
  const ltF3=LT.f3.map(row=>row.map(f=>lg(f)-base3));
  const qd=k=>{const q=D[k];return q[Math.floor(r()*q.length)];};
  for(let g=0;g<N;g++){
    const sc=[0,0],seg=[[0,0],[0,0],[0,0],[0,0],[0,0]];   // Q1..Q4 and OT points per side
    let fgm=[0,0],f3m=[0,0],reb=[0,0],np=[0,0];
    const tip=r()<0.5?0:1;let dbg=null;
    let per=1,ot=0;
    while(true){
      const plen=per<=4?720:300;
      let clock=plen;
      let off=per===1||per===4?tip:(per<=3?1-tip:(r()<0.5?0:1));
      let prev='start';
      const final=per>=4,ps=(G.h1s||0)*(per<=2?1:-1);      // calibration: scoring split between the halves
      while(clock>0){
        const o=off,d=1-off,S=side[o],m=sc[o]-sc[d];
        if(G.debug&&per===4&&dbg===null&&clock<=150)dbg=Math.abs(sc[0]-sc[1]);
        let pts=0,end='miss',used,res=null;
        if(clock<=3){
          if(r()<HV.p_score)pts=HV.pts_if>2.5?3:(r()<HV.pts_if-2?3:2);
          used=clock;end=pts?'make':'miss';
        }else if(final&&clock<=150){
          const tb=bin(LT.tedges,clock),mb=bin(LT.medges,m),u=r();
          if(u<LT.qf[tb][mb]){
            let last=false;for(let i=0;i<2;i++){last=r()<S.ft;if(last)pts++;}
            end=last?'make':'miss';used=Math.min(clock,LT.qf_dur*(0.5+r()));
            if(!last){if(r()<G.oreb_ft){used=Math.min(clock,used+2);} }
          }else if(r()<LT.kill[tb][mb]){
            used=Math.min(clock,24);end='miss';
          }else{
            res=playPoss(S,G,r,LT.shift[tb][mb]+ps,ltF3[tb][mb]);
            used=Math.min(clock,qd(prev)*LT.dur[tb][mb]/G.base_dur*durMul);
            // tied or down 1-2 inside the last shot clock: the offense plays for the last shot
            if(clock<=24&&m<=0&&m>=-2&&r()<(G.hold??0.8))used=Math.max(0.5,clock-2.5*r());
          }
        }else{
          const b=bin(SC.edges,m);
          res=playPoss(S,G,r,SC.shift[b]+ps,scF3[b]);
          used=Math.min(clock,qd(prev)*SC.dur[b]*durMul);
        }
        if(res){pts=res[0];end=res[1];fgm[o]+=res[2];f3m[o]+=res[3];reb[o]+=res[5];if(end==='miss')reb[d]++;}
        sc[o]+=pts;seg[Math.min(per,5)-1][o]+=pts;np[o]++;
        clock-=Math.max(used,0.5);prev=end;off=d;
      }
      if(per===4){if(r()<G.tech_pts/2){const k=r()<0.5?0:1;sc[k]++;seg[3][k]++;}if(r()<G.tech_pts/2){const k=r()<0.5?0:1;sc[k]++;seg[3][k]++;}}
      if(per>=4&&sc[0]!==sc[1]){if(per===4&&G.debug)(H.dbg=H.dbg||[]).push([dbg,0]);break;}
      if(per===4){H.regTie++;}
      if(per===4&&G.debug)(H.dbg=H.dbg||[]).push([dbg,1]);
      if(per>=5)ot++;
      per++;
      if(per>8){sc[r()<0.5?0:1]++;break;}
    }
    H.n++;
    if(per>=5){H.ot++;if(per>=6)H.ot2++;}
    if(H.rows){H.rows[2*g]=sc[0];H.rows[2*g+1]=sc[1];}
    inc(H.m,sc[0]-sc[1]);inc(H.t,sc[0]+sc[1]);inc(H.h,sc[0]);inc(H.a,sc[1]);
    const segs=[[seg[0][0],seg[0][1]],[seg[1][0],seg[1][1]],[seg[2][0],seg[2][1]],[seg[3][0],seg[3][1]],
      [seg[0][0]+seg[1][0],seg[0][1]+seg[1][1]],[seg[2][0]+seg[3][0]+seg[4][0],seg[2][1]+seg[3][1]+seg[4][1]]];
    for(let i=0;i<6;i++){const s=H.seg[SEG[i]],[x,y]=segs[i];inc(s.m,x-y);inc(s.t,x+y);inc(s.h,x);inc(s.a,y);}
    for(const [k,i] of [['h',0],['a',1]]){inc(H.stat[k].fgm,fgm[i]);inc(H.stat[k].fg3m,f3m[i]);inc(H.stat[k].reb,reb[i]);
      H.stat[k].poss+=np[i];H.sum[k].fgm+=fgm[i];H.sum[k].fg3m+=f3m[i];H.sum[k].reb+=reb[i];}
  }
  return H;
}

/* ---------- reading the simulation ---------- */
const hmean=h=>{let s=0,n=0;for(const k in h){s+=k*h[k];n+=h[k];}return n?s/n:0;};
const pGT=(h,L)=>{let o=0,n=0;for(const k in h){n+=h[k];if(+k>L)o+=h[k];}return n?o/n:0;};
const pEQ=(h,L)=>{let n=0;for(const k in h)n+=h[k];return n?(h[L]||0)/n:0;};
function mlHome(H){return pGT(H.m,0);}           // OT included: no ties at the end of a game
function coverHome(H,line){return pGT(H.m,-line);}  // home -line covers when margin > line (pushes counted as not covering)
function pushHome(H,line){return pEQ(H.m,-line);}
function overP(H,L){return pGT(H.t,L);}
function expTotal(H){return hmean(H.t);}

/* ---------- ratings -> engine inputs (mirror of pipeline/ratings.py probs()) ---------- */
// D = ratings.json. shiftH/shiftA: logit shifts on each offense's make rates (player availability). neutral: no home court.
function compsFor(D,off,dfn,home,shift){
  const i=D.teams.indexOf(off),j=D.teams.indexOf(dfn),c={};
  for(const k of ['tov','trip','s3','s1','m1','m2','m3','oreb','ft']){
    const R=D.comp[k];let v=R.mu+R.h*home+R.O[i]-R.D[j];
    if(k[0]==='m'&&k.length===2)v+=shift;
    c[k]=ex(v);
  }
  return c;
}
function paceFor(D,home,away){const P=D.pace;return Math.exp(P.mu+P.P[D.teams.indexOf(home)]+P.P[D.teams.indexOf(away)]);}
function expPPP(c,G){
  const sh=[(1-c.s3)*c.s1,(1-c.s3)*(1-c.s1),c.s3],m=[c.m1,c.m2,c.m3],a1=G.and1,ft=c.ft;
  let fp=0,fc=0;for(let z=0;z<3;z++){fp+=sh[z]*m[z]*((z<2?2:3)+a1[z]*ft);fc+=sh[z]*(1-m[z]);}
  fc*=c.oreb;
  const e=(1-c.tov)*(c.trip*(2+G.trip3)*ft+(1-c.trip)*fp),cont=(1-c.tov)*(c.trip*(1-ft)*G.oreb_ft+(1-c.trip)*fc);
  return e/(1-cont);
}
// availability: points per 100 change -> logit shift on makes
function availShift(pts100,ppp0,sens){return Math.log(Math.max(0.5,1+pts100/100/ppp0))/sens;}
function matchupInputs(D,G,home,away,neutral,shH,shA){
  const hm=neutral?0:1;
  return {h:compsFor(D,home,away,hm,shH||0),a:compsFor(D,away,home,0,shA||0),pace:paceFor(D,home,away)};
}


/* ---------- player availability (pipeline/players.py availability_delta) ---------- */
// Expected minutes of each available player = projected minutes when he plays x chance he gets in (pp), fitted to
// the team's 240 minutes by allocate(). The team's change in offense/defense is the
// sum over players of (tonight's share - the share behind the team rating) x his impact.
// Fit expected minutes to 240: the gap is spread in proportion to m x (1 - m/48)^3 (players.py allocate).
function allocate(m,total,cap){total=total||240;cap=cap||48;m=m.slice();
  for(let k=0;k<6;k++){const gap=total-m.reduce((a,b)=>a+b,0);if(Math.abs(gap)<0.01)break;
    const w=m.map(v=>v*Math.pow(Math.max(0.05,1-v/cap),3)),sw=w.reduce((a,b)=>a+b,0)||1;m=m.map((v,i)=>Math.min(cap,Math.max(0,v+gap*w[i]/sw)));}
  return m;}
function teamAvail(D,team,outIds,extraIds){
  const R=D.rosters.find(t=>t.abbr===team),out=new Set(outIds||[]),rep=D.avail.repl;
  let pl=R.roster.filter(p=>!out.has(p.pid));
  for(const id of extraIds||[]){if(!pl.some(p=>p.pid===id)&&D.pool[id])pl.push({...D.pool[id],pid:id});}
  const sh=allocate(pl.map(p=>p.pm*(p.pp??1))).map(v=>v/48);
  const sb=new Map(R.sbar.map(r=>[r[0],r]));
  let dO=0,dD=0;const seen=new Set();
  pl.forEach((p,i)=>{const b=sb.get(p.pid);const s0=b?b[1]:0;dO+=(sh[i]-s0)*p.O;dD+=(sh[i]-s0)*p.D;seen.add(p.pid);});
  for(const [id,b] of sb)if(!seen.has(id)){dO-=b[1]*b[2];dD-=b[1]*b[3];}
  return {dO,dD,players:pl.map((p,i)=>({...p,share:sh[i],min:sh[i]*48}))};
}
// Logit shifts on each offense's make rates from availability and rest (rest effects in points per 100 possessions).
function gameShifts(D,g,outH,outA,extraH,extraA){
  const h=teamAvail(D,g.home,outH,extraH),a=teamAvail(D,g.away,outA,extraA),A=D.avail,al=A.alpha,re=D.rest||{};
  let ph=al*(h.dO-a.dD),pa=al*(a.dO-h.dD);
  if(g.rest&&re.own_b2b!=null){ph+=re.own_b2b*g.rest.b2b_h+re.dfn_b2b*g.rest.b2b_a;pa+=re.own_b2b*g.rest.b2b_a+re.dfn_b2b*g.rest.b2b_h;}
  return {shH:availShift(ph,A.ppp0,A.sens),shA:availShift(pa,A.ppp0,A.sens),ptsH:ph,ptsA:pa,h,a};
}


/* ---------- matching the ratings' own expected points ---------- */
// The team ratings are averages over real games, which already include how teams play when ahead or behind; the
// simulation's score effects would count that twice and pull favourites' margins in. So, before Pinnacle anchoring,
// both offenses' make rates are shifted (common s, split d) until the simulated mean margin and mean total equal the
// ratings' closed-form expectation (expPPP x pace, minus the measured total offset G.tot_off). The score effects
// still shape the distribution (comebacks, garbage time, late fouling).
function closedForm(T,G){const ph=expPPP(T.h,G)*T.pace,pa=expPPP(T.a,G)*T.pace;return {m:ph-pa,t:ph+pa-(G.tot_off||0)};}
function selfAnchor(T,G,seed,N){
  N=N||2500;const tg=closedForm(T,G);let s=0,d=0,km=88,kt=88,pm=null,pt=null,ps=null,pd=null;
  for(let it=0;it<4;it++){
    const H=simulateGame(shifted(T,s,d),G,N,seed),em=hmean(H.m),et=hmean(H.t);
    if(pm!=null&&Math.abs(d-pd)>1e-4)km=Math.max(40,Math.min(150,(em-pm)/(d-pd)));
    if(pt!=null&&Math.abs(s-ps)>1e-4)kt=Math.max(40,Math.min(150,(et-pt)/(s-ps)));
    pm=em;pt=et;pd=d;ps=s;
    d+=(tg.m-em)/km;s+=(tg.t-et)/kt;
  }
  return {s,d,target:tg};
}

/* ---------- Pinnacle anchoring ---------- */
// Shift both offenses' make odds by a common logit amount s (total) and a split d (spread) until the simulation matches
// Pinnacle's no-vig cover probability at its spread and over probability at its total. Moneyline is not used.
function shifted(T,s,d){const f=(c,x)=>({...c,m1:ex(lg(c.m1)+x),m2:ex(lg(c.m2)+x),m3:ex(lg(c.m3)+x)});
  return {h:f(T.h,s+d),a:f(T.a,s-d),pace:T.pace};}
function coverNoPush(H,line){const p=pushHome(H,line);return p<1?coverHome(H,line)/(1-p):0.5;}
function overNoPush(H,L){const p=pEQ(H.t,L);return p<1?overP(H,L)/(1-p):0.5;}
function anchorToMarket(T,G,spLine,pCover,totLine,pOver,seed,N){
  // secant steps on the common shift s (total) and the split d (spread); each probability moves mostly with its own
  // shift, so the two are solved side by side, using the same random numbers every time
  N=N||6000;
  const hasS=spLine!=null&&pCover!=null,hasT=totLine!=null&&pOver!=null;
  let s=0,d=0,kc=2.5,ko=2.5,pc=null,po=null,ps=null,pd=null,best=null;
  for(let it=0;it<8;it++){
    const H=simulateGame(shifted(T,s,d),G,N,seed);
    const c=hasS?coverNoPush(H,spLine):null,o=hasT?overNoPush(H,totLine):null;
    const err=Math.max(hasS?Math.abs(c-pCover):0,hasT?Math.abs(o-pOver):0);
    if(!best||err<best.err)best={s,d,err,po:o,pc:c};
    if(err<0.002)break;
    if(pc!=null&&hasS&&Math.abs(d-pd)>1e-4)kc=Math.max(0.5,Math.min(8,(c-pc)/(d-pd)));
    if(po!=null&&hasT&&Math.abs(s-ps)>1e-4)ko=Math.max(0.5,Math.min(8,(o-po)/(s-ps)));
    pc=c;po=o;pd=d;ps=s;
    if(hasS)d+=Math.max(-0.15,Math.min(0.15,(pCover-c)/kc));
    if(hasT)s+=Math.max(-0.15,Math.min(0.15,(pOver-o)/ko));
  }
  return {s:best.s,d:best.d,miss:hasT?Math.abs(best.po-pOver):null,missSpread:hasS?Math.abs(best.pc-pCover):null,spreadOnly:!hasT,totalOnly:!hasS};
}

/* ---------- player props ---------- */
// A player's stat in a game = his share of the team quantity (points, field-goal makes for assists, threes made,
// rebound chances) x the team quantity in that simulated game, with negative-binomial noise around it. The team
// quantity's distribution comes from the same simulation (mixture over simulated games).
function nbPmf(m,phi,kmax){const p=phi/(phi+m),out=new Float64Array(kmax+1);let pr=Math.pow(p,phi);out[0]=pr;
  for(let i=0;i<kmax;i++){pr*=(i+phi)/(i+1)*(1-p);out[i+1]=pr;}return out;}
function propOver(teamHist,share,phi,line){
  let n=0,o=0;
  for(const k in teamHist){const w=teamHist[k],m=Math.max(1e-6,share*(+k));n+=w;
    const pm=nbPmf(m,phi,Math.floor(line));let c=0;for(const v of pm)c+=v;o+=w*(1-c);}
  return n?o/n:0;
}
function propMean(teamHist,share){return share*hmean(teamHist);}


// Props for the available players of one team. players: teamAvail(...).players (share = expected share of the 5
// floor spots incl. the chance he gets in; pp = that chance). hist: the team's simulated {pts, fgm, fg3m, reb}.
// A player's mean, given he plays = per-minute rate x minutes when he plays x (simulated team quantity / the team's
// sum of player expectations). Assists scale with the team's made field goals; rebounds with the team's simulated
// rebounds relative to the league reference. Lines: the mean rounded down + 0.5, plus any requested lines.
function playerProps(D,players,hist,minMin){
  const P=D.props||{},mk=P.markets||['pts'],phi=P.phi||{},disp=P.disp||{},out=[];
  // dispersion per market: one value for everyone, or proportional to the player's mean ('prop': phi = c x mean)
  // or 'pow': phi = f[0] x (mean/10)^f[1] (validation/tune_props2.py tune3)
  const phiOf=(k,m)=>{const d=disp[k];if(!d)return phi[k]||10;
    return d.mode==='prop'?Math.max(0.3,d.c*m):d.mode==='pow'?Math.max(0.3,d.f[0]*Math.pow(Math.max(m,1e-3)/10,d.f[1])):d.phi;};
  const tm=P.tmult||1;      // calibration of the simulated team points for the points-based props
  const R={pts:0,fgm:0,fg3m:0,ast:0,reb:0};
  // prop minutes: the props' own projection (pmp, tuned separately) when present, fitted to 240 like the team's
  const ems=players.some(p=>p.pmp!=null)?allocate(players.map(p=>(p.pmp??p.pm)*(p.pp??1))):players.map(p=>p.share*48);
  players.forEach((p,i)=>{for(const k in R)R[k]+=(p.r?p.r[k]:0)*ems[i];});
  for(const [i,p] of players.entries()){
    const em=ems[i],cm=Math.min(48,em/Math.max(0.05,p.pp??1));
    if(cm<(minMin??8)||!p.r)continue;
    const sh={pts:tm*p.r.pts*cm/R.pts,fg3m:p.r.fg3m*cm/R.fg3m,ast:p.r.ast*cm/R.fgm,reb:p.r.reb*cm/(P.reb_ref||R.reb)};
    const H={pts:hist.pts,fg3m:hist.fg3m,ast:hist.fgm,reb:hist.reb};
    const m={},line={},pr={};
    for(const k of mk){
      if(k==='pra'){const mt=hmean(hist.pts),mm=sh.pts*mt+sh.reb*hmean(hist.reb)+sh.ast*hmean(hist.fgm);
        m.pra=mm;line.pra=Math.floor(mm)+0.5;pr.pra=propOver(hist.pts,mm/mt,phiOf('pra',mm),line.pra);continue;}
      m[k]=sh[k]*hmean(H[k]);line[k]=Math.max(0.5,Math.floor(m[k])+0.5);pr[k]=propOver(H[k],sh[k],phiOf(k,m[k]),line[k]);
    }
    out.push({pid:p.pid,name:p.name,pos:p.pos,min:cm,pp:p.pp??1,sh,m,line,p:pr});
  }
  return out;
}

if (typeof module !== 'undefined') module.exports = { mulberry32, simulateGame, playPoss, mlHome, coverHome, pushHome, overP, expTotal,
  hmean, pGT, pEQ, compsFor, paceFor, expPPP, availShift, matchupInputs, teamAvail, gameShifts, shifted, closedForm, selfAnchor, anchorToMarket, coverNoPush, overNoPush, nbPmf, propOver, propMean, playerProps, SEG };
