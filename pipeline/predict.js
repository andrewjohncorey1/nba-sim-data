// Log the model's pre-game numbers for every slate game that has not started, and precompute the Pinnacle-anchored
// settings for games Pinnacle prices. Uses the same engine as the app.
// Usage (repo root): node pipeline/predict.js <stamp> <model-version>
// Writes data/predictions/<stamp>.csv (games), data/predictions/players_<stamp>.csv (props) and build/market.json
const fs = require('fs'), path = require('path');
const D = JSON.parse(fs.readFileSync('build/ratings.json', 'utf8'));
const SLATE = JSON.parse(fs.readFileSync('build/slate.json', 'utf8'));
const ODDS = fs.existsSync('build/odds.json') ? JSON.parse(fs.readFileSync('build/odds.json', 'utf8')) : {};
const E = require(path.resolve('pipeline/engine.js'));
const [stamp, ver] = process.argv.slice(2);
const N = 20000, now = Date.now(), G = D.engine;
function hashSeed(str) { let h = 2166136261; for (const c of str) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0) % 1000000 + 1; }
const csv = v => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
const f4 = v => (v === null || v === undefined || Number.isNaN(v)) ? '' : (+v).toFixed(4);
const f2 = v => (v === null || v === undefined || Number.isNaN(v)) ? '' : (+v).toFixed(2);
const half = x => Math.floor(x) + 0.5;

const cols = ['snap_utc', 'model_ver', 'game_id', 'start', 'home', 'away', 'neutral', 'out_h', 'out_a', 'avail_h', 'avail_a', 'b2b_h', 'b2b_a',
  // model only
  'p_home', 'exp_margin', 'exp_total', 'exp_home', 'exp_away', 'p_ot', 'sp_line', 'p_cover', 'tot_line', 'p_over',
  'h1_margin', 'h1_total', 'q1_total',
  // Pinnacle (no-vig) and the model's own price on Pinnacle's lines
  'pin_sp_line', 'pin_sp_home', 'model_sp_home', 'pin_tot_line', 'pin_over', 'model_over', 'pin_home',
  // anchored to Pinnacle (spread + total); moneyline from the anchored simulation is the independent check
  'anch_home', 'anch_exp_home', 'anch_exp_away', 'anch_h1_total', 'anch_q1_total', 'anch_miss', 'pin_snap'];
const scols = ['snap_utc', 'game_id', 'start', 'anchored', 'combo', 'side', 'pids', 'lines', 'legs', 'joint', 'legs_mult'], srow = [];
// fixed parlay shapes tested in validation/sgp6.py (A..K there)
function sgpCombos(W, PR, totLine) {
  const top = s => PR[s].filter(p => p.sg && p.line.pts != null).sort((x, y) => y.m.pts - x.m.pts);
  const T = { h: top('h'), a: top('a') }, out = [];
  const pl = (s, p) => ({ t: 'pts', side: s, pl: p, line: p.line.pts, pid: p.pid, name: p.name });
  for (const s of ['h', 'a']) {
    const o = s === 'h' ? 'a' : 'h', st = T[s][0], nd = T[s][1], op = T[o][0];
    if (!st) continue;
    const win = { t: 'win', side: s }, ov = { t: 'over', line: totLine };
    const shapes = [['star + team wins', [pl(s, st), win]], ['star + game over', [pl(s, st), ov]], ['star + team wins + game over', [pl(s, st), win, ov]]];
    if (nd) shapes.push(['star + teammate', [pl(s, st), pl(s, nd)]], ['star + teammate + team wins', [pl(s, st), pl(s, nd), win]],
      ['star + teammate + game over', [pl(s, st), pl(s, nd), ov]], ['star + teammate + team wins + game over', [pl(s, st), pl(s, nd), win, ov]]);
    if (op) { shapes.push(['star + opposing star + team wins', [pl(s, st), pl(o, op), win]]);
      if (s === 'h') shapes.push(['both stars', [pl(s, st), pl(o, op)]], ['both stars + game over', [pl(s, st), pl(o, op), ov]]); }
    for (const [k, legs] of shapes) { const r = E.sgpPrice(W, legs);
      out.push({ k, side: s, legs: legs.map(l => ({ t: l.t, side: l.side, pid: l.pid, name: l.name, line: l.line })), joint: +r.joint.toFixed(4), lm: +r.legs.toFixed(4) }); }
  }
  return out;
}
const pcols = ['snap_utc', 'game_id', 'start', 'team', 'opp', 'pid', 'name', 'exp_min', 'anchored', 'stat', 'mean', 'line', 'p_over'];
const rows = [], market = {}, prow = [];
const P = D.props || {};
for (const g of SLATE.games) {
  const ha = g.home_avail || {}, aa = g.away_avail || {};
  const sh = E.gameShifts(D, g, ha.out, aa.out, ha.extra, aa.extra);
  const T0 = E.matchupInputs(D, G, g.home, g.away, g.neutral === 1, sh.shH, sh.shA);
  const sa = E.selfAnchor(T0, G, hashSeed(g.id) + 7);          // match the ratings' own expected margin and total
  const T = E.shifted(T0, sa.s, sa.d);
  const od = ODDS[g.id];
  const seed = hashSeed(g.id) + 2;
  let mk = null;
  if (od && (od.sp_home_line != null || od.tot_line != null)) {
    mk = E.anchorToMarket(T, G, od.sp_home_line ?? null, od.sp_home_fair ?? null, od.tot_line ?? null, od.over_fair ?? null, hashSeed(g.id) + 5, 6000);
  }
  const H = E.simulateGame(T, G, N, seed, true);          // keepRows: every simulated score, for same-game parlays (no extra random numbers)
  const A = mk ? E.simulateGame(E.shifted(T, mk.s, mk.d), G, N, seed, true) : null;
  const fair = X => ({ sp: Math.round(-E.hmean(X.m) * 2) / 2, t: Math.round(E.expTotal(X) * 2) / 2 });
  market[g.id] = { fm: fair(H), pm: +E.mlHome(H).toFixed(4), ss: +sa.s.toFixed(5), sd: +sa.d.toFixed(5), shH: +sh.shH.toFixed(5), shA: +sh.shA.toFixed(5), ptsH: +sh.ptsH.toFixed(2), ptsA: +sh.ptsA.toFixed(2) };
  if (mk) Object.assign(market[g.id], { s: +mk.s.toFixed(5), d: +mk.d.toFixed(5), miss: mk.miss == null ? null : +mk.miss.toFixed(4),
    pa: +E.mlHome(A).toFixed(4), fa: fair(A) });
  if (new Date(g.start).getTime() <= now) continue;          // only genuinely pre-game numbers are logged
  const W = A || H;
  // player props from the simulation the app shows
  const PR = {};
  for (const [side, team, opp, av] of [['h', g.home, g.away, sh.h], ['a', g.away, g.home, sh.a]]) {
    const hist = { pts: W[side], fgm: W.stat[side].fgm, fg3m: W.stat[side].fg3m, reb: W.stat[side].reb, mb: W.mb[side] };
    PR[side] = E.playerProps(D, av.players, hist);
    for (const pr of PR[side]) {
      for (const k of Object.keys(pr.m)) prow.push([new Date(now).toISOString(), g.id, g.start, team, opp, pr.pid, pr.name, f2(pr.min), A ? 1 : 0,
        k, f2(pr.m[k]), pr.line[k], f4(pr.p[k])]);
    }
  }
  // same-game parlays at fixed shapes (Correlations tab; scored in the Track record): star = team's top projected
  // scorer among the props players, second = next; total at Pinnacle's line when anchored, else the fair line
  const SG = sgpCombos(W, PR, A && od && od.tot_line != null ? od.tot_line : Math.floor(E.expTotal(W)) + 0.5);
  market[g.id].sgp = SG;
  for (const c of SG) srow.push([new Date(now).toISOString(), g.id, g.start, A ? 1 : 0, c.k, c.side, c.legs.map(l => l.pid ?? '').join('|'),
    c.legs.map(l => l.line ?? '').join('|'), c.legs.map(l => l.t).join('|'), f4(c.joint), f4(c.lm)]);
  const spl = half(-E.hmean(H.m)), ttl = half(E.expTotal(H));
  rows.push([new Date(now).toISOString(), ver, g.id, g.start, g.home, g.away, g.neutral, (ha.out || []).length, (aa.out || []).length,
    f2(sh.ptsH), f2(sh.ptsA), g.rest ? g.rest.b2b_h : '', g.rest ? g.rest.b2b_a : '',
    f4(E.mlHome(H)), f2(E.hmean(H.m)), f2(E.expTotal(H)), f2(E.hmean(H.h)), f2(E.hmean(H.a)), f4(H.ot / H.n),
    spl, f4(E.coverNoPush(H, spl)), ttl, f4(E.overNoPush(H, ttl)),
    f2(E.hmean(H.seg.H1.m)), f2(E.hmean(H.seg.H1.t)), f2(E.hmean(H.seg.Q1.t)),
    od && od.sp_home_line != null ? od.sp_home_line : '', od && od.sp_home_fair != null ? f4(od.sp_home_fair) : '',
    od && od.sp_home_line != null ? f4(E.coverNoPush(H, od.sp_home_line)) : '',
    od && od.tot_line != null ? od.tot_line : '', od && od.over_fair != null ? f4(od.over_fair) : '',
    od && od.tot_line != null ? f4(E.overNoPush(H, od.tot_line)) : '', od && od.home_fair != null ? f4(od.home_fair) : '',
    A ? f4(E.mlHome(A)) : '', A ? f2(E.hmean(A.h)) : '', A ? f2(E.hmean(A.a)) : '', A ? f2(E.hmean(A.seg.H1.t)) : '', A ? f2(E.hmean(A.seg.Q1.t)) : '',
    mk && mk.miss != null ? f4(mk.miss) : '', od ? od.snap : '']);
}
fs.mkdirSync('data/predictions', { recursive: true });
if (srow.length) fs.writeFileSync(`data/predictions/sgp_${stamp}.csv`, [scols, ...srow].map(r => r.map(csv).join(',')).join('\n') + '\n');
if (rows.length) fs.writeFileSync(`data/predictions/${stamp}.csv`, [cols, ...rows].map(r => r.map(csv).join(',')).join('\n') + '\n');
if (prow.length) fs.writeFileSync(`data/predictions/players_${stamp}.csv`, [pcols, ...prow].map(r => r.map(csv).join(',')).join('\n') + '\n');
fs.writeFileSync('build/market.json', JSON.stringify(market));
const anch = Object.values(market).filter(v => v.s != null);
const bad = Object.entries(market).filter(([, v]) => v.miss != null && v.miss > 0.02);
const gap = anch.map(v => Math.abs(v.pa - (ODDS[Object.keys(market).find(k => market[k] === v)] || {}).home_fair)).filter(x => !Number.isNaN(x));
console.log(`PREDICTIONS ${rows.length} pre-game snapshots logged (${prow.length} player lines); ${anch.length} games anchored to Pinnacle` +
  (gap.length ? `; anchored moneyline vs Pinnacle's: mean gap ${(100 * gap.reduce((a, b) => a + b, 0) / gap.length).toFixed(1)} points, max ${(100 * Math.max(...gap)).toFixed(1)}` : ''));
if (bad.length) console.log(`NOTE ${bad.length} game(s) where the Pinnacle total could not be matched within 2 points: ${bad.map(b => b[0]).join(', ')}`);
