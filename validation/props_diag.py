"""Where does the points-prop compression come from? Decomposes the model's mean on a sample of dates (every 6th)
of a season into: per-minute rate (raw vs shrunk toward position average), minutes (projected + fitted to 240 vs
actual) and the rescaling to the simulated team points. Usage: python3 validation/props_diag.py <season>"""
import sys, json
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import players as PS
Y = int(sys.argv[1]); HL, PRIOR = 180, 300
sims = {s['game_id']: s for s in json.load(open(f'validation/sim_{Y}_base.json'))}
PL = PS.load(range(Y - 2, Y + 1))
PL['posg'] = PL.pos.fillna('F').str[0].map({'G': 'G', 'F': 'F', 'C': 'C'}).fillna('F')
cur = PL[(PL.season == Y) & (PL['min'] > 0) & PL.game_id.isin(sims)]
g = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id')
dates = sorted(cur.date.unique())[::6]
rows = []
for d in dates:
    gd = cur[cur.date == d]; hist = PL[PL.date < d]; x = hist[hist['min'] > 0]
    w = PS.weights(x.date, d, HL); x = x.assign(w=w)[w > 1e-3]
    A = x.assign(wm=x.w * x['min'], wp=x.w * x.pts).groupby('pid')[['wm', 'wp']].sum()
    pa = {k: float((v.w * v.pts).sum() / (v.w * v['min']).sum()) for k, v in x.groupby('posg')}
    pos = x.sort_values('date').groupby('pid').posg.last()
    pm = PS.projected_minutes(PL, d)
    sh = hist[(hist.season == Y) & (hist['min'] > 0)].groupby('pid').pts.agg(['mean', 'count'])
    for (gid, team), tp in gd.groupby(['game_id', 'team']):
        side = 'h' if g.loc[gid, 'home'] == team else 'a'
        v = np.asarray(sims[gid]['p' + side]); ET = float(np.dot(np.arange(60, 60 + len(v)), v))
        proj = pm.proj.reindex(tp.pid).fillna(12.0).values; alloc = PS.allocate(proj)
        raw = np.array([A.loc[p, 'wp'] / A.loc[p, 'wm'] if p in A.index and A.loc[p, 'wm'] > 0 else pa.get(pos.get(p, 'F'), pa['F']) for p in tp.pid])
        shr = np.array([(A.loc[p, 'wp'] + PRIOR * pa.get(pos.get(p, 'F'), pa['F'])) / (A.loc[p, 'wm'] + PRIOR) if p in A.index
                        else pa.get(pos.get(p, 'F'), pa['F']) for p in tp.pid])
        tot = (shr * alloc).sum()
        for i, (p, act) in enumerate(zip(tp.pid, tp.itertuples())):
            if p not in sh.index or sh.loc[p, 'count'] < 5: continue
            rows.append(dict(base=sh.loc[p, 'mean'], actual=act.pts, min_act=act.min, min_proj=proj[i], min_alloc=alloc[i],
                             rate_raw=raw[i], rate_shr=shr[i], rate_act=act.pts / act.min,
                             m_raw_actmin=raw[i] * act.min, m_shr_actmin=shr[i] * act.min, m_shr_alloc=shr[i] * alloc[i],
                             final=shr[i] * alloc[i] / tot * ET, n_played=len(tp), sum_proj=proj.sum()))
D = pd.DataFrame(rows); D['tier'] = pd.cut(D.base, [0, 5, 10, 15, 20, 25, 40])
cols = ['base', 'actual', 'm_raw_actmin', 'm_shr_actmin', 'm_shr_alloc', 'final', 'min_act', 'min_proj', 'min_alloc', 'rate_raw', 'rate_shr', 'rate_act']
print(f'{Y}: {len(D)} player-games on {len(dates)} dates; players who played per team {D.n_played.mean():.1f}, '
      f'sum of their projected minutes {D.sum_proj.mean():.0f} (fitted to 240)')
print(D.groupby('tier', observed=True)[cols].mean().round(2).to_string())
