"""Score simulated held-out games. Usage: python3 validation/score_sims.py <season> <variant> [--lines]
Log loss (lower is better) for the moneyline, spreads and totals at fixed lines and, for 2025-26, at DraftKings'
closing lines (from ESPN's game pages); plus the game-shape checks. Baselines: earlier seasons' rates for each
outcome, and a simple point-differential model (season-to-date margin per game + home court, normal errors)."""
import sys, json, os
import numpy as np, pandas as pd
from scipy.stats import norm

Y, var = int(sys.argv[1]), sys.argv[2]
sims = {s['game_id']: s for s in json.load(open(f'validation/sim_{Y}_{var}.json'))}
g = pd.read_parquet(f'validation/games_{Y}_{var}.parquet')
g = g[g.game_id.isin(sims)].copy()
T = pd.read_parquet(f'data/derived/teamgames_{Y}.parquet')
g = g.merge(T[T.home == 1][['game_id', 'poss']], on='game_id', how='left')
G = pd.read_parquet(f'data/derived/games_{Y}.parquet')[['game_id', 'h1', 'h2', 'a1', 'a2']]
g = g.merge(G, on='game_id', how='left')
hist = pd.concat([pd.read_parquet(f'data/derived/games_{y}.parquet') for y in range(Y - 3, Y) if os.path.exists(f'data/derived/games_{y}.parquet')])
hist = hist[hist.done]
mh_h = hist.home_pts - hist.away_pts; th_h = hist.home_pts + hist.away_pts


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cdfget(arr, lo, L):          # P(X > L) for integer-CDF arrays starting at lo, L may be x.5
    k = int(np.floor(L)) - lo
    if k < 0: return 1.0
    if k >= len(arr): return 0.0
    return arr[k]


g['marg'] = g.home_pts - g.away_pts; g['tot'] = g.home_pts + g.away_pts; g['hw'] = (g.marg > 0).astype(int)
g['pm'] = g.game_id.map(lambda i: sims[i]['pm']); g['mm'] = g.game_id.map(lambda i: sims[i]['mm']); g['mt'] = g.game_id.map(lambda i: sims[i]['mt'])
out = {'season': Y, 'variant': var, 'n': len(g)}
# simple baseline: season-to-date point differential (shrunk) + home court
gg = g.sort_values('date'); diff = {}; cnt = {}; base_p = []; base_m = []
hca = float(mh_h.mean())
for r in gg.itertuples():
    rh = diff.get(r.home, 0) / (cnt.get(r.home, 0) + 5); ra = diff.get(r.away, 0) / (cnt.get(r.away, 0) + 5)
    m = rh - ra + (0 if r.neutral else hca); base_m.append(m); base_p.append(norm.cdf(m / 13.5))
    for t, d in ((r.home, r.marg), (r.away, -r.marg)): diff[t] = diff.get(t, 0) + d; cnt[t] = cnt.get(t, 0) + 1
gg['bp'] = base_p; gg['bm'] = base_m; g = gg
hw = float((mh_h > 0).mean())
out['ml'] = {'model': ll(g.hw, g.pm), 'home_rate': ll(g.hw, np.full(len(g), hw)), 'point_diff': ll(g.hw, g.bp)}
out['margin_rmse'] = {'model': float(np.sqrt(np.mean((g.marg - g.mm) ** 2))), 'point_diff': float(np.sqrt(np.mean((g.marg - g.bm) ** 2))),
                      'home_rate': float(np.sqrt(np.mean((g.marg - mh_h.mean()) ** 2)))}
out['total_rmse'] = {'model': float(np.sqrt(np.mean((g.tot - g.mt) ** 2))), 'league_avg': float(np.sqrt(np.mean((g.tot - th_h.mean()) ** 2)))}
# fixed lines
res = {}
for L in (-7.5, -3.5, 3.5, 7.5):
    y = (g.marg + L > 0).astype(int); p = g.game_id.map(lambda i: cdfget(sims[i]['mc'], -40, -L))
    res[f'home {L:+.1f}'] = {'model': ll(y, p), 'base': ll(y, np.full(len(g), float((mh_h + L > 0).mean())))}
for L in (215.5, 225.5, 235.5):
    y = (g.tot > L).astype(int); p = g.game_id.map(lambda i: cdfget(sims[i]['tc'], 180, L))
    res[f'total {L}'] = {'model': ll(y, p), 'base': ll(y, np.full(len(g), float((th_h > L).mean())))}
for side, col, key in (('home', 'home_pts', 'hc'), ('away', 'away_pts', 'ac')):
    L = 114.5; y = (g[col] > L).astype(int); p = g.game_id.map(lambda i: cdfget(sims[i][key], 90, L))
    hb = hist.home_pts if side == 'home' else hist.away_pts
    res[f'{side} team total {L}'] = {'model': ll(y, p), 'base': ll(y, np.full(len(g), float((hb > L).mean())))}
h1t = g.h1 + g.h2 + g.a1 + g.a2; h1h = hist.h1 + hist.h2 + hist.a1 + hist.a2
L = 113.5; y = (h1t > L).astype(int); p = g.game_id.map(lambda i: cdfget(sims[i]['h1c'], 95, L))
res[f'1st half total {L}'] = {'model': ll(y, p), 'base': ll(y, np.full(len(g), float((h1h > L).mean())))}
y = ((g.h1 + g.h2) > (g.a1 + g.a2)).astype(int); p = g.game_id.map(lambda i: cdfget(sims[i]['h1m'], -20, 0))
res['1st half home wins'] = {'model': ll(y, p), 'base': ll(y, np.full(len(g), float(((hist.h1 + hist.h2) > (hist.a1 + hist.a2)).mean())))}
out['fixed'] = res
out['shape'] = {'total_pred': float(g.mt.mean()), 'total_act': float(g.tot.mean()), 'poss_pred': float(np.mean([sims[i]['poss'] for i in g.game_id])),
                'poss_act': float(g.poss.mean()), 'ot_pred': float(np.mean([sims[i]['ot'] for i in g.game_id])), 'ot_act': float((g.ot > 0).mean()),
                'h1_pred': float(np.mean([sims[i]['h1']['t'] for i in g.game_id])), 'h1_act': float(h1t.mean()),
                'margin_sd_resid': float(np.std(g.marg - g.mm))}
# calibration of the moneyline
g['bin'] = pd.cut(g.pm, [0, .2, .35, .5, .65, .8, 1])
out['calib'] = [[str(b), int(len(x)), round(float(x.pm.mean()), 3), round(float(x.hw.mean()), 3)] for b, x in g.groupby('bin', observed=True)]
# DraftKings closing lines (2025-26 only)
import os
lf = f'data/market_hist/espn_lines_{Y}.csv'
if os.path.exists(lf):
    l = pd.read_csv(lf).dropna(subset=['spread']).drop_duplicates('game_id')
    d = g.merge(l, on='game_id')
    # spread is the home team's line (negative = home favoured)
    s = d[(d.marg + d.spread) != 0]
    def cover(i, L):
        x = -L; pg = cdfget(sims[i]['mc'], -40, x)
        if float(x).is_integer():
            push = cdfget(sims[i]['mc'], -40, x - 1) - pg
            return pg / max(1e-6, 1 - push)
        return pg
    y = (s.marg + s.spread > 0).astype(int); p = [cover(i, L) for i, L in zip(s.game_id, s.spread)]
    out['dk_spread'] = {'n': len(s), 'model': ll(y, p), 'dk_50_50': ll(y, np.full(len(s), .5)),
                        'model_rmse_vs_margin': float(np.sqrt(np.mean((s.marg - s.mm) ** 2))), 'dk_rmse': float(np.sqrt(np.mean((s.marg + s.spread) ** 2))),
                        'gap_rmse': float(np.sqrt(np.mean((s.mm + s.spread) ** 2)))}
    t = d[d.total.notna() & (d.tot != d.total)]
    y = (t.tot > t.total).astype(int); p = [cdfget(sims[i]['tc'], 180, L) for i, L in zip(t.game_id, t.total)]
    out['dk_total'] = {'n': len(t), 'model': ll(y, p), 'dk_50_50': ll(y, np.full(len(t), .5)), 'model_rmse': float(np.sqrt(np.mean((t.tot - t.mt) ** 2))),
                       'dk_rmse': float(np.sqrt(np.mean((t.tot - t.total) ** 2)))}
    m = d[d.ml_home.notna() & d.ml_away.notna()]
    imp = lambda a: np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
    ph, pa = imp(m.ml_home.values), imp(m.ml_away.values); pdk = ph / (ph + pa)
    out['dk_ml'] = {'n': len(m), 'model': ll(m.hw, m.pm), 'dk_novig': ll(m.hw, pdk), 'blend_50_50': ll(m.hw, (m.pm + pdk) / 2)}
print(json.dumps(out, indent=1))
json.dump(out, open(f'validation/score_{Y}_{var}.json', 'w'), indent=1)
