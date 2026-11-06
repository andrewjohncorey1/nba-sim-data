"""
Game-level candidate adjustments, each tested on the held-out simulations (validation/rows_<Y>_env.bin: model only,
shared pace factor, +1 point per team for the current totals offset).

Usage (repo root): python3 validation/game_adj.py <season>      (2025 fits / picks; 2026 tests what 2025 picked)

Each candidate moves every simulated game's margin (home +d/2, away -d/2) and/or total (both +t/2). Prices at the
base simulation's fair lines: moneyline, spread at the fair line, total at the fair line, alt totals +-8.
  sched    schedule spots beyond back-to-backs (already in the model): 3 games in 4 nights, 4 in 6, 5th+ straight
           road game, West team in an Eastern-time game starting before 6 PM ET; coefficients fit on 2024-25
           residuals (ridge), applied unchanged to 2025-26
  homeT    team-specific home court: each team's home-minus-road margin over the two previous seasons, minus the
           league's, shrunk (K games), times a weight picked on 2024-25
  teamres  rating lag / roster change: each team's model margin residual in its earlier games this season (home +,
           away -), shrunk with K games, times a weight; also reported for each team's first 20 games
  totroll  rolling totals offset: the mean total residual of the league's previous N games this season replaces the
           fixed offset (shrunk toward it with K games)
"""
import sys, json
import numpy as np, pandas as pd

Y = int(sys.argv[1])
FIT = json.load(open('validation/game_adj_fit_2025.json')) if Y != 2025 else None
meta = json.load(open(f'validation/rows_{Y}_env.bin.json'))
R = np.fromfile(f'validation/rows_{Y}_env.bin', dtype=np.int16).reshape(len(meta['game_ids']), meta['N'], 2).astype(float) + 1.0
G = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id').loc[meta['game_ids']].reset_index()
H, A = G.home_pts.values, G.away_pts.values
mh, ma = R[:, :, 0].mean(1), R[:, :, 1].mean(1); res_m = (H - A) - (mh - ma); res_t = (H + A) - (mh + ma)
order = np.argsort(pd.to_datetime(G.start).values, kind='stable')
rng = np.random.default_rng(1)
WEST = {'GSW', 'LAL', 'LAC', 'SAC', 'POR', 'PHX', 'UTA', 'DEN'}
EAST_STATES = {'GA', 'NY', 'MA', 'NC', 'OH', 'MI', 'IN', 'FL', 'PA', 'ON', 'DC'}

# ---------- features ----------
allg = pd.concat([pd.read_parquet(f'data/derived/games_{y}.parquet') for y in (Y - 2, Y - 1, Y)], ignore_index=True)
allg['date'] = pd.to_datetime(allg.date)
long = pd.concat([allg[['game_id', 'date', 'home']].rename(columns={'home': 'team'}).assign(road=0),
                  allg[['game_id', 'date', 'away']].rename(columns={'away': 'team'}).assign(road=1)]).sort_values(['team', 'date'])
feat = {}
for t, x in long.groupby('team'):
    d = x.date.values; rd = x.road.values; streak = 0
    for i, gid in enumerate(x.game_id.values):
        n4 = ((d[:i + 1] >= d[i] - np.timedelta64(3, 'D'))).sum(); n6 = ((d[:i + 1] >= d[i] - np.timedelta64(5, 'D'))).sum()
        streak = streak + 1 if rd[i] else 0
        feat[(gid, t)] = (int(n4 >= 3), int(n6 >= 4), int(streak >= 5))
start_et = pd.to_datetime(G.start).dt.tz_convert('America/New_York')
early_east = ((start_et.dt.hour < 18) & G.state.isin(EAST_STATES)).values


def team_feats(i, side):
    t = G.home[i] if side == 'h' else G.away[i]
    f = feat.get((G.game_id[i], t), (0, 0, 0))
    return list(f) + [int(early_east[i] and t in WEST and side == 'a')]


Xs = np.array([np.array(team_feats(i, 'h')) - np.array(team_feats(i, 'a')) for i in range(len(G))], dtype=float)
Xt = np.array([np.array(team_feats(i, 'h'))[:3] + np.array(team_feats(i, 'a'))[:3] for i in range(len(G))], dtype=float)
SN = ['3 in 4', '4 in 6', '5th+ straight road', 'West early East start']

# team home court from the two previous seasons' results
prev = allg[(allg.season < Y) & (allg.done) & (allg.neutral == 0)]
hm = prev.assign(m=prev.home_pts - prev.away_pts).groupby('home').m.agg(['sum', 'count'])
am = prev.assign(m=prev.away_pts - prev.home_pts).groupby('away').m.agg(['sum', 'count'])
lg_h = (prev.home_pts - prev.away_pts).mean()


def home_edge(K):
    # (home margin - road margin)/2 relative to the league's home edge, shrunk with K games of each
    e = {}
    for t in hm.index:
        h_ = (hm.at[t, 'sum'] + K * lg_h) / (hm.at[t, 'count'] + K); a_ = (am.at[t, 'sum'] - K * lg_h) / (am.at[t, 'count'] + K)
        e[t] = (h_ - a_) / 2 - lg_h
    return np.array([e.get(t, 0.0) for t in G.home])


def team_resid(K):
    # each team's earlier residual this season (as of tip-off), shrunk; returns the home-minus-away expected residual
    acc = {}; out = np.zeros(len(G)); ng = np.zeros((len(G), 2))
    for i in order:
        h, a = G.home[i], G.away[i]; sh, nh = acc.get(h, (0.0, 0)); sa, na = acc.get(a, (0.0, 0))
        out[i] = sh / (nh + K) - sa / (na + K); ng[i] = (nh, na)
        acc[h] = (sh + res_m[i], nh + 1); acc[a] = (sa - res_m[i], na + 1)
    return out, ng


def tot_roll(N, K):
    out = np.zeros(len(G)); seen = []
    for i in order:
        r = np.array(seen[-N:]); out[i] = r.sum() / (len(r) + K) if len(r) else 0.0; seen.append(res_t[i])
    return out


# ---------- pricing ----------
base_line_sp = np.round(-(mh - ma)) + 0.5; base_line_t = np.floor(mh + ma) + 0.5


def lr(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gt(x, L):
    # P(x > L) for integer simulated scores moved by a fractional shift: each score spread over a 1-point window, so
    # it is exact when the shift is 0 and moves smoothly with it (a hard cut would ignore shifts under a point)
    return np.clip(x - L + 0.5, 0, 1).mean(1)


def price(dm, dt):
    m = (R[:, :, 0] - R[:, :, 1]) + dm[:, None]; t = R.sum(2) + dt[:, None]
    out = {'moneyline': lr((H > A).astype(float), gt(m, 0.0)),
           'spread': lr(((H - A) + base_line_sp > 0).astype(float), gt(m, -base_line_sp[:, None])),
           'total': lr((H + A > base_line_t).astype(float), gt(t, base_line_t[:, None]))}
    for d in (-8, 8):
        L = base_line_t + d; out[f'total {d:+d}'] = lr((H + A > L).astype(float), gt(t, L[:, None]))
    return out


Z0 = np.zeros(len(G)); BASE = price(Z0, Z0)


def compare(dm, dt, sel=None):
    P = price(dm, dt); r = {}
    for k in BASE:
        d = BASE[k] - P[k]
        if sel is not None: d = d[sel]
        if dt is Z0 and k.startswith('total'): continue
        if dm is Z0 and k in ('moneyline', 'spread'): continue
        bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)])
        r[k] = [round(float(d.mean()), 5), round(float(np.percentile(bt, 2.5)), 5), round(float(np.percentile(bt, 97.5)), 5)]
    return r


L = [f'game-level candidates, season ending {Y} ({len(G)} games); log loss gain vs current model (positive = better) [95% by game]']
out = {}


def report(name, r, extra=''):
    out[name] = r; L.append(f'{name:44s} ' + '  '.join(f'{k} {v[0]:+.5f} [{v[1]:+.5f},{v[2]:+.5f}]' for k, v in r.items()) + extra); print(L[-1], flush=True)


# schedule spots: fit on 2024-25 residuals, applied as fit
if Y == 2025:
    lam = 20.0
    bm = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ res_m)
    bt_ = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ res_t)
    fit = {'sched_margin': bm.tolist(), 'sched_total': bt_.tolist()}
else:
    bm, bt_ = np.array(FIT['sched_margin']), np.array(FIT['sched_total'])
L.append('  schedule coefficients (points, home minus away for margin; summed for total): margin ' + ', '.join(f'{n} {b:+.2f}' for n, b in zip(SN, bm))
         + ' | total ' + ', '.join(f'{n} {b:+.2f}' for n, b in zip(SN[:3], bt_)))
L.append('  games with each spot (either team): ' + ', '.join(f'{n} {int((np.abs(Xs[:, j]) > 0).sum())}' for j, n in enumerate(SN)))
print(L[-2]); print(L[-1])
report('sched (margin)', compare(Xs @ bm, Z0)); report('sched (total)', compare(Z0, Xt @ bt_))

# team home court
cands = {}
for K in (40, 80, 160):
    he = home_edge(K)
    for w in (0.5, 1.0):
        key = f'homeT K{K} w{w}'
        if Y == 2025 or FIT['pick'].get('homeT') == key: cands[key] = compare(w * he, Z0)
# team residuals (rating lag / roster turnover)
for K in (10, 20, 40):
    tr, ng = team_resid(K)
    for w in (0.5, 1.0):
        key = f'teamres K{K} w{w}'
        if Y == 2025 or FIT['pick'].get('teamres') == key:
            early = (ng.min(1) < 20)
            r = compare(w * tr, Z0); r2 = compare(w * tr, Z0, sel=early)
            cands[key] = r; cands[key + ' (first 20 games)'] = r2
for N in (100, 200, 400):
    for K in (25, 100):
        key = f'totroll N{N} K{K}'
        if Y == 2025 or FIT['pick'].get('totroll') == key: cands[key] = compare(Z0, tot_roll(N, K))
for k, r in cands.items(): report(k, r)

if Y == 2025:
    def best(prefix, metric):
        ks = [k for k in cands if k.startswith(prefix) and '(' not in k]
        return max(ks, key=lambda k: sum(cands[k][m][0] for m in metric))
    fit['pick'] = {'homeT': best('homeT', ('moneyline', 'spread')), 'teamres': best('teamres', ('moneyline', 'spread')),
                   'totroll': best('totroll', ('total', 'total -8', 'total +8'))}
    L.append('picked on 2024-25 (best summed gain): ' + json.dumps(fit['pick'])); print(L[-1])
    json.dump(fit, open('validation/game_adj_fit_2025.json', 'w'), indent=1)
json.dump(out, open(f'validation/game_adj_{Y}.json', 'w'), indent=1)
open(f'validation/game_adj_{Y}.log', 'w').write('\n'.join(L) + '\n')
