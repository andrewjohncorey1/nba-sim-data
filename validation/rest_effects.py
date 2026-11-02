"""Estimate rest / back-to-back / travel effects on offensive efficiency from 2021-22..2023-24 (before the tuning and
test seasons), with team-season offense and defense effects so schedule quirks are not confused with team quality.
Writes validation/rest_effects.json: points per 100 possessions for the team's own offense ('own_*') and for its
opponent's offense when the feature applies to the defending team ('dfn_*')."""
import sys, json
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import schedule_feats as SF
F = SF.features([2022, 2023, 2024]).set_index('game_id')
rows = []
for y in (2022, 2023, 2024):
    t = pd.read_parquet(f'data/derived/teamgames_{y}.parquet'); g = pd.read_parquet(f'data/derived/games_{y}.parquet')
    t = t.merge(g[['game_id', 'neutral', 'stype']], on='game_id'); t['season'] = y
    rows.append(t)
T = pd.concat(rows); T = T[T.stype == 2]
T['y'] = 100 * (T.pts) / T.poss
T = T.join(F, on='game_id', rsuffix='_f')
def side(r, own):
    s = ('h' if r == 1 else 'a') if own else ('a' if r == 1 else 'h')
    return s
feat = ['b2b', 'rest3', 'trav', 'tzE', 'tzW']
for who in ('own', 'dfn'):
    for f in feat:
        T[f'{who}_{f}'] = 0.0
for r in (1, 0):
    m = T.home == r
    for who, own in (('own', True), ('dfn', False)):
        s = side(r, own)
        T.loc[m, f'{who}_b2b'] = T.loc[m, f'b2b_{s}']
        T.loc[m, f'{who}_rest3'] = (T.loc[m, f'rest_{s}'] >= 3).astype(float)
        T.loc[m, f'{who}_trav'] = T.loc[m, f'trav_{s}'] / 1000
        T.loc[m, f'{who}_tzE'] = T.loc[m, f'tz_{s}'].clip(lower=0)
        T.loc[m, f'{who}_tzW'] = (-T.loc[m, f'tz_{s}']).clip(lower=0)
T['hm'] = np.where(T.neutral == 1, 0, T.home)
T['ok'] = T.team + T.season.astype(str); T['dk'] = T.opp + T.season.astype(str)
X = pd.concat([pd.get_dummies(T.ok, prefix='o').astype(float), -pd.get_dummies(T.dk, prefix='d').astype(float)], axis=1)
cols = ['hm'] + [f'{w}_{f}' for w in ('own', 'dfn') for f in feat]
X = pd.concat([X, T[cols].astype(float)], axis=1)
w = T.poss.values
beta, *_ = np.linalg.lstsq(X.values * np.sqrt(w)[:, None], T.y.values * np.sqrt(w), rcond=None)
res = T.y.values - X.values @ beta
sig2 = np.sum(w * res ** 2) / (len(T) - X.shape[1])
cov = sig2 * np.linalg.pinv((X.values * w[:, None]).T @ X.values)
out = {}
for i, c in enumerate(X.columns):
    if c in cols: out[c] = {'est': round(float(beta[i]), 3), 'se': round(float(np.sqrt(cov[i, i])), 3)}
print(json.dumps(out, indent=1))
json.dump({'fit_on': '2021-22..2023-24 regular seasons', 'units': 'points per 100 possessions of the offense', **out},
          open('validation/rest_effects.json', 'w'), indent=1)
