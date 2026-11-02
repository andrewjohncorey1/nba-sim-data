"""Tune rating and availability settings on 2024-25 only (closed-form expected margin and regulation total; every game rated on
earlier games only). Coordinate search over half-life and ridge strengths. Writes validation/tuned_params.json."""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import rolling as RO, ratings as RT, players as PS
from avail import Avail
Y = 2025
TG = RT.load_tg(range(Y - 3, Y + 1)); PL = PS.load(range(Y - 3, Y + 1))
PP = dict(half_life_p=400, lam_p=20)
base = dict(half_life=60, K_other=3, K_make=5, K_ft=20, K_pace=3, alpha=1.0)
grid = {'alpha': [0.75, 1.0, 1.25, 1.5], 'half_life': [45, 60, 90, 120, 180], 'K_make': [1, 2, 5, 10], 'K_other': [1, 2, 3, 5],
        'K_pace': [1, 3, 10]}
cache = {}
def obj(prm):
    key = tuple(sorted(prm.items()))
    if key not in cache:
        av = Avail(PL, PP, prm['half_life'], prm['alpha'], RO.E['ppp_sens'], RO.E['base_ppp'])
        s = RO.score_closed(RO.predict_season(Y, prm, TG, adj=av))
        cache[key] = s
        print(json.dumps({**prm, **{k: round(v, 4) for k, v in s.items()}}), flush=True)
    s = cache[key]
    return s['margin_rmse'] ** 2 + 0.5 * s['total_rmse'] ** 2      # margin matters most; totals too
best = dict(base)
for rnd in range(1):
    for k, vals in grid.items():
        res = {v: obj({**best, k: v}) for v in vals}
        best[k] = min(res, key=res.get)
    print('ROUND', rnd, best, flush=True)
json.dump({'ratings': best, 'score_2025': cache[tuple(sorted(best.items()))]}, open('validation/tuned_params.json', 'w'), indent=1)
