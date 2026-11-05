"""
SGP plan step 4 (scan): a shared game environment. Both teams' simulated points scaled by one factor per simulated
game, exp(sigma x z), z standard normal: a quick post-hoc version of a game-level pace / shooting swing, to size sigma
before it goes into the engine.

Usage (repo root): python3 validation/shared_env.py <season>
Checks (each simulated game's rows from validation/rows_<Y>_base.bin vs the real finals): totals log loss at the fair
line and alt lines (fair +-4, +-8, +-12), home/away team totals, spread at the fair line, home-away covariance.
"""
import sys, json
import numpy as np, pandas as pd

Y = int(sys.argv[1])
meta = json.load(open(f'validation/rows_{Y}_base.bin.json')); N = meta['N']
R = np.fromfile(f'validation/rows_{Y}_base.bin', dtype=np.int16).reshape(len(meta['game_ids']), N, 2).astype(float)
g = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id').loc[meta['game_ids']]
H, A = g.home_pts.values, g.away_pts.values
# rows were simulated before the last totals calibration: they sit 1.94 points under real totals in 2024-25 (1.97 in
# 2025-26); every row is shifted by the 2024-25 figure (half to each team) so width is not confused with the mean
R = R + 1.94 / 2
rng = np.random.default_rng(11); Z = rng.standard_normal((len(R), N))


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


L = [f'shared game environment scan, season ending {Y}: points x exp(sigma z), same z both teams']
for s in (0, 0.02, 0.03, 0.04, 0.05, 0.06):
    f = np.exp(s * Z); h, a = np.round(R[:, :, 0] * f), np.round(R[:, :, 1] * f); t = h + a; m = h - a
    c = np.floor(t.mean(1)) + 0.5
    tl = {d: ll((H + A > c + d).astype(float), (t > (c + d)[:, None]).mean(1)) for d in (-12, -8, -4, 0, 4, 8, 12)}
    ch, ca = np.floor(h.mean(1)) + 0.5, np.floor(a.mean(1)) + 0.5
    tt = ll((H > ch).astype(float), (h > ch[:, None]).mean(1)) + ll((A > ca).astype(float), (a > ca[:, None]).mean(1))
    tt_alt = sum(ll((H > ch + d).astype(float), (h > (ch + d)[:, None]).mean(1)) + ll((A > ca + d).astype(float), (a > (ca + d)[:, None]).mean(1)) for d in (-8, 8))
    sp = np.round(-m.mean(1)) + 0.5
    spl = ll((H - A + sp > 0).astype(float), ((m + sp[:, None]) > 0).mean(1))
    ml = ll((H > A).astype(float), (m > 0).mean(1))
    cov = np.mean([np.cov(h[i], a[i])[0, 1] for i in range(0, len(R), 5)])
    L.append(f'sigma {s:.2f}: totals LL fair {tl[0]:.5f} alt avg {np.mean([tl[d] for d in tl if d]):.5f} (+-12: {tl[-12]:.4f}/{tl[12]:.4f})  '
             f'team totals {tt / 2:.5f} alt {tt_alt / 4:.5f}  spread {spl:.5f}  ML {ml:.5f}  sd total {t.std(1).mean():.2f}  cov h,a {cov:.1f}')
    print(L[-1])
open(f'validation/shared_env_{Y}.log', 'w').write('\n'.join(L) + '\n')
