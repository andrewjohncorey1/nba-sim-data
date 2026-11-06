"""
Same-game parlays, more leg types (built from validation/sgp6.py, same model and rows): player points unders, team
spread at the fair line, game under, mixed directions; side + total controls.
Previous docstring (sgp6):
SGP plan step 6: the full same-game joint test with everything kept so far, including 3- and 4-leg parlays.

Usage (repo root): python3 validation/sgp6.py <season>      (2025 = tuning season, for reference; 2026 = verdict)

Per simulated game (validation/rows_<Y>_env.bin: shared pace factor, +1 point per team for the totals offset):
team wins / game over are 0/1 in that simulation; a player's P(points over) is as in validation/sgp3.py (minutes
tied to the simulated margin, scoring plays, that simulation's team points). Two teammates in the same simulated game
are joined by the Gaussian copula from validation/sgp5.py (rho -0.2, chosen on 2024-25); players on opposite teams
are independent given the simulated game. Points lines book-style (season average rounded down + 0.5, 5+ games);
game over at the model's fair total. Each combo against its legs multiplied (the product of the model's own single
legs): the joint price is only worth using where it beats that, with the game-bootstrap interval above zero.

Combos (star = team's top projected scorer, second = next, opp = other team's top scorer):
  A star + team wins          B star + game over          F star + team wins + game over
  D star + second             G star + second + team wins H star + second + game over
  E star + opp                I star + opp + game over    K star + opp + team wins
  J star + second + team wins + game over (4 legs)
"""
import sys, json, os
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import norm as NRM

Y = int(sys.argv[1]); RHO = -0.2
os.environ.setdefault('ROWSFILE', f'validation/rows_{Y}_env.bin'); os.environ.setdefault('SHIFT', '1')
src = open('validation/sgp3.py').read().split('rows = {m: [] for m in')[0]
sys.argv = ['sgp3', str(Y)]; exec(src)
s5 = open('validation/sgp5.py').read(); exec(s5[s5.index('GL_x, GL_w'):s5.index('pairs = []')])

rows = []
for gid, gd in D[ok].groupby('game_id'):
    i = gidx[gid]; g = games.loc[gid]; h, a = R[i, :, 0], R[i, :, 1]; marg = h - a; tot = h + a
    mb = np.minimum(np.searchsorted(MC['medges'], np.abs(marg), side='right') - 1, 2)
    Lt = np.floor(tot.mean()) + 0.5; ov = (tot > Lt).astype(float); yov = int(g.home_pts + g.away_pts > Lt)
    am = g.home_pts - g.away_pts; per = {}
    for side in ('h', 'a'):
        sd = gd[gd.side == side]
        if len(sd) < 2: continue
        idx = sd.index.values; line = np.floor(sd.b_pts.values) + 0.5
        ps = p_over_sims(idx, h if side == 'h' else a, mb, line, 'margin')
        o = np.argsort(-mu['pts'][1][idx])[:2]
        per[side] = dict(p=[ps[o[0]], ps[o[1]]], y=[int(sd.a_pts.values[o[0]] > line[o[0]]), int(sd.a_pts.values[o[1]] > line[o[1]])],
                         win=((marg > 0) if side == 'h' else (marg < 0)).astype(float), won=int(am > 0) if side == 'h' else int(am < 0))
    lh = np.round(-marg.mean()) + 0.5; un = 1 - ov; yun = 1 - yov
    for side, X in per.items():
        s1, s2 = X['p']; y1, y2 = X['y']; w, yw = X['win'], X['won']
        cov = ((marg + lh) > 0) if side == 'h' else ((-marg - lh) > 0); ycov = int(am + lh > 0) if side == 'h' else int(-am - lh > 0)
        cov = cov.astype(float)
        both = joint(s1, s2, RHO); u1, u2 = 1 - s1, 1 - s2
        bu = joint(u1, u2, RHO)                      # both under (copula is symmetric in the flip of both legs)
        o1u2 = s1 - both                             # star over, teammate under
        legs = {'star': (s1, y1), 'star_u': (u1, 1 - y1), 'second': (s2, y2), 'second_u': (u2, 1 - y2), 'win': (w, yw), 'lose': (1 - w, 1 - yw),
                'over': (ov, yov), 'under': (un, yun), 'cover': (cov, ycov)}
        C = {'U1': (u1 * (1 - w), ['star_u', 'lose']), 'U2': (u1 * un, ['star_u', 'under']), 'U3': (u1 * (1 - w) * un, ['star_u', 'lose', 'under']),
             'S1': (s1 * cov, ['star', 'cover']), 'S2': (s1 * cov * ov, ['star', 'cover', 'over']), 'S3': (both * cov, ['star', 'second', 'cover']),
             'N1': (s1 * un, ['star', 'under']), 'N2': (o1u2, ['star', 'second_u']), 'N3': (bu, ['star_u', 'second_u']),
             'T1': (cov * ov, ['cover', 'over']), 'T2': (cov * un, ['cover', 'under'])}
        for k, (js, L) in C.items():
            rows.append(dict(game_id=gid, kind=k, n_legs=len(L), joint=float(js.mean()), legs=float(np.prod([legs[x][0].mean() for x in L])),
                             y=int(all(legs[x][1] for x in L))))

df = pd.DataFrame(rows); rng = np.random.default_rng(9)


def lr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


out = {'season': Y, 'rho_team': RHO, 'combos': {}}
names = {'U1': 'star under + team loses', 'U2': 'star under + game under', 'U3': 'star under + team loses + game under',
         'S1': 'star over + team covers', 'S2': 'star over + team covers + game over', 'S3': 'star + teammate over + team covers',
         'N1': 'star over + game under', 'N2': 'star over + teammate under', 'N3': 'star under + teammate under',
         'T1': 'team covers + game over (control)', 'T2': 'team covers + game under (control)'}
Lg = [f'SGP legs beyond the first test (unders, spreads, mixed directions), season ending {Y}: joint price vs legs multiplied (positive = joint better; 95% interval by game)']
for k in names:
    x = df[df.kind == k]; d = lr(x.y.values, x.legs.values) - lr(x.y.values, x.joint.values)
    gs = pd.Series(d).groupby(x.game_id.values).sum().values; n = len(gs)
    b = np.array([gs[rng.integers(0, n, n)].sum() for _ in range(2000)]) / len(d); lo, hi = np.percentile(b, [2.5, 97.5])
    # calibration of the joint level: predicted vs actual in thirds of the joint price
    x = x.assign(bin=pd.qcut(x.joint, 3, labels=False, duplicates='drop'))
    cal = x.groupby('bin').agg(pred=('joint', 'mean'), legs=('legs', 'mean'), hit=('y', 'mean'))
    e = {'legs_n': int(x.n_legs.iloc[0]), 'n': int(len(x)), 'gain': round(float(d.mean()), 5), 'ci': [round(float(lo), 5), round(float(hi), 5)],
         'pred': round(float(x.joint.mean()), 4), 'legs_mult': round(float(x.legs.mean()), 4), 'hit': round(float(x.y.mean()), 4),
         'thirds': [[round(float(r.pred), 4), round(float(r.legs), 4), round(float(r.hit), 4)] for r in cal.itertuples()],
         'pass': bool(lo > 0)}
    out['combos'][k] = e
    Lg.append(f'{k} {names[k]:40s} legs {e["legs_n"]} n {e["n"]:5d}  gain {e["gain"]:+.5f} [{lo:+.5f},{hi:+.5f}] {"PASS" if lo > 0 else "    "}  '
              f'avg joint {e["pred"]:.4f} legs mult {e["legs_mult"]:.4f} actual {e["hit"]:.4f}  thirds (joint/legs/actual) '
              + ' | '.join(f'{p:.3f}/{l:.3f}/{h_:.3f}' for p, l, h_ in e['thirds']))
    print(Lg[-1], flush=True)
Lg.append('pooled by family (every combo of the family in a game counted, interval by game):')
out['families'] = {}
for fam, ks in (('unders', ['U1', 'U2', 'U3']), ('spreads', ['S1', 'S2', 'S3']), ('mixed directions', ['N1', 'N2', 'N3']), ('controls', ['T1', 'T2'])):
    x = df[df.kind.isin(ks)]; d = lr(x.y.values, x.legs.values) - lr(x.y.values, x.joint.values)
    gs = pd.Series(d).groupby(x.game_id.values).sum().values; n = len(gs)
    b = np.array([gs[rng.integers(0, n, n)].sum() for _ in range(2000)]) / len(d); lo, hi = np.percentile(b, [2.5, 97.5])
    out['families'][fam] = {'n': int(len(d)), 'gain': round(float(d.mean()), 5), 'ci': [round(float(lo), 5), round(float(hi), 5)],
                            'pred': round(float(x.joint.mean()), 4), 'legs_mult': round(float(x.legs.mean()), 4), 'hit': round(float(x.y.mean()), 4)}
    Lg.append(f'  {fam:20s} n {len(d):6d}  gain {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] {"PASS" if lo > 0 else "    "}  avg joint {x.joint.mean():.4f} legs mult {x.legs.mean():.4f} actual {x.y.mean():.4f}')
    print(Lg[-1], flush=True)
df.to_parquet(f'validation/sgp7_rows_{Y}.parquet', index=False)
json.dump(out, open(f'validation/sgp7_{Y}.json', 'w'), indent=1)
open(f'validation/sgp7_{Y}.log', 'w').write('\n'.join(Lg) + '\n')
