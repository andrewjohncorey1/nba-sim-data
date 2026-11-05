"""
SGP plan step 5: teammates compete for the team's points; opposing stars share a scoring environment.

Usage (repo root): python3 validation/sgp5.py <season> [rho_team rho_opp]
  without rhos: scan a grid (tuning season 2024-25); with rhos: report those (test season 2025-26).

Each player's per-simulation P(points over | that simulated game) is as in validation/sgp3.py (minutes tied to the
simulated margin, scoring plays, team points from that simulation; rows validation/rows_<Y>_env.bin from the engine
with the shared pace factor, +1 point per team for the totals offset). Within one simulated game the model so far
treats two players as independent. Here two teammates' results in the same simulated game are joined by a Gaussian
copula with correlation rho_team (negative = they compete for shots and minutes), and the two teams' top scorers
with rho_opp. A copula leaves every single leg exactly as it was; only joint prices move.
Pairs: D top two scorers of a team (by projection), D_all every teammate pair projected 10+ points, G D + their team
wins (3 legs), E the two teams' top scorers, E_all every cross pair projected 15+.
"""
import sys, json, os
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import norm as NRM

Y = int(sys.argv[1]); FIX = [float(x) for x in sys.argv[2:4]] if len(sys.argv) > 3 else None
os.environ.setdefault('ROWSFILE', f'validation/rows_{Y}_env.bin'); os.environ.setdefault('SHIFT', '1')
src = open('validation/sgp3.py').read().split('rows = {m: [] for m in')[0]
sys.argv = ['sgp3', str(Y)]; exec(src)

GL_x, GL_w = np.polynomial.legendre.leggauss(24)


def phi2(h, k, rho):
    """Bivariate normal CDF P(X<h, Y<k) with correlation rho (vectorised; Gauss-Legendre over the correlation)."""
    if rho == 0: return NRM.cdf(h) * NRM.cdf(k)
    r = 0.5 * rho * (GL_x + 1); w = 0.5 * rho * GL_w
    h, k = h[..., None], k[..., None]; s = 1 - r * r
    f = np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * s)) / np.sqrt(s)
    return NRM.cdf(h[..., 0]) * NRM.cdf(k[..., 0]) + (f * w).sum(-1) / (2 * np.pi)


def joint(pa, pb, rho):
    """P(both over | simulation) with copula correlation rho; over = latent above threshold."""
    pa, pb = np.clip(pa, 1e-9, 1 - 1e-9), np.clip(pb, 1e-9, 1 - 1e-9)
    return phi2(NRM.ppf(pa), NRM.ppf(pb), rho)      # P(Z_a < Phi^-1(pa), Z_b < Phi^-1(pb)) = P(both over) by symmetry


pairs = []          # (kind, game_id, p_a per sim, p_b per sim, y_a, y_b, win per sim or None, won)
for gid, gd in D[ok].groupby('game_id'):
    i = gidx[gid]; g = games.loc[gid]; h, a = R[i, :, 0], R[i, :, 1]; marg = h - a
    mb = np.minimum(np.searchsorted(MC['medges'], np.abs(marg), side='right') - 1, 2)
    am = g.home_pts - g.away_pts; per = {}
    for side in ('h', 'a'):
        sd = gd[gd.side == side]
        if not len(sd): continue
        idx = sd.index.values; line = np.floor(sd.b_pts.values) + 0.5
        ps = p_over_sims(idx, h if side == 'h' else a, mb, line, 'margin')
        win = (marg > 0) if side == 'h' else (marg < 0); won = int(am > 0) if side == 'h' else int(am < 0)
        L = sorted([(mu['pts'][1][ii], ps[j].astype(np.float32), int(sd.a_pts.values[j] > line[j])) for j, ii in enumerate(idx)], key=lambda x: -x[0])
        per[side] = L
        if len(L) >= 2:
            pairs.append(('D', gid, L[0][1], L[1][1], L[0][2], L[1][2], None, 0, min(L[0][0], L[1][0])))
            pairs.append(('G', gid, L[0][1], L[1][1], L[0][2], L[1][2], win.astype(np.float32), won, min(L[0][0], L[1][0])))
        for x in range(len(L)):
            for y in range(x + 1, len(L)):
                if L[y][0] >= 10: pairs.append(('D_all', gid, L[x][1], L[y][1], L[x][2], L[y][2], None, 0, L[y][0]))
    if len(per) == 2:
        H_, A_ = per['h'], per['a']
        pairs.append(('E', gid, H_[0][1], A_[0][1], H_[0][2], A_[0][2], None, 0, min(H_[0][0], A_[0][0])))
        for x in H_:
            for y in A_:
                if x[0] >= 15 and y[0] >= 15: pairs.append(('E_all', gid, x[1], y[1], x[2], y[2], None, 0, min(x[0], y[0])))

rng = np.random.default_rng(3)


def lr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(kind, rho):
    X = [p for p in pairs if p[0] == kind]
    pj, pl, pa, pb, ya, yb, gids = [], [], [], [], [], [], []
    for _, gid, a_, b_, y1, y2, win, won, _m in X:
        j = joint(a_, b_, rho)
        if win is not None:
            pj.append((j * win).mean()); pl.append(a_.mean() * b_.mean() * win.mean()); pb.append((b_ * win).mean()); yb.append(int(y2 and won))
        else:
            pj.append(j.mean()); pl.append(a_.mean() * b_.mean()); pb.append(b_.mean()); yb.append(y2)
        pa.append(a_.mean()); ya.append(y1); gids.append(gid)
    pj, pl, pa, pb, ya, yb = map(np.array, (pj, pl, pa, pb, ya, yb)); y = ya * yb
    return dict(pj=pj, pl=pl, pa=pa, pb=pb, ya=ya, yb=yb, y=y, g=np.array(gids))


def summary(S, ref=None):
    lj, lg = lr(S['y'], S['pj']), lr(S['y'], S['pl'])
    gA = S['pj'] / S['pb'].clip(1e-6); gN = (S['pa'] - S['pj']) / (1 - S['pb']).clip(1e-6); hb, mb_ = S['yb'] == 1, S['yb'] == 0
    out = {'n': int(len(lj)), 'll_joint': round(float(lj.mean()), 5), 'll_legs': round(float(lg.mean()), 5), 'gain_vs_legs': round(float((lg - lj).mean()), 5),
           'gap_pred': round(100 * float(gA[hb].mean() - gN[mb_].mean()), 1), 'gap_act': round(100 * float(S['ya'][hb].mean() - S['ya'][mb_].mean()), 1),
           'pred': round(float(S['pj'].mean()), 4), 'hit': round(float(S['y'].mean()), 4)}
    for nm, base in (('vs_legs', lg), ('vs_rho0', None if ref is None else lr(ref['y'], ref['pj']))):
        if base is None: continue
        d = pd.Series(base - lj).groupby(S['g']).sum().values; n = len(d)
        b = np.array([d[rng.integers(0, n, n)].sum() for _ in range(2000)]) / len(lj)
        out['ci_' + nm] = [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)]
        if nm == 'vs_rho0': out['gain_vs_rho0'] = round(float((base - lj).mean()), 5)
    return out


res = {'season': Y, 'rows': os.environ['ROWSFILE'], 'kinds': {}}
L = [f'SGP step 5, season ending {Y}: copula between players in the same simulated game (single legs unchanged)']
grid = {'D': [0, -0.1, -0.2, -0.3, -0.4], 'D_all': [0, -0.1, -0.2, -0.3], 'G': [0, -0.1, -0.2, -0.3, -0.4], 'E': [0, 0.05, 0.1, 0.15, 0.2], 'E_all': [0, 0.05, 0.1, 0.15]}
if FIX: grid = {k: [0, FIX[0] if k[0] in 'DG' else FIX[1]] for k in grid}
for kind, rhos in grid.items():
    ref = score(kind, 0.0); res['kinds'][kind] = {}
    for rho in rhos:
        S = ref if rho == 0 else score(kind, rho); e = summary(S, None if rho == 0 else ref); res['kinds'][kind][str(rho)] = e
        L.append(f'{kind:5s} rho {rho:+.2f}  n {e["n"]:5d}  joint LL {e["ll_joint"]:.5f} (legs multiplied {e["ll_legs"]:.5f})  '
                 + (f'vs rho 0 {e["gain_vs_rho0"]:+.5f} [{e["ci_vs_rho0"][0]:+.5f},{e["ci_vs_rho0"][1]:+.5f}]  ' if rho else '')
                 + f'vs legs {e["gain_vs_legs"]:+.5f} [{e["ci_vs_legs"][0]:+.5f},{e["ci_vs_legs"][1]:+.5f}]  gap pred {e["gap_pred"]:+.1f} act {e["gap_act"]:+.1f}  joint {e["pred"]:.4f} hit {e["hit"]:.4f}')
        print(L[-1], flush=True)
tag = 'test' if FIX else 'scan'
json.dump(res, open(f'validation/sgp5_{tag}_{Y}.json', 'w'), indent=1)
open(f'validation/sgp5_{tag}_{Y}.log', 'w').write('\n'.join(L) + '\n')
