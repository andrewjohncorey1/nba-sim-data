"""
SGP plan step 3: players inside each simulated game, with minutes tied to the simulated margin (blowouts).

Usage (repo root): python3 validation/sgp3.py <season>        (2025 = check season, 2026 = test season)
Needs validation/rows_<Y>_base.bin (every simulated final score), validation/props2_pre_<Y>.pkl (player-games,
earlier-games-only histories), validation/props5_tuned.json (minutes model and rates), and
validation/minutes_dist_for_<Y>.json / minutes_distc_for_<Y>.json (minutes ratios by role, and by role x margin,
fit on earlier seasons only).

Two joint models for the same player-games and simulations, each against its own legs multiplied:
  uncond  each simulated game uses the player's minutes distribution for his role (as the single-leg props)
  margin  each simulated game uses his role's distribution for that simulation's final margin (0-9, 10-19, 20+):
          starters play less in blowouts, bench players more. Normalized so his average night is unchanged.
Points given the night: scoring plays (free throws, twos, threes in his own mix), mean = his share x that
simulation's team points x the night factor (power beta, as tuned). Book-style line (season average rounded down +
0.5, players with 5+ games). Combos as validation/sgp.py: A points over + own team wins, B points over + game over,
F A+B (3 legs), D top two teammates both over, E both teams' top scorers over, G D + their team wins.
"""
import sys, json
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import props_minutes as PM, points_shape as PSH, tune_props2 as T

Y = int(sys.argv[1])
T5 = json.load(open('validation/props5_tuned.json')); cfg = T5['cfg']; BETA = T5['choice']['pts']['beta']
MD = json.load(open(f'validation/minutes_dist_for_{Y}.json')); MC = json.load(open(f'validation/minutes_distc_for_{Y}.json'))
P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); D = P['D']; nodes = P['nodes']
meta = json.load(open(f'validation/rows_{Y}_base.bin.json')); N = meta['N']
R = np.fromfile(f'validation/rows_{Y}_base.bin', dtype=np.int16).reshape(len(meta['game_ids']), N, 2).astype(int)
gidx = {g: i for i, g in enumerate(meta['game_ids'])}
games = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id')

mu, _ = PM.build(P, cfg, MD, 'rotation')
alloc = T.project_minutes(P, cfg['min'])
band = np.clip(np.searchsorted(MD['edges'], alloc, side='right') - 1, 0, len(MD['edges']) - 2)
QI = PM.QI
qu = np.array(MD['q'])[:, QI]; Eu = qu.mean(1)                                  # unconditional, by band
qc = np.array(MC['q'])[:, :, QI]                                                # band x margin bucket x quantile
share_m = np.array(MC['share'])
fac_u = (qu / Eu[:, None]) ** BETA; fac_u = fac_u / fac_u.mean(1, keepdims=True)
fc = (qc / Eu[:, None, None]) ** BETA
norm = (fc.mean(2) * share_m[None, :]).sum(1)                                   # average night = 1 over real margins
fac_c = fc / norm[:, None, None]
fr = PSH.mix_parts(P, cfg)
ET = np.array([nodes[(g, s, 'pts_mean')] for g, s in zip(D.game_id, D.side)])
ok = (D.n_season >= 5).values & D.game_id.isin(gidx).values


def p_over_sims(rows_i, T_s, mb_s, line, mode):
    """P(over | simulation) for players rows_i (indices into D) given their team's points T_s and margin bucket mb_s."""
    K = int(np.floor(line.max())); keys = T_s * 4 + mb_s
    uk, inv = np.unique(keys, return_inverse=True); Tk, mk = uk // 4, uk % 4
    out = np.empty((len(rows_i), len(T_s)))
    for j, i in enumerate(rows_i):
        base = mu['pts'][1][i] / ET[i] * Tk                                       # mean at each (T, bucket) before the night factor
        f = fac_u[band[i]][None, :].repeat(len(uk), 0) if mode == 'uncond' else fac_c[band[i]][mk]
        lam = base[:, None] * f                                                   # (combos, quantiles)
        l1, l2, l3 = lam * fr[0][i], lam * fr[1][i] / 2, lam * fr[2][i] / 3
        p = np.zeros((K + 1,) + lam.shape); p[0] = np.exp(-(l1 + l2 + l3))
        for x in range(1, K + 1):
            a = l1 * p[x - 1]
            if x >= 2: a = a + 2 * l2 * p[x - 2]
            if x >= 3: a = a + 3 * l3 * p[x - 3]
            p[x] = a / x
        Kl = int(np.floor(line[j])); cdf = p[:Kl + 1].sum(0).mean(1)               # P(points <= floor(line)), mixed over the night
        out[j] = (1 - cdf)[inv]
    return out


rows = {m: [] for m in ('uncond', 'margin')}
for gid, gd in D[ok].groupby('game_id'):
    i = gidx[gid]; g = games.loc[gid]; h, a = R[i, :, 0], R[i, :, 1]; marg = h - a; tot = h + a
    mb = np.minimum(np.searchsorted(MC['medges'], np.abs(marg), side='right') - 1, 2)
    Lt = np.floor(tot.mean()) + 0.5; over_s = (tot > Lt).astype(float)
    am, at = g.home_pts - g.away_pts, g.home_pts + g.away_pts
    for mode in rows:
        per = []
        for side in ('h', 'a'):
            sd = gd[gd.side == side]
            if not len(sd): continue
            idx = sd.index.values; line = np.floor(sd.b_pts.values) + 0.5
            Ts = h if side == 'h' else a
            ps = p_over_sims(idx, Ts, mb, line, mode)
            win_s = (marg > 0).astype(float) if side == 'h' else (marg < 0).astype(float)
            won = int(am > 0) if side == 'h' else int(am < 0)
            for j, ii in enumerate(idx):
                yA = int(sd.a_pts.values[j] > line[j]); p_s = ps[j]
                rows[mode] += [dict(game_id=gid, kind='A', mean=mu['pts'][1][ii], pA=p_s.mean(), pB=win_s.mean(), joint=(p_s * win_s).mean(), yA=yA, yB=won),
                               dict(game_id=gid, kind='B', mean=mu['pts'][1][ii], pA=p_s.mean(), pB=over_s.mean(), joint=(p_s * over_s).mean(), yA=yA, yB=int(at > Lt)),
                               dict(game_id=gid, kind='F', mean=mu['pts'][1][ii], pA=p_s.mean(), pB=(win_s * over_s).mean(),
                                    legs3=p_s.mean() * win_s.mean() * over_s.mean(), joint=(p_s * win_s * over_s).mean(), yA=yA, yB=int(won and at > Lt))]
                per.append((side, mu['pts'][1][ii], p_s, yA, win_s, won))
        tops = {}
        for side in ('h', 'a'):
            L = sorted([x for x in per if x[0] == side], key=lambda x: -x[1])
            if L: tops[side] = L[0]
            if len(L) >= 2:
                (_, m1, p1, y1, ws, wn), (_, m2, p2, y2, _, _) = L[0], L[1]
                rows[mode] += [dict(game_id=gid, kind='D', mean=min(m1, m2), pA=p1.mean(), pB=p2.mean(), joint=(p1 * p2).mean(), yA=y1, yB=y2),
                               dict(game_id=gid, kind='G', mean=min(m1, m2), pA=p1.mean(), pB=(p2 * ws).mean(), legs3=p1.mean() * p2.mean() * ws.mean(),
                                    joint=(p1 * p2 * ws).mean(), yA=y1, yB=int(y2 and wn))]
        if len(tops) == 2:
            (_, m1, p1, y1, _, _), (_, m2, p2, y2, _, _) = tops['h'], tops['a']
            rows[mode].append(dict(game_id=gid, kind='E', mean=min(m1, m2), pA=p1.mean(), pB=p2.mean(), joint=(p1 * p2).mean(), yA=y1, yB=y2))

rng = np.random.default_rng(7)


def lr(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(x):
    lj, lg = lr(x.y.values, x.joint.values), lr(x.y.values, x.legs.values)
    d = pd.Series(lg - lj).groupby(x.game_id.values).sum().values; Gn = len(d)
    boot = np.array([d[rng.integers(0, Gn, Gn)].sum() for _ in range(2000)]) / len(x)
    pAgB = x.joint / x.pB.clip(1e-6); pAgnB = (x.pA - x.joint) / (1 - x.pB).clip(1e-6); hb, mbm = x.yB == 1, x.yB == 0
    return {'n': int(len(x)), 'll_joint': round(float(lj.mean()), 5), 'll_legs': round(float(lg.mean()), 5),
            'improvement': round(float(lg.mean() - lj.mean()), 5), 'ci': [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)],
            'hit': round(float(x.y.mean()), 4), 'pred_joint': round(float(x.joint.mean()), 4), 'pred_legs': round(float(x.legs.mean()), 4),
            'gap_pred': round(float(pAgB[hb].mean() - pAgnB[mbm].mean()), 4), 'gap_act': round(float(x.yA[hb].mean() - x.yA[mbm].mean()), 4),
            'legA': [round(float(x.pA.mean()), 4), round(float(x.yA.mean()), 4)]}


out = {'season': Y, 'combos': {}}
L = [f'SGP step 3 (players inside each simulated game), season ending {Y}; points book lines, minutes model, scoring plays, beta {BETA}']
for mode in rows:
    df = pd.DataFrame(rows[mode]); df['legs'] = df.legs3.fillna(df.pA * df.pB) if 'legs3' in df else df.pA * df.pB; df['y'] = df.yA * df.yB
    out['combos'][mode] = {}
    for k in ('A', 'B', 'F', 'D', 'E', 'G'):
        for sub, x in (('all', df[df.kind == k]), ('15plus', df[(df.kind == k) & (df['mean'] >= 15)])):
            e = evaluate(x); out['combos'][mode][f'{k} {sub}'] = e
            L.append(f'{mode:6s} {k} {sub:6s} n {e["n"]:5d}  joint LL {e["ll_joint"]:.5f} vs legs {e["ll_legs"]:.5f}  gain {e["improvement"]:+.5f} '
                     f'[{e["ci"][0]:+.5f},{e["ci"][1]:+.5f}]  gap pred {100 * e["gap_pred"]:+.1f} act {100 * e["gap_act"]:+.1f}  '
                     f'hit {e["hit"]:.3f} joint {e["pred_joint"]:.3f} legs {e["pred_legs"]:.3f}  legA {e["legA"][0]:.3f}/{e["legA"][1]:.3f}')
dfs = {m: pd.DataFrame(rows[m]) for m in rows}
for m in dfs:
    dfs[m]['y'] = dfs[m].yA * dfs[m].yB
L.append('margin vs uncond (joint log loss, positive = margin-linked minutes better; 95% interval by game); single leg = points over alone')
out['margin_vs_uncond'] = {}
for k in ('A', 'B', 'F', 'D', 'E', 'G', 'leg'):
    xu, xm = (dfs['uncond'][dfs['uncond'].kind == ('A' if k == 'leg' else k)], dfs['margin'][dfs['margin'].kind == ('A' if k == 'leg' else k)])
    if k == 'leg': d = lr(xu.yA.values, xu.pA.values) - lr(xm.yA.values, xm.pA.values)
    else: d = lr(xu.y.values, xu.joint.values) - lr(xm.y.values, xm.joint.values)
    g = pd.Series(d).groupby(xu.game_id.values).sum().values; Gn = len(g)
    boot = np.array([g[rng.integers(0, Gn, Gn)].sum() for _ in range(2000)]) / len(d)
    e = {'n': int(len(d)), 'gain': round(float(d.mean()), 5), 'ci': [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)]}
    out['margin_vs_uncond'][k] = e
    L.append(f'  {k:4s} n {e["n"]:5d}  gain {e["gain"]:+.5f} [{e["ci"][0]:+.5f},{e["ci"][1]:+.5f}]')
json.dump(out, open(f'validation/sgp3_{Y}.json', 'w'), indent=1)
open(f'validation/sgp3_{Y}.log', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
