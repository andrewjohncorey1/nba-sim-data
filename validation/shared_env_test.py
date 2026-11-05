"""
SGP plan step 4 (test): the engine's shared game environment (G.env_sd, one pace factor per simulated game for both
teams) against the same engine without it, on every game market and on the same-game combos.

Usage (repo root): python3 validation/shared_env_test.py <season> <tag> <rows_base.bin> <rows_env.bin>
  rows files from validation/sim_games.js (ROWS=...), same inputs, GOVR='{"env_sd": x}' for the second.
Both are shifted by +1 point per team (the held-out simulations sit 1.94 points under real totals in 2024-25; see
validation/shared_env.py), so width is not confused with the mean. Markets: totals and team totals at the fair line
and alt lines, spread at the fair line and alt spreads, moneyline, first-half / quarter are not in the rows file.
Controls: home covers + game over, home wins + game over (side + total must stay near independent).
Same-game combos come from validation/sgp3.py run on each rows file (SGP3_SUFFIX); this script compares them.
"""
import sys, json
import numpy as np, pandas as pd

Y, TAG, FB, FE = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
g = None
rng = np.random.default_rng(5)


def load(f):
    meta = json.load(open(f + '.json'))
    R = np.fromfile(f, dtype=np.int16).reshape(len(meta['game_ids']), meta['N'], 2).astype(float) + 1.0
    return meta['game_ids'], R


ids, RB = load(FB); ids2, RE = load(FE); assert ids == ids2
g = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id').loc[ids]
H, A = g.home_pts.values.astype(float), g.away_pts.values.astype(float)


def lr(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def markets(R, ref):
    """Per-game log loss per market. Lines come from the base simulation (ref) so both models price the same lines."""
    h, a = R[:, :, 0], R[:, :, 1]; t, m = h + a, h - a
    th, ah, tt, mm = ref
    out = {}
    for d in (0, -4, 4, -8, 8, -12, 12):
        L = tt + d; out[f'total {d:+d}'] = lr((H + A > L).astype(float), (t > L[:, None]).mean(1))
    for d in (0, -4, 4, -8, 8):
        Lh, La = th + d, ah + d
        out[f'team total {d:+d}'] = (lr((H > Lh).astype(float), (h > Lh[:, None]).mean(1)) + lr((A > La).astype(float), (a > La[:, None]).mean(1))) / 2
    for d in (0, -6, 6):
        L = mm + d; out[f'spread {d:+d}'] = lr((H - A + L > 0).astype(float), ((m + L[:, None]) > 0).mean(1))
    out['moneyline'] = lr((H > A).astype(float), (m > 0).mean(1))
    ov = t > tt[:, None]; yo = (H + A > tt)
    for k, side, ys in (('C home covers + over', (m + mm[:, None]) > 0, H - A + mm > 0), ('C2 home wins + over', m > 0, H > A)):
        j = (side & ov).mean(1); legs = side.mean(1) * ov.mean(1); y = (ys & yo).astype(float)
        out[k] = lr(y, j); out[k + ' (legs multiplied)'] = lr(y, legs)
    out['_sd_total'] = t.std(1); out['_cov'] = np.array([np.cov(h[i], a[i])[0, 1] for i in range(len(R))])
    return out


hb, ab = RB[:, :, 0], RB[:, :, 1]
ref = (np.floor(hb.mean(1)) + 0.5, np.floor(ab.mean(1)) + 0.5, np.floor((hb + ab).mean(1)) + 0.5, np.round(-(hb - ab).mean(1)) + 0.5)
MB, ME = markets(RB, ref), markets(RE, ref)


def boot(d):
    n = len(d); b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


res = {'season': Y, 'tag': TAG, 'markets': {}, 'combos': {}}
L = [f'shared game environment test ({TAG}), season ending {Y}: log loss without -> with (positive gain = with is better; 95% by game)',
     f'  sd of simulated total {MB["_sd_total"].mean():.2f} -> {ME["_sd_total"].mean():.2f} (real, around the simulated mean: {np.std(H + A - (RB.sum(2)).mean(1)):.2f}); '
     f'home/away covariance {MB["_cov"].mean():.1f} -> {ME["_cov"].mean():.1f} (real {np.cov(H - RB[:, :, 0].mean(1), A - RB[:, :, 1].mean(1))[0, 1]:.1f})']
res['coverage'] = {}
for nm_, R_ in (('without', RB), ('with', RE)):
    for k, sim, act in (('total', R_.sum(2), H + A), ('team total', np.concatenate([R_[:, :, 0], R_[:, :, 1]]), np.concatenate([H, A])), ('margin', R_[:, :, 0] - R_[:, :, 1], H - A)):
        pit = (sim < act[:, None]).mean(1) + 0.5 * (sim == act[:, None]).mean(1)
        c = {'in80': round(float(((pit > .1) & (pit < .9)).mean()), 3), 'in50': round(float(((pit > .25) & (pit < .75)).mean()), 3),
             'below10': round(float((pit < .1).mean()), 3), 'above90': round(float((pit > .9).mean()), 3)}
        res['coverage'][f'{k} {nm_}'] = c
        L.append(f'  {k:10s} {nm_:7s}: real result inside the simulated 80% range {c["in80"]:.3f} (should be 0.80), 50% range {c["in50"]:.3f}; below 10th {c["below10"]:.3f}, above 90th {c["above90"]:.3f}')
for k in MB:
    if k.startswith('_'): continue
    d = MB[k] - ME[k]; lo, hi = boot(d)
    res['markets'][k] = {'without': round(float(MB[k].mean()), 5), 'with': round(float(ME[k].mean()), 5), 'gain': round(float(d.mean()), 5), 'ci': [round(lo, 5), round(hi, 5)]}
    L.append(f'  {k:34s} {MB[k].mean():.5f} -> {ME[k].mean():.5f}  gain {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]')
for k in ('C home covers + over', 'C2 home wins + over'):
    for nm, M in (('without', MB), ('with', ME)):
        L.append(f'  {k} {nm}: joint {M[k].mean():.5f} vs legs multiplied {M[k + " (legs multiplied)"].mean():.5f}')

try:
    SB = pd.read_parquet(f'validation/sgp3_rows_{Y}_{TAG}_base.parquet'); SE = pd.read_parquet(f'validation/sgp3_rows_{Y}_{TAG}_env.parquet')
    L.append('  same-game combos (points over, minutes tied to margin; validation/sgp3.py on each rows file): joint log loss without -> with')
    for k in ('A', 'B', 'F', 'D', 'E', 'G'):
        xb, xe = SB[SB.kind == k].reset_index(drop=True), SE[SE.kind == k].reset_index(drop=True)
        d = lr(xb.y.values, xb.joint.values) - lr(xe.y.values, xe.joint.values)
        gsum = pd.Series(d).groupby(xb.game_id.values).sum().values; n = len(gsum)
        bt = np.array([gsum[rng.integers(0, n, n)].sum() for _ in range(2000)]) / len(d)
        lo, hi = np.percentile(bt, [2.5, 97.5])

        def gap(x):
            pAgB = x.joint / x.pB.clip(1e-6); pAgnB = (x.pA - x.joint) / (1 - x.pB).clip(1e-6)
            return 100 * float(pAgB[x.yB == 1].mean() - pAgnB[x.yB == 0].mean()), 100 * float(x.yA[x.yB == 1].mean() - x.yA[x.yB == 0].mean())
        gb, ge = gap(xb), gap(xe)
        legb = lr(xb.yA.values, xb.pA.values).mean() - lr(xe.yA.values, xe.pA.values).mean()
        res['combos'][k] = {'gain': round(float(d.mean()), 5), 'ci': [round(float(lo), 5), round(float(hi), 5)], 'gap_without': round(gb[0], 1),
                            'gap_with': round(ge[0], 1), 'gap_actual': round(gb[1], 1), 'leg_gain': round(float(legb), 5)}
        L.append(f'  {k}: gain {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  gap {gb[0]:+.1f} -> {ge[0]:+.1f} (actual {gb[1]:+.1f})  first leg alone {legb:+.5f}')
except FileNotFoundError:
    L.append('  (same-game combos: run validation/sgp3.py on both rows files first)')
json.dump(res, open(f'validation/shared_env_test_{Y}_{TAG}.json', 'w'), indent=1)
open(f'validation/shared_env_test_{Y}_{TAG}.log', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
