"""
Points distribution test: negative binomial (current) vs "scoring plays" (compound Poisson).

Scoring-plays model: a player's points = free throws made (1 each) + twos made (2) + threes made (3), each Poisson,
with the means split by his own mix (per-minute rates of points, field goals made and threes made, same tuned rates
as the means), all multiplied by one shared gamma factor for his night (shape k: minutes, usage, hot or cold), so
the spread can be wider than Poisson. The pmf comes from Panjer's recursion. The mean is identical to the current
model's; only the shape changes. k is tuned on 2024-25 as k = k10 x (mean / 10)^g.

Usage (repo root): python3 validation/points_shape.py tune 2025 | test 2026
Same rows, means and lines as validation/tune_props2.py (book-style line + fixed lines, players with 5+ games).
"""
import sys, json
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import gamma as G, skewnorm
import tune_props2 as T

NMAX = 60
GQ = 9            # quantile points for the gamma night factor


def mix_parts(P, cfg):
    """Each player-game's split of expected points into free throws, twos, threes (fractions of points)."""
    D = P['D']
    rp, rf, r3 = T.rates(D, 'pts', cfg['rate']['pts']), T.rates(D, 'fgm', cfg['rate']['fgm']), T.rates(D, 'fg3m', cfg['rate']['fg3m'])
    p3 = 3 * r3; p2 = 2 * np.maximum(rf - r3, 0); p1 = np.maximum(rp - p2 - p3, 0)
    s = p1 + p2 + p3 + 1e-9
    return p1 / s, p2 / s, p3 / s


def sf_points(meanN, fr, k, lines, chunk=2000):
    n = meanN.shape[0]
    if n > chunk:
        parts = [sf_points(meanN[i:i + chunk], [f[i:i + chunk] for f in fr], k if isinstance(k, tuple) else k[i:i + chunk],
                           [L[i:i + chunk] for L in lines], chunk)
                          for i in range(0, n, chunk)]
        return [np.concatenate([p[j] for p in parts]) for j in range(len(lines))]
    return _sf_points(meanN, fr, k, lines)


def _sf_points(meanN, fr, k, lines):
    """P(points > L) for each row and line. meanN: (n, nodes) means over the team-points nodes; fr: (f1, f2, f3);
    k: (n,) gamma shapes; lines: list of (n,) arrays. Returns list of (n,) arrays."""
    n, m = meanN.shape
    q = (np.arange(GQ) + 0.5) / GQ
    if isinstance(k, tuple):        # ('skew', sigma, a): night factor 1 + sigma x standardized skew-normal(a), mean 1
        _, sig, a = k; d = a / np.sqrt(1 + a * a); mz, sz = d * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * d * d / np.pi)
        z = (skewnorm.ppf(q, a) - mz) / sz; gq = np.clip(1 + sig * z, 0.05, None); gq = gq / gq.mean()
        gq = np.broadcast_to(gq[None, :], (meanN.shape[0], GQ))
    else:
        gq = G.ppf(q[None, :], k[:, None], scale=1 / k[:, None])                # (n, GQ), mean 1
    lam = meanN[:, :, None] * gq[:, None, :]                                     # (n, nodes, GQ) expected points
    l1 = lam * fr[0][:, None, None]; l2 = lam * fr[1][:, None, None] / 2; l3 = lam * fr[2][:, None, None] / 3
    tot = l1 + l2 + l3
    pmf = np.zeros((NMAX + 1,) + lam.shape); pmf[0] = np.exp(-tot)
    jl = [None, l1, 2 * l2, 3 * l3]                                              # j x lambda_j
    for x in range(1, NMAX + 1):
        acc = np.zeros(lam.shape)
        for j in (1, 2, 3):
            if x - j >= 0: acc += jl[j] * pmf[x - j]
        pmf[x] = acc / x
    cdf = np.cumsum(pmf, axis=0).mean(axis=(2, 3))                               # (NMAX+1, n) averaged over nodes, gamma
    out = []
    for L in lines:
        idx = np.clip(np.floor(L).astype(int), 0, NMAX)
        out.append(1 - cdf[idx, np.arange(n)])
    return out


def evaluate(P, cfg, k10, g, ok_only=True, skew=False):
    """gamma night factor (shape k10 x (mean/10)^g), or with skew=True a skew-normal one: k10 = relative sd, g = shape."""
    D = P['D']; mu = T.means(P, cfg)['pts']; ok = (D.n_season >= 5).values
    meanN, m = mu[0][ok], mu[1][ok]; fr = [f[ok] for f in mix_parts(P, cfg)]
    b = D.b_pts.values[ok]; a = D.a_pts.values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED['pts']]
    k = ('skew', k10, g) if skew else np.maximum(k10 * (np.maximum(m, 0.5) / 10) ** g, 0.5)
    pr = sf_points(meanN, fr, k, lines)
    y = [(a > L).astype(float) for L in lines]
    allll = T.ll(np.concatenate(y), np.concatenate(pr)); book = T.ll(y[0], pr[0])
    qs = np.quantile(b, [0, .2, .4, .6, .8, 1]); tiers = []
    for i in range(5):
        sel = (b >= qs[i]) & ((b <= qs[i + 1]) if i == 4 else (b < qs[i + 1]))
        tiers.append([round(float(b[sel].mean()), 2), round(float(m[sel].mean()), 2), round(float(a[sel].mean()), 2),
                      round(float(pr[0][sel].mean()), 3), round(float(y[0][sel].mean()), 3)])
    return {'ll': round(allll, 5), 'book_ll': round(book, 5), 'k10': k10, 'g': g, 'skew': skew, 'tiers': tiers}


def run(mode, Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl')
    T3 = json.load(open('validation/props3_tuned.json')); cfg = T3['cfg']
    nb = T.score(P, T.means(P, cfg), 'pts', phi=tuple(T3['dispersion']['pts']['f']), mode='pow')
    print(f'{Y} negative binomial (current, step 3): all lines {nb["ll"]:.5f} book {nb["book_ll"]:.5f}')
    print('   tiers [season avg, model mean, actual mean, pred over, actual over]: ' + ' | '.join(str(t) for t in nb['tiers']))
    if mode == 'tune':
        res = []
        for k10 in (3, 5, 8, 12, 20, 40):
            for g in (0.0, 0.5, 1.0, 1.5):
                r = evaluate(P, cfg, k10, g); res.append(r)
                print(f'   scoring plays k10={k10:>3} g={g}: all {r["ll"]:.5f} book {r["book_ll"]:.5f} top tier over {r["tiers"][-1][3]}', flush=True)
        for sig in (0.15, 0.22, 0.3, 0.38):
            for a in (0.0, -2.0, -4.0, -8.0):
                r = evaluate(P, cfg, sig, a, skew=True); res.append(r)
                print(f'   scoring plays, skewed night factor sd={sig} shape={a}: all {r["ll"]:.5f} book {r["book_ll"]:.5f} top tier over {r["tiers"][-1][3]}', flush=True)
        best = min(res, key=lambda r: r['ll'])
        json.dump({'tuned_on': Y, 'k10': best['k10'], 'g': best['g'], 'nb_step3': nb, 'best': best, 'grid': res},
                  open('validation/points_shape_tuned.json', 'w'), indent=1)
    else:
        t = json.load(open('validation/points_shape_tuned.json')); best = evaluate(P, cfg, t['k10'], t['g'], skew=t['best'].get('skew', False))
        json.dump({'season': Y, 'nb_step3': nb, 'scoring_plays': best}, open(f'validation/points_shape_test_{Y}.json', 'w'), indent=1)
    print(f'{Y} scoring plays (k10={best["k10"]}, g={best["g"]}, skew={best["skew"]}): all lines {best["ll"]:.5f} book {best["book_ll"]:.5f}')
    print('   tiers [season avg, model mean, actual mean, pred over, actual over]: ' + ' | '.join(str(t) for t in best['tiers']))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
