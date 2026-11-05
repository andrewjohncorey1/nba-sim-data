"""
Props step 4: price every prop over the player's minutes distribution (validation/minutes_dist.py).

For each player-game: band = his projected-minutes role; r = 11 quantiles of that band's minutes ratio (fit on
earlier seasons only); expected minutes = projection x E[r] (enters the team shares); on a night with ratio r his
stat's mean is mean x (r/E[r])^beta / E[(r/E[r])^beta], so the mean is unchanged and beta > 1 lets scoring fall
faster than minutes on short nights (measured: 0.43 points a minute below 0.6x his minutes vs 0.56-0.59 normally).
Given the night, negative binomial with dispersion phi = phi10 x (conditional mean / 10)^g, mixed over the
simulated team distribution (5 nodes) as before. beta and the dispersion are tuned per market on 2024-25.

Usage (repo root): python3 validation/props_minutes.py tune 2025 | test 2026
"""
import sys, json, time
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import nbinom
import tune_props2 as T

QI = list(range(0, 21, 2))            # 11 of the 21 stored quantiles
NI = [1, 4, 7, 10, 13]                # 5 of the 15 team-distribution nodes
BETAS = (1.0, 1.15, 1.3, 1.5)
DGRID = [(p, g) for p in (4, 6, 9, 14, 20, 30, 45) for g in (0.5, 1.0, 1.5)]


def setup(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl')
    T3 = json.load(open('validation/props3_tuned.json')); cfg = T3['cfg']
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json'))
    alloc = T.project_minutes(P, cfg['min'])
    band = np.clip(np.searchsorted(MD['edges'], alloc, side='right') - 1, 0, len(MD['edges']) - 2)
    rq = np.array(MD['q'])[band][:, QI]; Er = rq.mean(1)
    mu = T.means(P, cfg, alloc=alloc * Er)
    return P, T3, cfg, rq / Er[:, None], mu


def price(nodesM, m, rel, beta, f, L):
    """P(X > L): mean over team nodes (n, 5) and minutes quantiles (n, 11)."""
    fac = rel ** beta; fac = fac / fac.mean(1, keepdims=True)
    M = np.maximum(nodesM[:, :, None] * fac[:, None, :], 1e-6)                     # (n, 5, 11)
    ph = np.maximum(f[0] * (M / 10) ** f[1], 0.3)
    return nbinom.sf(np.floor(L)[:, None, None], ph, ph / (ph + M)).mean(axis=(1, 2))


def evaluate(P, mu, rel, k, beta, f):
    D = P['D']; ok = (D.n_season >= 5).values
    nodesM, m, r = mu[k][0][ok][:, NI], mu[k][1][ok], rel[ok]
    b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED[k]]
    pr = [price(nodesM, m, r, beta, f, L) for L in lines]; y = [(a > L).astype(float) for L in lines]
    qs = np.quantile(b, [0, .2, .4, .6, .8, 1]); tiers = []
    for i in range(5):
        sel = (b >= qs[i]) & ((b <= qs[i + 1]) if i == 4 else (b < qs[i + 1]))
        tiers.append([round(float(b[sel].mean()), 2), round(float(m[sel].mean()), 2), round(float(a[sel].mean()), 2),
                      round(float(pr[0][sel].mean()), 3), round(float(y[0][sel].mean()), 3)])
    return {'ll': round(T.ll(np.concatenate(y), np.concatenate(pr)), 5), 'book_ll': round(T.ll(y[0], pr[0]), 5),
            'beta': beta, 'f': list(f), 'tiers': tiers}


def step3(P, T3, k):
    """The step-3 model on the same rows (no minutes mixture), for comparison."""
    d = T3['dispersion'][k]; mu = T.means(P, T3['cfg'])
    return T.score(P, mu, k, phi=tuple(d['f']) if d['mode'] == 'pow' else d.get('c') or d.get('phi'), mode=d['mode'])


def run(mode, Y):
    P, T3, cfg, rel, mu = setup(Y)
    out = {'season': Y, 'markets': {}}
    chosen = json.load(open('validation/props4_tuned.json'))['choice'] if mode == 'test' else {}
    for k in T.MK:
        t0 = time.time(); s3 = step3(P, T3, k)
        if mode == 'tune':
            best = None
            for beta in BETAS:
                for f in DGRID:
                    r = evaluate(P, mu, rel, k, beta, f)
                    if best is None or r['ll'] < best['ll']: best = r
            chosen[k] = {'beta': best['beta'], 'f': best['f']}
        else:
            best = evaluate(P, mu, rel, k, chosen[k]['beta'], tuple(chosen[k]['f']))
        out['markets'][k] = {'step3': {kk: s3[kk] for kk in ('ll', 'book_ll', 'tiers')}, 'step4': best}
        print(f'{k:5s} step 3 all {s3["ll"]:.5f} book {s3["book_ll"]:.5f} | minutes model (beta {best["beta"]}, phi10/g {best["f"]}) '
              f'all {best["ll"]:.5f} book {best["book_ll"]:.5f}   [{time.time() - t0:.0f}s]', flush=True)
        print('      top tier [season avg, mean, actual, pred over, actual over]  step 3 ' + str(s3['tiers'][-1]) + '  step 4 ' + str(best['tiers'][-1]))
        print('      all tiers pred/actual over, step 4: ' + ' | '.join(f'{t[3]}/{t[4]}' for t in best['tiers']), flush=True)
    if mode == 'tune':
        json.dump({'tuned_on': Y, 'choice': chosen, 'results': out}, open('validation/props4_tuned.json', 'w'), indent=1)
    else:
        json.dump(out, open(f'validation/props4_test_{Y}.json', 'w'), indent=1)


def cp_price(P, cfg, mu, rel, beta, lines, ok):
    """Points as scoring plays given the night: compound Poisson (1s, 2s, 3s in the player's own mix), mixed over
    team nodes and his minutes quantiles."""
    import points_shape as PSH
    nodesM = mu['pts'][0][ok][:, NI]; r = rel[ok]
    fac = r ** beta; fac = fac / fac.mean(1, keepdims=True)
    M = (nodesM[:, :, None] * fac[:, None, :]).reshape(len(r), -1)
    fr = [f[ok] for f in PSH.mix_parts(P, cfg)]
    return PSH.sf_points(M, fr, np.full(len(r), 1e6), lines)


def tiers_of(b, m, a, pr0, y0):
    qs = np.quantile(b, [0, .2, .4, .6, .8, 1]); out = []
    for i in range(5):
        sel = (b >= qs[i]) & ((b <= qs[i + 1]) if i == 4 else (b < qs[i + 1]))
        out.append([round(float(b[sel].mean()), 2), round(float(m[sel].mean()), 2), round(float(a[sel].mean()), 2),
                    round(float(pr0[sel].mean()), 3), round(float(y0[sel].mean()), 3)])
    return out


def build(P, cfg, MD, er_mode):
    """Minutes stage: projection -> fitted to 240 (cfg aexp) -> expected minutes (x E[r] for none / rotation (24+) / all)."""
    alloc = T.project_minutes(P, cfg['min'])
    band = np.clip(np.searchsorted(MD['edges'], alloc, side='right') - 1, 0, len(MD['edges']) - 2)
    rq = np.array(MD['q'])[band][:, QI]; Er = rq.mean(1)
    mult = {'none': np.ones_like(Er), 'rotation': np.where(alloc >= 24, Er, 1.0), 'all': Er}[er_mode]
    mu = T.means(P, cfg, alloc=alloc * mult)
    return mu, rq / Er[:, None]


def obj_points(P, cfg, MD, er_mode, beta):
    D = P['D']; ok = (D.n_season >= 5).values
    mu, rel = build(P, cfg, MD, er_mode)
    b = D.b_pts.values[ok]; a = D.a_pts.values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED['pts']]
    pr = cp_price(P, cfg, mu, rel, beta, lines, ok); y = [(a > L).astype(float) for L in lines]
    allll, book = T.ll(np.concatenate(y), np.concatenate(pr)), T.ll(y[0], pr[0])
    return {'obj': allll + book, 'll': round(allll, 5), 'book_ll': round(book, 5), 'tiers': tiers_of(b, mu['pts'][1][ok], a, pr[0], y[0])}


def tune2(Y):
    """Final pass on 2024-25 only. Objective = all-lines + book-line log loss. (1) minutes stage and points shape
    (scoring plays given minutes, power beta) on points; (2) for assists, threes and PRA, beta and NB dispersion given
    the night, with that minutes stage."""
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl')
    T3 = json.load(open('validation/props3_tuned.json')); MD = json.load(open(f'validation/minutes_dist_for_{Y}.json'))
    base = json.loads(json.dumps(T3['cfg']))
    res = []
    grid = [(a, pw, tg) for a in (3, 1, 0) for pw in (0.25, 0.5, 1.0) for tg in ('14', 'season')] + [(a, 0, '14') for a in (3, 1, 0)]
    for aexp, pw, tg in grid:
        for er, beta in (('rotation', 1.15),):
            cfg = json.loads(json.dumps(base)); cfg['min'] = {**cfg['min'], 'pw': pw, 'target': tg, 'aexp': aexp}
            r = obj_points(P, cfg, MD, er, beta); r.update(aexp=aexp, pw=pw, target=tg, er=er, beta=beta); res.append(r)
            print(f'  points  leftover-minutes spread {aexp} prior {pw} toward {tg:6s} E[r] {er}: all {r["ll"]:.5f} book {r["book_ll"]:.5f}  '
                  f'top tier mean {r["tiers"][-1][1]}/{r["tiers"][-1][2]} over {r["tiers"][-1][3]}/{r["tiers"][-1][4]}  '
                  f'all tiers over ' + ' | '.join(f'{t[3]}/{t[4]}' for t in r['tiers']), flush=True)
    best = min(res, key=lambda r: r['obj'])
    for er, beta in (('none', 1.15), ('rotation', 1.3), ('none', 1.3)):        # check E[r] and beta at the chosen minutes
        cfg = json.loads(json.dumps(base)); cfg['min'] = {**cfg['min'], 'pw': best['pw'], 'target': best['target'], 'aexp': best['aexp']}
        r = obj_points(P, cfg, MD, er, beta); r.update(aexp=best['aexp'], pw=best['pw'], target=best['target'], er=er, beta=beta); res.append(r)
        print(f'  points  check E[r] {er} beta {beta}: all {r["ll"]:.5f} book {r["book_ll"]:.5f}', flush=True)
    best = min(res, key=lambda r: r['obj'])
    cfg = json.loads(json.dumps(base)); cfg['min'] = {**cfg['min'], 'pw': best['pw'], 'target': best['target'], 'aexp': best['aexp']}
    choice = {'minutes': {'aexp': best['aexp'], 'pw': best['pw'], 'target': best['target'], 'er': best['er']}, 'pts': {'shape': 'scoring plays', 'beta': best['beta']}}
    D = P['D']; ok = (D.n_season >= 5).values
    mu, rel = build(P, cfg, MD, best['er'])
    other = {}
    for k in ('ast', 'fg3m', 'pra', 'reb'):
        b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
        lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED[k]]; y = [(a > L).astype(float) for L in lines]
        bb = None
        for beta in (1.0, 1.15, 1.3):
            for f in [(p, g) for p in (6, 9, 14, 20, 30, 45, 80) for g in (0.5, 1.0, 1.5)]:
                pr = [price(mu[k][0][ok][:, NI], mu[k][1][ok], rel[ok], beta, f, L) for L in lines]
                al, bk = T.ll(np.concatenate(y), np.concatenate(pr)), T.ll(y[0], pr[0])
                if bb is None or al + bk < bb[0]: bb = (al + bk, beta, f, al, bk, tiers_of(b, mu[k][1][ok], a, pr[0], y[0]))
        choice[k] = {'shape': 'nb', 'beta': bb[1], 'f': list(bb[2])}
        other[k] = {'ll': round(bb[3], 5), 'book_ll': round(bb[4], 5), 'tiers': bb[5]}
        print(f'  {k:5s} beta {bb[1]} phi10/g {bb[2]}: all {bb[3]:.5f} book {bb[4]:.5f}  tiers over ' + ' | '.join(f'{t[3]}/{t[4]}' for t in bb[5]), flush=True)
    json.dump({'tuned_on': Y, 'cfg': cfg, 'choice': choice, 'points': best, 'other': other, 'grid': res}, open('validation/props5_tuned.json', 'w'), indent=1)
    print('chosen', json.dumps(choice))


def test2(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); T5 = json.load(open('validation/props5_tuned.json'))
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json')); T3 = json.load(open('validation/props3_tuned.json'))
    cfg, ch = T5['cfg'], T5['choice']; D = P['D']; ok = (D.n_season >= 5).values
    mu, rel = build(P, cfg, MD, ch['minutes']['er'])
    out = {'season': Y, 'choice': ch, 'markets': {}}
    for k in ('pts', 'ast', 'fg3m', 'pra', 'reb'):
        s3 = step3(P, T3, k)
        b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
        lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED[k]]; y = [(a > L).astype(float) for L in lines]
        if k == 'pts': pr = cp_price(P, cfg, mu, rel, ch['pts']['beta'], lines, ok)
        else: pr = [price(mu[k][0][ok][:, NI], mu[k][1][ok], rel[ok], ch[k]['beta'], tuple(ch[k]['f']), L) for L in lines]
        r = {'ll': round(T.ll(np.concatenate(y), np.concatenate(pr)), 5), 'book_ll': round(T.ll(y[0], pr[0]), 5),
             'tiers': tiers_of(b, mu[k][1][ok], a, pr[0], y[0])}
        out['markets'][k] = {'step3': {kk: s3[kk] for kk in ('ll', 'book_ll', 'tiers')}, 'step5': r}
        print(f'{k:5s} step 3 all {s3["ll"]:.5f} book {s3["book_ll"]:.5f} | minutes model all {r["ll"]:.5f} book {r["book_ll"]:.5f}')
        print('      tiers [season avg, mean, actual, pred over, actual over]')
        print('      step 3  ' + ' | '.join(str(t) for t in s3['tiers'])); print('      minutes ' + ' | '.join(str(t) for t in r['tiers']))
    json.dump(out, open(f'validation/props5_test_{Y}.json', 'w'), indent=1)


if __name__ == '__main__':
    {'tune2': tune2, 'test2': test2}.get(sys.argv[1], lambda y: run(sys.argv[1], y))(int(sys.argv[2]))
