"""Props step 2 of the SGP plan: rebounds. The simulation counts every rebound, including team rebounds (out of
bounds, end of period) that box scores credit to no player, so player rebound shares were scaled by a total about 12%
too big. rmult (share of simulated rebounds credited to players) is tuned on 2024-25 together with the rebound and
PRA dispersion (minutes model as in validation/props5_tuned.json), then tested on 2025-26.
Usage (repo root): python3 validation/props_reb.py tune 2025 | test 2026"""
import sys, json
sys.path.insert(0, 'validation')
import numpy as np, pandas as pd
import props_minutes as PM, tune_props2 as T

GRIDF = [(p, g) for p in (6, 9, 14, 20, 30, 45, 80, 1000) for g in (0.5, 1.0)]


_CACHE = {}


def evalk(P, cfg, MD, k, beta, f, sel_min=None):
    D = P['D']; ok = (D.n_season >= 5).values
    key = json.dumps(cfg, sort_keys=True)
    if key not in _CACHE: _CACHE.clear(); _CACHE[key] = PM.build(P, cfg, MD, 'rotation')
    mu, rel = _CACHE[key]
    b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED[k]]; y = [(a > L).astype(float) for L in lines]
    pr = [PM.price(mu[k][0][ok][:, PM.NI], mu[k][1][ok], rel[ok], beta, f, L) for L in lines]
    res = {'ll': round(T.ll(np.concatenate(y), np.concatenate(pr)), 5), 'book_ll': round(T.ll(y[0], pr[0]), 5),
           'tiers': PM.tiers_of(b, mu[k][1][ok], a, pr[0], y[0])}
    if sel_min:
        s = T.project_minutes(P, cfg['min'])[ok] >= sel_min
        res['tiers_18'] = PM.tiers_of(b[s], mu[k][1][ok][s], a[s], pr[0][s], y[0][s])
    return res


def tune(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); T5 = json.load(open('validation/props5_tuned.json'))
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json'))
    best = None
    for rm in (1.0, 0.85, 0.87, 0.89, 0.91):
        cfg = json.loads(json.dumps(T5['cfg'])); cfg['rmult'] = rm
        out = {}
        for k in ('reb', 'pra'):
            bb = None
            for beta in (1.0, 1.15):
                for f in GRIDF:
                    r = evalk(P, cfg, MD, k, beta, f)
                    if bb is None or r['ll'] + r['book_ll'] < bb[0]: bb = (r['ll'] + r['book_ll'], beta, f, r)
            out[k] = bb
        tot = out['reb'][0] + out['pra'][0]
        print(f'rebound multiplier {rm}: reb all {out["reb"][3]["ll"]:.5f} book {out["reb"][3]["book_ll"]:.5f} (beta {out["reb"][1]}, f {out["reb"][2]}) tiers '
              + ' | '.join(f'{t[3]}/{t[4]}' for t in out['reb'][3]['tiers']) + f';  pra all {out["pra"][3]["ll"]:.5f} book {out["pra"][3]["book_ll"]:.5f} '
              f'(beta {out["pra"][1]}, f {out["pra"][2]}) tiers ' + ' | '.join(f'{t[3]}/{t[4]}' for t in out['pra'][3]['tiers']), flush=True)
        if best is None or tot < best[0]: best = (tot, rm, out)
    T5['cfg']['rmult'] = best[1]
    for k in ('reb', 'pra'): T5['choice'][k] = {'shape': 'nb', 'beta': best[2][k][1], 'f': list(best[2][k][2])}
    T5['reb_note'] = f'rmult {best[1]} chosen on 2024-25 (validation/props_reb_tune_2025.log)'
    json.dump(T5, open('validation/props5_tuned.json', 'w'), indent=1)
    print('chosen rmult', best[1], {k: T5['choice'][k] for k in ('reb', 'pra')})


def test(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); T5 = json.load(open('validation/props5_tuned.json'))
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json')); out = {}
    old = json.loads(json.dumps(T5['cfg'])); old['rmult'] = 1.0
    for k in ('reb', 'pra'):
        c = T5['choice'][k]
        r = evalk(P, T5['cfg'], MD, k, c['beta'], tuple(c['f']), sel_min=18)
        print(f'{k}: all {r["ll"]:.5f} book {r["book_ll"]:.5f}')
        print('   all players [season avg, mean, actual, pred over, actual over]: ' + ' | '.join(str(t) for t in r['tiers']))
        print('   18+ minutes: ' + ' | '.join(str(t) for t in r['tiers_18']))
        out[k] = r
    json.dump(out, open(f'validation/props_reb_test_{Y}.json', 'w'), indent=1)


if __name__ == '__main__':
    {'tune': tune, 'test': test}[sys.argv[1]](int(sys.argv[2]))
