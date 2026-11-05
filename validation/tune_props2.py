"""
Fast tuning harness for the player-prop means (minutes projection and per-minute rates), built to fix the
compression found on Nov 5, 2026 (stars projected too low, bench too high).

Usage (repo root):
  python3 validation/tune_props2.py prep <season>           precompute each player-game's history (once per season)
  python3 validation/tune_props2.py tune 2025               coordinate search on 2024-25 only -> validation/props2_tuned.json
  python3 validation/tune_props2.py test 2026               old vs tuned settings on 2025-26 (untouched)

Same rules as validation/props.py: every player-game uses only earlier games; the players are those who played;
minutes fitted to the team's 240 (players.allocate); a player's mean = his share of the team's expected rate x
minutes, times the simulated team quantity (points; made field goals for assists; threes; rebounds), priced with a
negative binomial mixed over the simulated team distribution. For speed the team distribution is reduced to 15
equal-weight quantile nodes; the final numbers are re-run through validation/props.py.

Settings searched
  minutes: half-life in games (hl), prior weight in games (pw), prior target: 14 minutes (old) or the player's own
           season-to-date average minutes (falls back to last season's, then 14)
  rates:   half-life in days, shrinkage minutes (K), shrinkage target: position average (old) or the player's own
           long-run rate (half-life 720 days, itself shrunk to the position average with 300 minutes)
Objective: log loss over book-style + fixed lines (as in validation/props_final.py 'all'), dispersion re-chosen per
setting from the same grid. Minutes are shared by every market and tuned on the sum over markets; rate settings are
tuned per market.
"""
import sys, os, json, itertools, time
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import nbinom
import players as PS

HLS = (90, 180, 365, 720)
STATS = ('pts', 'reb', 'ast', 'fg3m', 'fgm')
MK = ('pts', 'reb', 'ast', 'fg3m', 'pra')
GRID = [3, 5, 8, 12, 20, 40]
FIXED = {'pts': (9.5, 14.5, 19.5, 24.5, 29.5), 'reb': (3.5, 5.5, 7.5, 9.5), 'ast': (1.5, 3.5, 5.5, 7.5),
         'fg3m': (0.5, 1.5, 2.5, 3.5), 'pra': (14.5, 19.5, 24.5, 29.5, 34.5)}
NODES = 15
OLD = {'min': {'hl': 8, 'pw': 3, 'target': '14'},       # pw is players.projected_minutes' prior_games
       'rate': {k: {'hl': 180, 'K': 300, 'target': 'pos'} for k in ('pts', 'reb', 'ast', 'fg3m', 'fgm')}}


def prep(Y):
    sims = {s['game_id']: s for s in json.load(open(f'validation/sim_{Y}_base.json'))}
    PL = PS.load(range(Y - 2, Y + 1))
    PL['posg'] = PL.pos.fillna('F').str[0].map({'G': 'G', 'F': 'F', 'C': 'C'}).fillna('F')
    g = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id')
    cur = PL[(PL.season == Y) & (PL['min'] > 0) & PL.game_id.isin(sims)].copy()
    out, mins_arr, t0 = [], [], time.time()
    for d, gd in cur.groupby('date'):
        hist = PL[PL.date < d]; x = hist[hist['min'] > 0]
        age = (pd.Timestamp(d) - x.date).dt.days.values.astype(float)
        F = pd.DataFrame(index=gd.pid.unique())
        for h in HLS:
            w = 0.5 ** (age / h); keep = w > 1e-3
            xx = x[keep]; ww = w[keep]
            agg = pd.DataFrame({'wm': ww * xx['min'].values, **{f'w{s}': ww * xx[s].values for s in STATS}, 'pid': xx.pid.values}).groupby('pid').sum()
            F = F.join(agg.add_suffix(f'_{h}'))
        w = 0.5 ** (age / 180); keep = w > 1e-3; xx = x[keep]; ww = w[keep]
        pa = {k: {s: float((ww[m] * xx[s].values[m]).sum() / (ww[m] * xx['min'].values[m]).sum()) for s in STATS}
              for k in ('G', 'F', 'C') for m in [(xx.posg == k).values]}
        pos = xx.sort_values('date').groupby('pid').posg.last()
        # minutes history: last 30 played games, most recent first
        last = x.sort_values('date').groupby('pid').tail(30)
        last = last.assign(k=last.groupby('pid').cumcount(ascending=False))
        M = last.pivot(index='pid', columns='k', values='min').reindex(columns=range(30))
        smin = hist[(hist.season == Y) & (hist['min'] > 0)].groupby('pid')['min'].mean()
        lmin = hist[(hist.season == Y - 1) & (hist['min'] > 0)].groupby('pid')['min'].mean()
        sh = hist[(hist.season == Y) & (hist['min'] > 0)].groupby('pid')[['pts', 'reb', 'ast', 'fg3m']].agg(['mean', 'count'])
        for r in gd.itertuples():
            p = r.pid; side = 'h' if g.loc[r.game_id, 'home'] == r.team else 'a'
            pg = pos.get(p, r.posg)
            row = dict(date=d, game_id=r.game_id, team=r.team, side=side, pid=p, posg=pg,
                       a_pts=r.pts, a_reb=r.reb, a_ast=r.ast, a_fg3m=r.fg3m, smin=smin.get(p, np.nan), lmin=lmin.get(p, np.nan),
                       n_season=int(sh.loc[p, ('pts', 'count')]) if p in sh.index else 0)
            for s in ('pts', 'reb', 'ast', 'fg3m'):
                row[f'b_{s}'] = float(sh.loc[p, (s, 'mean')]) if p in sh.index else np.nan
            for s in STATS: row[f'pa_{s}'] = pa[pg][s]
            for h in HLS:
                for c in ['wm'] + [f'w{s}' for s in STATS]:
                    v = F.at[p, f'{c}_{h}'] if f'{c}_{h}' in F.columns else np.nan
                    row[f'{c}_{h}'] = 0.0 if pd.isna(v) else float(v)
            out.append(row)
            mins_arr.append(M.loc[p].values.astype(float) if p in M.index else np.full(30, np.nan))
    D = pd.DataFrame(out); D['b_pra'] = D.b_pts + D.b_reb + D.b_ast; D['a_pra'] = D.a_pts + D.a_reb + D.a_ast
    # team distribution nodes per game side: points, made field goals (assists), threes, rebounds
    keys = {'pts': ('p', 60), 'fgm': ('fg', 20), 'fg3m': ('f3', 0), 'reb': ('rb', 20)}
    nodes = {}
    for gid, s in sims.items():
        for side in 'ha':
            for k, (pre, lo) in keys.items():
                pmf = np.asarray(s[pre + side]); cdf = np.cumsum(pmf); vals = np.arange(lo, lo + len(pmf))
                q = (np.arange(NODES) + 0.5) / NODES
                nodes[(gid, side, k)] = vals[np.minimum(np.searchsorted(cdf, q), len(vals) - 1)].astype(float)
                nodes[(gid, side, k + '_mean')] = float(np.dot(vals, pmf))
    pd.to_pickle({'D': D, 'M': np.array(mins_arr), 'nodes': nodes}, f'validation/props2_pre_{Y}.pkl')
    print(f'prep {Y}: {len(D)} player-games, {D.date.nunique()} dates, {time.time() - t0:.0f}s')


def project_minutes(P, cfg):
    M = P['M']; D = P['D']
    k = np.arange(30); w = 0.5 ** (k / cfg['hl'])
    have = ~np.isnan(M)
    sw = (w * have).sum(1); swm = np.nansum(w * np.nan_to_num(M), 1)
    if cfg['target'] == '14': tgt = np.full(len(D), 14.0)
    else: tgt = D.smin.fillna(D.lmin).fillna(14.0).values
    pw = cfg['pw'] * 0.5                                # same 0.5 factor as players.projected_minutes
    proj = (swm + tgt * pw) / np.maximum(sw + pw, 1e-9)
    proj[sw == 0] = 12.0                                # no history: 12 minutes, as before
    alloc = np.empty(len(D))
    for _, ix in D.groupby(['game_id', 'team']).indices.items():
        alloc[ix] = PS.allocate(proj[ix])
    return alloc


def rates(D, s, cfg):
    h = cfg['hl']; wm, ws, pa = D[f'wm_{h}'].values, D[f'w{s}_{h}'].values, D[f'pa_{s}'].values
    if cfg['target'] == 'pos': tgt = pa
    else:
        tgt = (D[f'w{s}_720'].values + 300 * pa) / (D['wm_720'].values + 300)
    r = (ws + cfg['K'] * tgt) / np.maximum(wm + cfg['K'], 1e-9)
    return np.where(wm > 0, r, pa)


def means(P, cfg, alloc=None):
    D = P['D']; nodes = P['nodes']
    if alloc is None: alloc = project_minutes(P, cfg['min'])
    raw = {s: rates(D, s, cfg['rate'][s]) * alloc for s in STATS}
    grp = D.groupby(['game_id', 'team']).ngroup().values
    tot = {s: np.bincount(grp, weights=raw[s]) for s in STATS}
    share = {'pts': raw['pts'] / tot['pts'][grp], 'reb': raw['reb'] / tot['reb'][grp],
             'ast': raw['ast'] / tot['fgm'][grp], 'fg3m': raw['fg3m'] / tot['fg3m'][grp]}
    key = {'pts': 'pts', 'reb': 'reb', 'ast': 'fgm', 'fg3m': 'fg3m'}
    gs = list(zip(D.game_id.values, D.side.values))
    out = {}
    for k in ('pts', 'reb', 'ast', 'fg3m'):
        T = np.array([nodes[(g, sd, key[k])] for g, sd in gs]); ET = np.array([nodes[(g, sd, key[k] + '_mean')] for g, sd in gs])
        out[k] = (share[k][:, None] * T, share[k] * ET)
    Tp = np.array([nodes[(g, sd, 'pts')] for g, sd in gs]); ETp = np.array([nodes[(g, sd, 'pts_mean')] for g, sd in gs])
    mp = out['pts'][1] + out['reb'][1] + out['ast'][1]
    out['pra'] = ((mp / ETp)[:, None] * Tp, mp)
    return out


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


PGRID = [0.25, 0.4, 0.6, 0.9, 1.4, 2.0, 3.0]      # dispersion proportional to the player's mean: phi = c x mean


def nb_mix(L, nodesM, f, m, mode):
    ph = f if mode == 'fixed' else np.maximum(f * m, 0.3)[:, None]
    return nbinom.sf(np.floor(L)[:, None], ph, ph / (ph + np.maximum(nodesM, 1e-6))).mean(1)


def score(P, mu, k, phi=None, mode='fixed'):
    """log loss over book + fixed lines for market k (players with 5+ season games), best dispersion from the grid.
    mode 'fixed': one dispersion for everyone (old); 'prop': dispersion = c x the player's mean, so the variance grows
    in proportion to the mean (var = m (1 + 1/c)) instead of with its square."""
    D = P['D']; ok = (D.n_season >= 5).values
    nodesM, m = mu[k][0][ok], mu[k][1][ok]
    b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in FIXED[k]]
    res = {}
    for f in ([phi] if phi else (GRID if mode == 'fixed' else PGRID)):
        ys, ps = [], []
        for L in lines:
            ys.append((a > L).astype(float)); ps.append(nb_mix(L, nodesM, f, m, mode))
        res[f] = ll(np.concatenate(ys), np.concatenate(ps))
    f = min(res, key=res.get)
    # calibration by season-average tier on the book line
    L = lines[0]; pr = nb_mix(L, nodesM, f, m, mode)
    q = np.quantile(b, [0, .2, .4, .6, .8, 1])
    tiers = []
    for i in range(5):
        sel = (b >= q[i]) & (b <= q[i + 1]) if i == 4 else (b >= q[i]) & (b < q[i + 1])
        tiers.append([round(float(b[sel].mean()), 2), round(float(m[sel].mean()), 2), round(float(a[sel].mean()), 2),
                      round(float(pr[sel].mean()), 3), round(float((a[sel] > L[sel]).mean()), 3)])
    return {'ll': round(res[f], 5), 'phi': f, 'mode': mode, 'book_ll': round(ll((a > L).astype(float), pr), 5),
            'mae': round(float(np.mean(np.abs(a - m))), 3), 'tiers': tiers}


def total_score(P, cfg, alloc=None):
    mu = means(P, cfg, alloc)
    return {k: score(P, mu, k) for k in MK}


def tune(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl')
    cfg = json.loads(json.dumps(OLD)); log = []
    def note(tag, c, s):
        line = {'step': tag, 'cfg': json.loads(json.dumps(c)), 'll': {k: s[k]['ll'] for k in MK}, 'sum': round(sum(s[k]['ll'] for k in MK), 5)}
        log.append(line); print(json.dumps(line), flush=True)
    base = total_score(P, cfg); note('old', cfg, base)
    MINGRID = [{'hl': hl, 'pw': 0, 'target': '14'} for hl in (4, 8, 15)] + \
              [{'hl': hl, 'pw': pw, 'target': t} for hl in (4, 8, 15) for pw in (0.5, 1.5, 4) for t in ('14', 'season')]
    RATEGRID = [{'hl': hl, 'K': 0, 'target': 'pos'} for hl in (90, 180, 365)] + \
               [{'hl': hl, 'K': K, 'target': t} for hl in (90, 180, 365) for K in (50, 150, 300) for t in ('pos', 'own')]
    for rnd in range(2):
        best = None
        for mc in MINGRID:
            c = {**cfg, 'min': mc}; s = total_score(P, c); tot = sum(s[k]['ll'] for k in MK)
            if best is None or tot < best[0]: best = (tot, mc)
        cfg['min'] = best[1]; note(f'minutes round {rnd}', cfg, total_score(P, cfg))
        alloc = project_minutes(P, cfg['min'])
        # rate settings per underlying stat; fgm (assist denominator) follows assists
        for s, mkts in (('pts', ['pts', 'pra']), ('reb', ['reb', 'pra']), ('ast', ['ast', 'pra']), ('fg3m', ['fg3m'])):
            best = None
            for rc in RATEGRID:
                c = json.loads(json.dumps(cfg)); c['rate'][s] = rc
                if s == 'ast': c['rate']['fgm'] = rc
                mu = means(P, c, alloc); tot = sum(score(P, mu, k)['ll'] for k in mkts)
                if best is None or tot < best[0]: best = (tot, rc)
            cfg['rate'][s] = best[1]
            if s == 'ast': cfg['rate']['fgm'] = best[1]
        note(f'rates round {rnd}', cfg, total_score(P, cfg))
    final = total_score(P, cfg)
    json.dump({'tuned_on': Y, 'cfg': cfg, 'old': base, 'tuned': final, 'log': log}, open('validation/props2_tuned.json', 'w'), indent=1)
    report(Y, base, final)


def report(Y, base, new):
    print(f'\nseason {Y}: market, log loss old -> new (dispersion), book-line log loss old -> new, MAE old -> new')
    for k in MK:
        print(f'  {k:5s} {base[k]["ll"]:.5f} -> {new[k]["ll"]:.5f} (phi {base[k]["phi"]} -> {new[k]["phi"]})   book {base[k]["book_ll"]:.5f} -> {new[k]["book_ll"]:.5f}   '
              f'MAE {base[k]["mae"]:.3f} -> {new[k]["mae"]:.3f}')
        print('        tiers [season avg, model mean, actual mean, pred over, actual over]')
        print('        old ' + ' | '.join(str(t) for t in base[k]['tiers']))
        print('        new ' + ' | '.join(str(t) for t in new[k]['tiers']))


def test(Y):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl')
    T = json.load(open('validation/props2_tuned.json'))
    base = total_score(P, OLD)
    mu = means(P, T['cfg']); disp = T.get('dispersion', {})
    new = {k: score(P, mu, k, phi=(disp[k].get('c') or disp[k].get('phi')) if k in disp else None,
                    mode=disp[k]['mode'] if k in disp else 'fixed') for k in MK}
    for k in MK: base[k] = score(P, means(P, OLD), k, phi=base[k]['phi'])
    json.dump({'season': Y, 'cfg': T['cfg'], 'old': base, 'tuned': new}, open(f'validation/props2_test_{Y}.json', 'w'), indent=1)
    report(Y, base, new)


if __name__ == '__main__':
    {'prep': prep, 'tune': tune, 'test': test}[sys.argv[1]](int(sys.argv[2]))
