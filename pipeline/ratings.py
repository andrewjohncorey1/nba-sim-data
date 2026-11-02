"""
Team ratings from possession-level data.

Every component is a ridge-penalized model fit on per-game counts, weighted by game age (half-life in days):
  logit P = mu + home*h + Off[offense] - Def[defense]
for: turnover per opportunity (tov), free-throw trip instead of a shot (trip), three-point share of field-goal attempts
(s3), rim share of two-point attempts (s1), make rate at the rim / mid-range / three (m1, m2, m3), offensive rebound
on a missed field goal (oreb), and free-throw percentage (ft, offense only). Pace is a ridge regression on log
possessions per 48 minutes: log pace = mu + h*home? (none) + Pace[home team] + Pace[away team].
Ridge strength is set in "games of league-average data": the prior a team starts from is worth K games.

Usage as a module: fit(TG, asof, params) -> R ; probs(R, off, dfn, home) -> component probabilities ; ppp(c, E) ->
expected points per possession (closed form; engine.js expPPP is the same formula).
"""
import json, os
import numpy as np, pandas as pd

S = json.load(open('pipeline/settings.json'))
TEAMS = sorted(S['team_names'])
TI = {t: i for i, t in enumerate(TEAMS)}
NT = len(TEAMS)
COMPS = {  # name: (success column, trials expression, group)
    'tov': ('tov', 'opps', 'other'), 'trip': ('trips', 'shots', 'other'), 's3': ('fga3', 'fga', 'other'),
    's1': ('fga1', 'fga12', 'other'), 'm1': ('fgm1', 'fga1', 'make'), 'm2': ('fgm2', 'fga2', 'make'),
    'm3': ('fgm3', 'fga3', 'make'), 'oreb': ('oreb_fg', 'miss_fg', 'other'), 'ft': ('ftm', 'fta', 'ft')}


def load_tg(seasons):
    """Team-game rows (one per offense) with date, neutral flag and regulation minutes."""
    out = []
    for y in seasons:
        if not os.path.exists(f'data/derived/teamgames_{y}.parquet'): continue
        tg = pd.read_parquet(f'data/derived/teamgames_{y}.parquet')
        g = pd.read_parquet(f'data/derived/games_{y}.parquet')[['game_id', 'date', 'neutral', 'ot', 'stype', 'season']]
        tg = tg.merge(g, on='game_id')
        out.append(tg)
    tg = pd.concat(out, ignore_index=True)
    tg['fga'] = tg.fga1 + tg.fga2 + tg.fga3
    tg['fga12'] = tg.fga1 + tg.fga2
    tg['shots'] = tg.fga + tg.trips
    tg['hm'] = np.where(tg.neutral == 1, 0, tg.home)
    tg['o'] = tg.team.map(TI); tg['d'] = tg.opp.map(TI)
    tg['mins'] = 48 + 5 * tg.ot.fillna(0)
    return tg.dropna(subset=['o', 'd']).astype({'o': int, 'd': int})


def _logit_fit(k, n, w, o, d, hm, lam, two_sided=True, iters=30):
    """Weighted binomial ridge, params [mu, h, Off(NT), Def(NT)]. Rows are first collapsed to (offense, defense, home)
    cells: the weighted successes and trials are sufficient statistics. Returns (mu, h, Off, Def)."""
    P = 2 + NT * (2 if two_sided else 1)
    cell = (o * NT + d) * 2 + hm.astype(int)
    K = np.bincount(cell, weights=w * k, minlength=NT * NT * 2)
    N = np.bincount(cell, weights=w * n, minlength=NT * NT * 2)
    ix = np.nonzero(N > 0)[0]
    K, N = K[ix], N[ix]
    co, cd, ch = ix // 2 // NT, ix // 2 % NT, ix % 2
    X = np.zeros((len(ix), P)); X[:, 0] = 1; X[:, 1] = ch
    X[np.arange(len(ix)), 2 + co] = 1
    if two_sided: X[np.arange(len(ix)), 2 + NT + cd] = -1
    beta = np.zeros(P); tot = K.sum() / N.sum(); beta[0] = np.log(tot / (1 - tot))
    pen = np.full(P, lam); pen[:2] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (K - N * p) - pen * beta
        H = (X * (N * p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(P)
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-8: break
    Off = beta[2:2 + NT]; Def = beta[2 + NT:] if two_sided else np.zeros(NT)
    return float(beta[0]), float(beta[1]), Off, Def


def weights(dates, asof, half_life):
    age = (pd.Timestamp(asof) - pd.to_datetime(dates)).dt.days.values.astype(float)
    return np.where(age > 0, 0.5 ** (age / half_life), 0.0)


def fit(TG, asof, prm):
    """Ratings from games strictly before `asof`. prm: half_life, K_other, K_make, K_ft, K_pace."""
    x = TG[TG.date < pd.Timestamp(asof)]
    w = weights(x.date, asof, prm['half_life'])
    x, w = x[w > 1e-4], w[w > 1e-4]
    o, d, hm = x.o.values, x.d.values, x.hm.values.astype(float)
    R = {'asof': str(pd.Timestamp(asof).date()), 'n_games': int(x.game_id.nunique())}
    for c, (kc, nc, grp) in COMPS.items():
        k, n = x[kc].values.astype(float), x[nc].values.astype(float)
        pbar = k.sum() / n.sum(); nbar = n.mean()
        K = prm['K_' + grp]
        lam = K * nbar * pbar * (1 - pbar)
        mu, h, Off, Def = _logit_fit(k, n, w, o, d, hm, lam, two_sided=(c != 'ft'))
        R[c] = {'mu': mu, 'h': h, 'O': Off.tolist(), 'D': Def.tolist()}
    # pace: per game, log of possessions per 48 minutes (both teams averaged)
    g = x.groupby('game_id').agg(poss=('poss', 'mean'), mins=('mins', 'first'), date=('date', 'first'))
    hh = x[x.home == 1].set_index('game_id'); aa = x[x.home == 0].set_index('game_id')
    g = g.join(hh[['o']].rename(columns={'o': 'ht'})).join(aa[['o']].rename(columns={'o': 'at'})).dropna()
    yv = np.log(g.poss / g.mins * 48).values
    wg = weights(g.date, asof, prm['half_life'])
    ht, at = g.ht.values.astype(int), g['at'].values.astype(int)
    mu = np.average(yv, weights=wg)
    resid_var = np.average((yv - mu) ** 2, weights=wg)
    lam = prm['K_pace'] * 1.0          # in games: each game has unit weight relative to residual variance
    A = np.zeros((NT, NT)); b = np.zeros(NT)
    np.add.at(A, (ht, ht), wg); np.add.at(A, (at, at), wg); np.add.at(A, (ht, at), wg); np.add.at(A, (at, ht), wg)
    np.add.at(b, ht, wg * (yv - mu)); np.add.at(b, at, wg * (yv - mu))
    pace = np.linalg.solve(A + lam * np.eye(NT), b)
    R['pace'] = {'mu': float(mu), 'P': pace.tolist(), 'sd': float(np.sqrt(resid_var))}
    return R


EX = lambda v: 1 / (1 + np.exp(-v))


def probs(R, off, dfn, home, shift_o=0.0):
    """Component probabilities for offense `off` vs defense `dfn`; home = 1 / 0 (0 for either side at a neutral site).
    shift_o: extra logit shift on all make probabilities (player availability, score effects)."""
    i, j = TI[off], TI[dfn]
    c = {}
    for k in COMPS:
        r = R[k]
        v = r['mu'] + r['h'] * home + r['O'][i] - r['D'][j]
        if k in ('m1', 'm2', 'm3'): v += shift_o
        c[k] = EX(v)
    return c


def ppp(c, E):
    """Expected points per possession (closed form). E = engine_params.json."""
    sh = [(1 - c['s3']) * c['s1'], (1 - c['s3']) * (1 - c['s1']), c['s3']]
    m = [c['m1'], c['m2'], c['m3']]; a1 = E['and1']; ft = c['ft']
    fg_pts = sum(sh[z] * m[z] * ((2 if z < 2 else 3) + a1[z] * ft) for z in range(3))
    fg_cont = sum(sh[z] * (1 - m[z]) for z in range(3)) * c['oreb']
    e = (1 - c['tov']) * (c['trip'] * (2 + E['trip3']) * ft + (1 - c['trip']) * fg_pts)
    cont = (1 - c['tov']) * (c['trip'] * (1 - ft) * E['oreb_ft'] + (1 - c['trip']) * fg_cont)
    return e / (1 - cont)


def pace48(R, home, away):
    p = R['pace']
    return float(np.exp(p['mu'] + p['P'][TI[home]] + p['P'][TI[away]]))
