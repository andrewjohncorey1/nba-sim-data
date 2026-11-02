"""
Player impact (for availability adjustments) and player rates (for props).

Impact model ("box-score-informed adjusted plus-minus", fit on whole games):
  for each team-game, offense points per 100 possessions minus the league mean and home term =
      sum over the offense's players of (minutes share x O_p)  -  sum over the defense's players of (minutes share x D_p)
  where a team's five minutes shares add to 5 (minutes / game minutes). Rows weighted by possessions and game age.
  Each player's O_p and D_p are ridge-penalized toward a box-score prior: a linear function of his recency-weighted
  per-100-possession box stats (coefficients fit once on 2021-22..2023-24, validation/player_prior.json).

Availability adjustment for a team in a game:
  delta_O = sum_p (s_hat_p - s_bar_p) O_p,  delta_D likewise,
  s_bar = the team's minutes shares in the games behind its team rating (same recency weights as the ratings),
  s_hat = tonight's projected shares for the players available (recent minutes, renormalized to 5).
"""
import json, os
import numpy as np, pandas as pd
import scipy.sparse as sp

FEATS = ['pts', 'fga', 'fta', 'fg3a', 'fg3m', 'ast', 'tov', 'oreb', 'dreb', 'stl', 'blk', 'pf']


def load(seasons):
    out = []
    for y in seasons:
        if not os.path.exists(f'data/derived/players_{y}.parquet'): continue
        pl = pd.read_parquet(f'data/derived/players_{y}.parquet')
        tg = pd.read_parquet(f'data/derived/teamgames_{y}.parquet')[['game_id', 'team', 'poss', 'pts', 'tech_pts']]
        g = pd.read_parquet(f'data/derived/games_{y}.parquet')[['game_id', 'ot', 'neutral', 'season']]
        pl = pl.merge(tg, on=['game_id', 'team'], suffixes=('', '_team')).merge(g, on='game_id')
        out.append(pl)
    pl = pd.concat(out, ignore_index=True)
    pl['gmin'] = 48 + 5 * pl.ot.fillna(0)
    pl['share'] = (pl['min'] / pl.gmin).clip(0, 1)
    pl['pid'] = pl.pid.astype('int64')
    return pl


def weights(dates, asof, half_life):
    age = (pd.Timestamp(asof) - pd.to_datetime(dates)).dt.days.values.astype(float)
    return np.where(age > 0, 0.5 ** (age / half_life), 0.0)


def box_features(PL, asof, half_life, prior_min=250.0):
    """Recency-weighted per-100-possession box rates per player (shrunk toward low-minute averages) + minutes/game."""
    x = PL[(PL.date < pd.Timestamp(asof)) & (PL['min'] > 0)]
    w = weights(x.date, asof, half_life)
    x = x.assign(w=w)[w > 1e-3]
    onposs = x.share * x.poss                      # possessions on court (approx.)
    agg = {'w_min': (x.w * x['min']), 'w_poss': x.w * onposs, 'w_g': x.w}
    for f in FEATS: agg[f] = x.w * x[f]
    A = pd.DataFrame(agg).assign(pid=x.pid.values).groupby('pid').sum()
    # shrink toward the rates of fringe players (prior_min minutes worth)
    lowA = A[A.w_min < 300].sum()
    poss_per_min = A.w_poss.sum() / A.w_min.sum()
    F = pd.DataFrame(index=A.index)
    for f in FEATS:
        prior_rate = lowA[f] / lowA.w_poss if lowA.w_poss > 0 else 0
        F[f] = 100 * (A[f] + prior_rate * prior_min * poss_per_min) / (A.w_poss + prior_min * poss_per_min)
    F['mpg'] = A.w_min / A.w_g
    F['wmin'] = A.w_min
    return F


def design(F):
    X = F[FEATS].copy()
    X['mpg'] = F.mpg
    X['one'] = 1.0
    return X


def fit(PL, asof, prm, prior=None):
    """Returns DataFrame indexed by pid with O, D (points per 100 possessions while on court, relative to average)."""
    x = PL[(PL.date < pd.Timestamp(asof))]
    w_all = weights(x.date, asof, prm['half_life_p'])
    x = x.assign(w=w_all)[w_all > 1e-3]
    x = x[x['min'] > 0]
    F = box_features(PL, asof, prm['half_life_p'])
    pids = F.index.values; pix = {p: i for i, p in enumerate(pids)}; n = len(pids)
    # rows: one per team-game offense
    tg = x.groupby(['game_id', 'team']).agg(opp=('opp', 'first'), home=('home', 'first'), neutral=('neutral', 'first'),
                                              poss=('poss', 'first'), pts=('pts_team', 'first'), tech=('tech_pts', 'first'),
                                              w=('w', 'first')).reset_index()
    tg['ortg'] = 100 * (tg.pts - tg.tech) / tg.poss
    tg['hm'] = np.where(tg.neutral == 1, 0, tg.home)
    mu = np.average(tg.ortg, weights=tg.w * tg.poss)
    hh = np.average(tg.ortg[tg.hm == 1], weights=(tg.w * tg.poss)[tg.hm == 1]) - np.average(tg.ortg[tg.hm == 0], weights=(tg.w * tg.poss)[tg.hm == 0])
    y = (tg.ortg - mu - hh * (tg.hm - tg.hm.mean())).values
    rw = (tg.w * tg.poss / 100).values
    ri = {(g, t): i for i, (g, t) in enumerate(zip(tg.game_id, tg.team))}
    # offense players
    io = x[['game_id', 'team']].apply(tuple, axis=1).map(ri).values
    idf = x[['game_id', 'opp']].apply(tuple, axis=1).map(ri).values      # row where this player is the defense
    col = x.pid.map(pix).values
    sh = x.share.values * 5 / x.groupby(['game_id', 'team']).share.transform('sum').values   # shares add to 5
    ok_o = ~pd.isna(io); ok_d = ~pd.isna(idf)
    rows = np.concatenate([io[ok_o], idf[ok_d]]).astype(int)
    cols = np.concatenate([col[ok_o], n + col[ok_d]]).astype(int)
    vals = np.concatenate([sh[ok_o], -sh[ok_d]])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(tg), 2 * n))
    W = sp.diags(rw)
    if prior is not None:
        Z = design(F)
        bO = Z.values @ np.array([prior['O'][c] for c in Z.columns])
        bD = Z.values @ np.array([prior['D'][c] for c in Z.columns])
    else:
        bO = np.zeros(n); bD = np.zeros(n)
    b = np.concatenate([bO, bD])
    lam = prm['lam_p']
    A = (X.T @ W @ X).toarray() + lam * np.eye(2 * n)
    rhs = X.T @ (rw * y) + lam * b
    beta = np.linalg.solve(A, rhs)
    out = F[['mpg', 'wmin']].copy()
    out['O'], out['D'], out['bO'], out['bD'] = beta[:n], beta[n:], bO, bD
    out.attrs['mu'] = mu; out.attrs['home'] = hh
    return out


def fit_prior(PL, asof, prm):
    """Box-score prior coefficients: regress plain ridge APM estimates on box features (minutes-weighted)."""
    R = fit(PL, asof, prm, prior=None)
    F = box_features(PL, asof, prm['half_life_p'])
    Z = design(F).loc[R.index]
    wt = R.wmin.values
    out = {}
    for k in ('O', 'D'):
        Wz = Z.values * wt[:, None]
        coef = np.linalg.solve(Z.values.T @ Wz + 1e-3 * np.eye(Z.shape[1]), Wz.T @ R[k].values)
        out[k] = dict(zip(Z.columns, coef.tolist()))
    return out


def team_shares(PL, team, asof, half_life):
    """s_bar: the team's recency-weighted minutes shares (adding to 5) over its games before asof."""
    x = PL[(PL.team == team) & (PL.date < pd.Timestamp(asof))]
    w = weights(x.date, asof, half_life)
    x = x.assign(w=w)[w > 1e-4]
    if not len(x): return pd.Series(dtype=float)
    s = (x.w * x.share).groupby(x.pid).sum() / x.groupby('game_id').w.first().sum()
    return s * 5 / s.sum()


def projected_minutes(PL, asof, hl_games=8, prior_games=3, prior_min=14.0):
    """Each player's projected minutes when he plays: weighted average of his recent games (half-life in games),
    shrunk toward prior_min with prior_games of weight. Uses only games before asof."""
    x = PL[(PL.date < pd.Timestamp(asof)) & (PL['min'] > 0)].sort_values('date')
    x = x.groupby('pid').tail(30)
    k = x.groupby('pid').cumcount(ascending=False).values
    w = 0.5 ** (k / hl_games)
    x = x.assign(w=w, wm=w * x['min'])
    g = x.groupby('pid').agg(wm=('wm', 'sum'), w=('w', 'sum'), last=('date', 'max'), team=('team', 'last'))
    g['proj'] = (g.wm + prior_min * prior_games * 0.5) / (g.w + prior_games * 0.5)
    return g


def allocate(m, total=240.0, cap=48.0):
    """Fit expected minutes to the team total: the gap is spread in proportion to m * (1 - m/48)^3, so rotation
    regulars move little and the bench absorbs most of it (engine.js allocate is the same)."""
    m = np.asarray(m, float).copy()
    for _ in range(6):
        gap = total - m.sum()
        if abs(gap) < 0.01: break
        w = m * np.clip(1 - m / cap, 0.05, None) ** 3
        m = np.clip(m + gap * w / w.sum(), 0, cap)
    return m


def availability_delta(R, sbar, avail_min, repl):
    """delta O and D for a team. avail_min: Series pid -> projected minutes for players available tonight."""
    if not len(avail_min): return 0.0, 0.0
    s_hat = pd.Series(allocate(avail_min.values) / 48, index=avail_min.index)
    ids = sbar.index.union(s_hat.index)
    sb = sbar.reindex(ids).fillna(0); shh = s_hat.reindex(ids).fillna(0)
    O = R.O.reindex(ids).fillna(repl[0]); D = R.D.reindex(ids).fillna(repl[1])
    d = shh - sb
    return float((d * O).sum()), float((d * D).sum())
