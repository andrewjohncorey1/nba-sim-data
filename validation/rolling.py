"""
Rolling held-out predictions: every game is predicted from ratings fit only on games before its date (refit daily).

Usage as a module: predict_season(Y, prm, TG=None, avail=None) -> DataFrame with per-game expected points, pace and
the inputs the engine needs. Used by tune.py (closed-form expected margin and total) and by the simulation tests.
"""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
import numpy as np, pandas as pd
import ratings as RT

E = json.load(open('pipeline/engine_params.json'))


def games(Y):
    g = pd.read_parquet(f'data/derived/games_{Y}.parquet')
    g = g[g.done & g.has_pbp & (g.n_per >= 4)].copy()
    g['reg_h'] = g[['h1', 'h2', 'h3', 'h4']].sum(axis=1); g['reg_a'] = g[['a1', 'a2', 'a3', 'a4']].sum(axis=1)
    return g.sort_values('start')


def predict_season(Y, prm, TG=None, adj=None, dates=None):
    """adj(game_row, R) -> (home_shift, away_shift) logit shifts on make probabilities (player availability etc.)."""
    TG = TG if TG is not None else RT.load_tg(range(Y - 3, Y + 1))
    g = games(Y)
    out = []
    for d, gd in g.groupby('date'):
        if dates is not None and d not in dates: continue
        R = RT.fit(TG, d, prm)
        for r in gd.itertuples():
            hm = 0 if r.neutral else 1
            sh, sa = adj(r, R) if adj else (0.0, 0.0)
            ch, ca = RT.probs(R, r.home, r.away, hm, sh), RT.probs(R, r.away, r.home, 0, sa)
            pace = RT.pace48(R, r.home, r.away)
            ph, pa = RT.ppp(ch, E), RT.ppp(ca, E)
            out.append(dict(game_id=r.game_id, date=d, home=r.home, away=r.away, stype=r.stype, neutral=r.neutral,
                            pace=pace, ppp_h=ph, ppp_a=pa, eh=ph * pace, ea=pa * pace, home_pts=r.home_pts, away_pts=r.away_pts,
                            reg_h=r.reg_h, reg_a=r.reg_a, ot=r.ot, ch=ch, ca=ca))
    return pd.DataFrame(out)


def score_closed(df, sd=12.8):
    """Closed-form scores: margin and regulation-total RMSE, and a normal-approximation win log loss."""
    from scipy.stats import norm
    m = df.eh - df.ea; act = df.home_pts - df.away_pts
    p = norm.cdf(m / sd).clip(1e-4, 1 - 1e-4); y = (act > 0).astype(int)
    return dict(n=len(df), margin_rmse=float(np.sqrt(np.mean((act - m) ** 2))),
                total_rmse=float(np.sqrt(np.mean((df.reg_h + df.reg_a - df.eh - df.ea) ** 2))),
                total_bias=float(np.mean(df.reg_h + df.reg_a - df.eh - df.ea)),
                ml_ll=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
