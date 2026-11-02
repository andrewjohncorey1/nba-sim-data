"""
Score logged pre-game predictions against results -> build/scorecard.json

- The prediction used for each game is the last snapshot taken before its scheduled start.
- Results come from data/derived/games_<season>.parquet (matched on ESPN game id); games include overtime.
- Model vs Pinnacle vs baseline: moneyline, spread at Pinnacle's line and total at Pinnacle's line (pushes dropped).
  Baselines: the home win rate of the previous three seasons; for spreads and totals, the share of games in those
  seasons that went over / covered that line. The anchored moneyline is an independent check against Pinnacle's.
- Player props: the model's chance of going over its own line vs a baseline that prices the same line from the
  player's season-to-date average with the same negative-binomial spread.
"""
import glob, json, sys, os
import numpy as np, pandas as pd
from scipy.stats import nbinom

Y = int(sys.argv[1])
S = json.load(open('pipeline/settings.json'))
files = sorted(f for f in glob.glob('data/predictions/*.csv') if not os.path.basename(f).startswith('players_'))
out = {'generated': pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), 'n_scored': 0, 'n_pending': 0}


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


games = pd.concat([pd.read_parquet(f) for f in glob.glob('data/derived/games_*.parquet')], ignore_index=True)
games['game_id'] = games.game_id.astype(str)
hist = games[(games.season < Y) & (games.season >= Y - 3) & games.done]
if files:
    pr = pd.concat([pd.read_csv(f, dtype={'game_id': str}) for f in files], ignore_index=True)
    pr['snap'] = pd.to_datetime(pr.snap_utc, utc=True, format='ISO8601'); pr['st'] = pd.to_datetime(pr.start, utc=True, format='ISO8601')
    pr = pr[pr.snap < pr.st].sort_values('snap').drop_duplicates('game_id', keep='last')
    d = pr.merge(games[['game_id', 'home_pts', 'away_pts', 'ot', 'h1', 'h2', 'a1', 'a2', 'done']], on='game_id', how='left')
    done = d[d.done.fillna(False).astype(bool) & d.home_pts.notna()].copy()
    out['n_scored'] = int(len(done)); out['n_pending'] = int(len(d) - len(done))
    if len(done):
        done['home_win'] = (done.home_pts > done.away_pts).astype(int)
        done['total'] = done.home_pts + done.away_pts; done['marg'] = done.home_pts - done.away_pts
        hw = float((hist.home_pts > hist.away_pts).mean())
        out['ml'] = {'n': len(done), 'model': ll(done.home_win, done.p_home), 'base': ll(done.home_win, np.full(len(done), hw))}
        pm = done[done.pin_home.notna()]
        if len(pm):
            out['ml_pin'] = {'n': len(pm), 'model': ll(pm.home_win, pm.p_home), 'pin': ll(pm.home_win, pm.pin_home),
                             'anch': ll(pm.home_win, pm.anch_home) if pm.anch_home.notna().all() else None,
                             'base': ll(pm.home_win, np.full(len(pm), hw)),
                             'gap': float(np.mean(np.abs(pm.anch_home - pm.pin_home))) if pm.anch_home.notna().any() else None}
        mh = hist.home_pts - hist.away_pts; th = hist.home_pts + hist.away_pts
        p = done[done.pin_sp_line.notna() & ((done.marg + done.pin_sp_line) != 0)]
        if len(p):
            y = ((p.marg + p.pin_sp_line) > 0).astype(int)
            base = p.pin_sp_line.map(lambda L: float(((mh + L) > 0).sum() / max(1, ((mh + L) != 0).sum())))
            out['spread'] = {'n': len(p), 'model': ll(y, p.model_sp_home), 'pin': ll(y, p.pin_sp_home), 'base': ll(y, base)}
        t = done[done.pin_tot_line.notna() & (done.total != done.pin_tot_line)]
        if len(t):
            y = (t.total > t.pin_tot_line).astype(int)
            base = t.pin_tot_line.map(lambda L: float((th > L).sum() / max(1, (th != L).sum())))
            out['total'] = {'n': len(t), 'model': ll(y, t.model_over), 'pin': ll(y, t.pin_over), 'base': ll(y, base)}
        bins = [0, .25, .4, .5, .6, .75, 1]
        done['bin'] = pd.cut(done.p_home, bins)
        out['calib'] = [{'bin': f'{int(b.left * 100)}-{int(b.right * 100)}%', 'n': int(len(g)), 'pred': float(g.p_home.mean()), 'act': float(g.home_win.mean())}
                        for b, g in done.groupby('bin', observed=True)]
        out['shape'] = [
            {'k': 'Points per game (both teams)', 'pred': f'{done.exp_total.mean():.1f}', 'act': f'{done.total.mean():.1f}'},
            {'k': 'Home margin', 'pred': f'{done.exp_margin.mean():+.1f}', 'act': f'{done.marg.mean():+.1f}'},
            {'k': '1st-half points', 'pred': f'{done.h1_total.mean():.1f}', 'act': f'{(done.h1 + done.h2 + done.a1 + done.a2).mean():.1f}'},
            {'k': 'Overtime', 'pred': f'{100 * done.p_ot.mean():.1f}%', 'act': f'{100 * (done.ot > 0).mean():.1f}%'},
            {'k': 'Over model total line', 'pred': f'{100 * done.p_over.mean():.1f}%', 'act': f'{100 * (done.total > done.tot_line).mean():.1f}%'}]
        r = done.sort_values('st', ascending=False).head(40)
        out['recent'] = [{'date': str(x.st.tz_convert('America/New_York').date()), 'home': x.home, 'away': x.away, 'p_home': float(x.p_home),
                          'pin_home': None if pd.isna(x.pin_home) else float(x.pin_home),
                          'score': f'{x.away} {int(x.away_pts)}-{int(x.home_pts)} {x.home}{" OT" if x.ot > 0 else ""}',
                          'home_win': int(x.home_win), 'exp_total': float(x.exp_total), 'total': int(x.total),
                          'pin_total': None if pd.isna(x.pin_tot_line) else float(x.pin_tot_line)} for x in r.itertuples()]

# ---------- player props ----------
pf = sorted(glob.glob('data/predictions/players_*.csv'))
if pf:
    pp = pd.concat([pd.read_csv(f, dtype={'game_id': str}) for f in pf], ignore_index=True)
    pp['snap'] = pd.to_datetime(pp.snap_utc, utc=True, format='ISO8601'); pp['st'] = pd.to_datetime(pp.start, utc=True, format='ISO8601')
    pp = pp[(pp.snap < pp.st) & pp.pid.notna()].sort_values('snap').drop_duplicates(['game_id', 'pid', 'stat'], keep='last')
    pp['pid'] = pp.pid.astype('int64')
    act = pd.read_parquet(f'data/derived/players_{Y}.parquet')
    act = act[act['min'] > 0].copy(); act['game_id'] = act.game_id.astype(str); act['pra'] = act.pts + act.reb + act.ast
    act = act.sort_values(['pid', 'date'])
    statcols = ['pts', 'reb', 'ast', 'fg3m', 'pra']
    for c in statcols:
        act[f'b_{c}'] = act.groupby('pid')[c].transform(lambda s: s.shift().expanding().mean())
    phi = S['props'].get('phi', {})
    res = []
    for k in statcols:
        x = pp[pp.stat == k].merge(act[['game_id', 'pid', k, f'b_{k}']], on=['game_id', 'pid'], how='inner')
        x = x[x[f'b_{k}'].notna()]
        if not len(x): continue
        y = (x[k] > x.line).astype(int)
        f = phi.get(k, 10); m = x[f'b_{k}'].clip(lower=0.05)
        base = 1 - nbinom.cdf(np.floor(x.line), f, f / (f + m))
        res.append({'k': S['props'].get('labels', {}).get(k, k), 'n': int(len(x)), 'model': ll(y, x.p_over), 'base': ll(y, base),
                    'pred': float(x.p_over.mean()), 'act': float(y.mean())})
    out['players'] = {'n': int(sum(r['n'] for r in res)), 'n_logged': int(len(pp)), 'markets': res}
json.dump(out, open('build/scorecard.json', 'w'))
msg = f'SCORE {out["n_scored"]} scored, {out["n_pending"]} pending'
if out.get('ml'): msg += f'; moneyline log loss model {out["ml"]["model"]:.4f} vs baseline {out["ml"]["base"]:.4f}'
if out.get('ml_pin'): msg += f' (Pinnacle-priced: model {out["ml_pin"]["model"]:.4f}, Pinnacle {out["ml_pin"]["pin"]:.4f})'
if out.get('players'): msg += f'; {out["players"]["n"]} player lines scored'
print(msg)
