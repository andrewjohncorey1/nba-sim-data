"""Held-out player-prop test (same method as pipeline/engine.js playerProps).
Usage: python3 validation/props.py <season> <variant> [phi_pts,phi_reb,phi_ast,phi_fg3m,phi_pra] [half_life] [prior_min]
Every player-game uses only earlier games: per-minute rates (recency-weighted, shrunk toward the position average)
x projected minutes (fitted to the team's 240, players.allocate), scaled by the simulated team quantity (points, made field goals
for assists, threes made, rebounds) and priced with a negative binomial mixed over the simulated team distribution.
The players are those who played (props are void otherwise). Lines: book-style = the player's season-to-date average
rounded down + 0.5 (needs 5+ games), plus fixed lines for each stat. The
negative-binomial dispersion is chosen per stat on all of its lines together (for each baseline too). Baselines price the same lines with the same
distribution shape from the season-to-date average and from the last-10-games average."""
import sys, os, json
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
from scipy.stats import nbinom
import players as PS

Y, var = int(sys.argv[1]), sys.argv[2]
phis = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '10,10,10,10,10').split(',')]
PHI = dict(zip(['pts', 'reb', 'ast', 'fg3m', 'pra'], phis))
HL = float(sys.argv[4]) if len(sys.argv) > 4 else 180; PRIOR = float(sys.argv[5]) if len(sys.argv) > 5 else 300
V2 = os.environ.get('PROPS_V2')            # path to validation/props2_tuned.json: tuned minutes, rates and dispersion
CFG = json.load(open(V2)) if V2 else None
sims = {s['game_id']: s for s in json.load(open(f'validation/sim_{Y}_{var}.json'))}
PL = PS.load(range(Y - 2, Y + 1))
PL['posg'] = PL.pos.fillna('F').str[0].map({'G': 'G', 'F': 'F', 'C': 'C'}).fillna('F')
PL['pra'] = PL.pts + PL.reb + PL.ast
stats = ['pts', 'reb', 'ast', 'fg3m', 'fgm']
cur = PL[(PL.season == Y) & (PL['min'] > 0) & PL.game_id.isin(sims)]
g = pd.read_parquet(f'data/derived/games_{Y}.parquet').set_index('game_id')
REB_REF = None


def nb_over(m, phi, L):
    m = np.maximum(np.asarray(m, float), 1e-6)
    return 1 - nbinom.cdf(np.floor(L), phi, phi / (phi + m))


def mix_over(pmf, lo, scale, phi, L):
    """sum_T pmf[T] * P(NB(scale*T) > L)"""
    T = np.arange(lo, lo + len(pmf)); w = np.asarray(pmf)
    return float(np.sum(w * nb_over(scale * np.maximum(T, 0.5), phi, L)))


GRID = [3, 5, 8, 12, 20, 40]
FIXED = {'pts': (9.5, 14.5, 19.5, 24.5, 29.5), 'reb': (3.5, 5.5, 7.5, 9.5), 'ast': (1.5, 3.5, 5.5, 7.5),
         'fg3m': (0.5, 1.5, 2.5, 3.5), 'pra': (14.5, 19.5, 24.5, 29.5, 34.5)}
rows = []
for d, gd in cur.groupby('date'):
    hist = PL[PL.date < d]
    x = hist[hist['min'] > 0]
    w = PS.weights(x.date, d, HL); x = x.assign(w=w)[w > 1e-3]
    A = x.assign(wm=x.w * x['min'], **{f'w_{s}': x.w * x[s] for s in stats}).groupby('pid')[['wm'] + [f'w_{s}' for s in stats]].sum()
    pa = {k: {s: float((v.w * v[s]).sum() / (v.w * v['min']).sum()) for s in stats} for k, v in x.groupby('posg')}
    pos = x.sort_values('date').groupby('pid').posg.last()
    if CFG:
        mc = CFG['cfg']['min']
        pm = PS.projected_minutes(PL, d, hl_games=mc['hl'], prior_games=mc['pw'], prior_min=14.0)
        AH = {}
        def agg_h(h):
            if h not in AH:
                xh = hist[hist['min'] > 0]; wh = PS.weights(xh.date, d, h); xh = xh.assign(w=wh)[wh > 1e-3]
                AH[h] = xh.assign(wm=xh.w * xh['min'], **{f'w_{s}': xh.w * xh[s] for s in stats}).groupby('pid')[['wm'] + [f'w_{s}' for s in stats]].sum()
            return AH[h]
        A720 = agg_h(720)
        def rate_v2(p, s):
            rc = CFG['cfg']['rate'][s]; pav = pa.get(pos.get(p, 'F'), pa['F'])[s]; Ah = agg_h(rc['hl'])
            tgt = (A720.at[p, f'w_{s}'] + 300 * pav) / (A720.at[p, 'wm'] + 300) if (rc['target'] == 'own' and p in A720.index) else pav
            if p not in Ah.index: return tgt if p in A720.index else pav
            return (Ah.at[p, f'w_{s}'] + rc['K'] * tgt) / (Ah.at[p, 'wm'] + rc['K'])
    else:
        pm = PS.projected_minutes(PL, d)
    season_hist = hist[(hist.season == Y) & (hist['min'] > 0)]
    sb = season_hist.groupby('pid')[['pts', 'reb', 'ast', 'fg3m', 'pra']].agg(['mean', 'count'])
    l10 = season_hist.sort_values('date').groupby('pid').tail(10).groupby('pid')[['pts', 'reb', 'ast', 'fg3m', 'pra']].mean()
    for (gid, team), tp in gd.groupby(['game_id', 'team']):
        side = 'h' if g.loc[gid, 'home'] == team else 'a'
        S = sims[gid]
        mins = pm.proj.reindex(tp.pid).fillna(12.0).values
        mins = PS.allocate(mins)
        if CFG:
            rt = np.array([[rate_v2(p, s) for s in stats] for p in tp.pid])
        else:
            rt = np.array([[ (A.loc[p, f'w_{s}'] + PRIOR * pa.get(pos.get(p, 'F'), pa['F'])[s]) / (A.loc[p, 'wm'] + PRIOR) if p in A.index
                             else pa.get(pos.get(p, 'F'), pa['F'])[s] for s in stats] for p in tp.pid])
        raw = rt * mins[:, None]
        R = raw.sum(axis=0)
        pmfs = {'pts': (S['p' + side], 60), 'fgm': (S['fg' + side], 20), 'fg3m': (S['f3' + side], 0), 'reb': (S['rb' + side], 20)}
        ET = {k: float(np.dot(np.arange(lo, lo + len(v)), v)) for k, (v, lo) in pmfs.items()}
        for i, (p, act) in enumerate(zip(tp.pid, tp.itertuples())):
            if p not in sb.index or sb.loc[p, ('pts', 'count')] < 5: continue
            sh = {'pts': raw[i, 0] / R[0], 'reb': raw[i, 1] / R[1], 'ast': raw[i, 2] / R[4], 'fg3m': raw[i, 3] / R[3]}
            means = {'pts': sh['pts'] * ET['pts'] * (CFG['cfg'].get('tmult', 1.0) if CFG else 1.0), 'reb': sh['reb'] * ET['reb'], 'ast': sh['ast'] * ET['fgm'], 'fg3m': sh['fg3m'] * ET['fg3m']}
            means['pra'] = means['pts'] + means['reb'] + means['ast']
            for k in ('pts', 'reb', 'ast', 'fg3m', 'pra'):
                b = sb.loc[p, (k, 'mean')]; L = np.floor(b) + 0.5
                hk = {'pts': 'pts', 'reb': 'reb', 'ast': 'fgm', 'fg3m': 'fg3m', 'pra': 'pts'}[k]
                v, lo = pmfs[hk]
                scale = means[k] / ET[hk]
                lines = [('book', L)] + [('fixed', x) for x in FIXED[k]]
                for kind, LL in lines:
                    r = dict(date=d, game_id=gid, pid=p, stat=k, kind=kind, line=LL, actual=getattr(act, k), mean=means[k], base=b)
                    if CFG:
                        dc = CFG['dispersion'][k]
                        f_ = (max(dc['c'] * means[k], 0.3) if dc['mode'] == 'prop' else
                              max(dc['f'][0] * (max(means[k], 1e-3) / 10) ** dc['f'][1], 0.3) if dc['mode'] == 'pow' else dc['phi'])
                        r['mnew'] = mix_over(v, lo, scale, f_, LL)
                    for f in GRID:
                        if not CFG: r[f'm{f}'] = mix_over(v, lo, scale, f, LL)
                        r[f's{f}'] = float(nb_over(b, f, LL)); r[f'r{f}'] = float(nb_over(l10.loc[p, k], f, LL))
                    rows.append(r)
df = pd.DataFrame(rows)
df['y'] = (df.actual > df.line).astype(int)
if CFG:      # the tuned model has its own dispersion: every grid column is the same tuned price (keeps the scoring code below)
    for f in GRID: df[f'm{f}'] = df['mnew']


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


res = {}
chosen = {}
for k, x in df.groupby('stat'):
    chosen[k] = {}
    for c, name in (('m', 'model'), ('s', 'season_avg'), ('r', 'recent_10')):
        if c == 'm' and len(sys.argv) > 3: chosen[k][name] = PHI[k]; continue
        cand = {f: ll(x.y, x[f'{c}{f}']) for f in GRID}
        chosen[k][name] = min(cand, key=cand.get)
for (k, kind), x in df.groupby(['stat', 'kind']):
    ph = chosen[k]
    res[f'{k} {kind}'] = {'n': len(x), **{n: round(ll(x.y, x[f"{c}{ph[n]}"]), 4) for c, n in (('m', 'model'), ('s', 'season_avg'), ('r', 'recent_10'))},
                          'phi': ph, 'pred_over': round(float(x[f"m{ph['model']}"].mean()), 3), 'act_over': round(float(x.y.mean()), 3),
                          'mae_model': round(float(np.mean(np.abs(x.actual - x['mean']))), 3), 'mae_season': round(float(np.mean(np.abs(x.actual - x.base))), 3)}
print(json.dumps({'season': Y, 'phi_fixed': PHI if len(sys.argv) > 3 else None, 'half_life': HL, 'prior_min': PRIOR, 'results': res}, indent=0))
df.to_parquet(f'validation/props_{Y}_{var}{os.environ.get("PROPS_SUFFIX", "_v2") if CFG else ""}.parquet', index=False)
