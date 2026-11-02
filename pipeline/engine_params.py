"""
League-wide game-flow settings for the possession engine, measured from the possession tables.

Usage (repo root): python3 pipeline/engine_params.py <first_season> <last_season>   (e.g. 2022 2025)
Writes pipeline/engine_params.json

- dur:    possession length quantiles by how the previous possession ended (made basket or free throw, defensive
          rebound, turnover, period start), normal play.
- score:  score effects outside the late-game window: by the offense's margin, points per possession relative to
          what the two teams' season averages imply (applied as a shift of every make probability), possession length and three-point share.
- late:   last 150 seconds of the 4th quarter or overtime, by seconds left and the offense's margin:
          qf   = possession ends with a quick foul (free-throw trip within 5 s, no shot): intentional fouling
          kill = possession with no shot, no turnover, no free throws and no points (leader runs the clock out)
          for the other possessions: mean length, points per possession relative to normal, three-point share.
- heave:  possessions starting with 3 s or less on the period clock: chance of scoring.
- league constants: and-one rate after a make by zone, offensive rebound rate after a missed last free throw,
          share of free-throw trips that are 3 shots, technical free-throw points per game, period lengths.
"""
import sys, json
import numpy as np, pandas as pd

y0, y1 = int(sys.argv[1]), int(sys.argv[2])
def load_poss(y):
    p = pd.read_parquet(f'data/derived/poss_{y}.parquet')
    # expected points per possession from the two teams' season averages, so effects by margin are not confused with
    # team quality (bad teams trail more often)
    p['dfn'] = p.groupby(['game_id', 'per']).off.shift(1).fillna(p.groupby(['game_id', 'per']).off.shift(-1))
    lgm = p.pts.mean()
    p['exp'] = p.off.map(p.groupby('off').pts.mean()) + p.dfn.map(p.groupby('dfn').pts.mean()) - lgm
    return p


P = pd.concat([load_poss(y) for y in range(y0, y1 + 1)], ignore_index=True)
T = pd.concat([pd.read_parquet(f'data/derived/teamgames_{y}.parquet') for y in range(y0, y1 + 1)], ignore_index=True)
P['prev'] = P.groupby(['game_id', 'per']).end.shift().fillna('start').replace('', 'miss')
P['late'] = (P.per >= 4) & (P.sec_left <= 150)
P['qf'] = ((P.first_trip >= 0) & (P.first_trip <= 5) & (P.fga == 0)).astype(int)
P['kill'] = ((P.fga == 0) & (P.tov == 0) & (P.trips == 0) & (P.pts == 0)).astype(int)
normal = P[~P.late & (P.sec_left > 24)]
base_ppp = float(normal.pts.mean())
base_f3 = float(normal.fga3.sum() / normal.fga.sum())
base_dur = float(normal.dur.mean())
Q = np.linspace(0.0125, 0.9875, 40)
out = {'seasons': [y0, y1], 'base_ppp': base_ppp, 'base_f3': base_f3, 'base_dur': base_dur, 'dur': {}}
for k in ('make', 'miss', 'tov', 'start'):
    d = normal[normal.prev == k].dur.clip(0, 35)
    out['dur'][k] = [round(float(v), 2) for v in np.quantile(d, Q)]

# league component averages (for the score-effect shift)
fga = T.fga1 + T.fga2 + T.fga3
L = dict(tov=T.tov.sum() / T.opps.sum(), trip=T.trips.sum() / (T.trips.sum() + fga.sum()),
         s3=T.fga3.sum() / fga.sum(), s1=T.fga1.sum() / (T.fga1.sum() + T.fga2.sum()),
         m1=T.fgm1.sum() / T.fga1.sum(), m2=T.fgm2.sum() / T.fga2.sum(), m3=T.fgm3.sum() / T.fga3.sum(),
         oreb=T.oreb_fg.sum() / T.miss_fg.sum(), ft=T.ftm.sum() / T.fta.sum())
out['and1'] = [float(T[f'and1_{z}'].sum() / T[f'fgm{z}'].sum()) for z in (1, 2, 3)]
out['oreb_ft'] = float(T.oreb_ft.sum() / T.miss_ftlast.sum())
out['trip3'] = float(T.trips3.sum() / T.trips.sum())
out['tech_pts'] = float(T.tech_pts.sum() / T.game_id.nunique())
out['league'] = {k: float(v) for k, v in L.items()}


def ppp(c, shift=0.0):
    """Expected points per possession from component probabilities (same formula as engine.js expPPP)."""
    lg = lambda p: np.log(p / (1 - p)); ex = lambda x: 1 / (1 + np.exp(-x))
    m = [ex(lg(c['m1']) + shift), ex(lg(c['m2']) + shift), ex(lg(c['m3']) + shift)]
    sh = [(1 - c['s3']) * c['s1'], (1 - c['s3']) * (1 - c['s1']), c['s3']]
    a1 = out['and1']; ft = c['ft']; t3 = out['trip3']
    fg_pts = sum(sh[z] * m[z] * ((2 if z < 2 else 3) + a1[z] * ft) for z in range(3))
    fg_cont = sum(sh[z] * (1 - m[z]) for z in range(3)) * c['oreb']
    trip_pts = (2 + t3) * ft
    trip_cont = (1 - ft) * out['oreb_ft']
    e = (1 - c['tov']) * (c['trip'] * trip_pts + (1 - c['trip']) * fg_pts)
    cont = (1 - c['tov']) * (c['trip'] * trip_cont + (1 - c['trip']) * fg_cont)
    return e / (1 - cont)


p0 = ppp(L)
sens = (np.log(ppp(L, 0.01)) - np.log(ppp(L, -0.01))) / 0.02
out['ppp_formula'] = float(p0); out['ppp_sens'] = float(sens)

edges = [-99, -20, -12, -6, -1, 0, 5, 11, 19, 99]
x = normal.copy(); x['b'] = pd.cut(x.margin, edges, labels=False)
g = x.groupby('b').agg(pts=('pts', 'sum'), exp=('exp', 'sum'), dur=('dur', 'mean'), f3=('fga3', 'sum'), fga=('fga', 'sum'))
out['score'] = {'edges': edges, 'shift': [float(np.log(a / b) / sens) for a, b in zip(g.pts, g.exp)],
                'dur': [float(v / base_dur) for v in g.dur], 'f3': [float(a / b) for a, b in zip(g.f3, g.fga)]}

tedges, medges = [0, 24, 60, 150], [-99, -10, -7, -4, -3, -1, 0, 2, 3, 6, 9, 99]
Lt = P[P.late & (P.sec_left > 3)].copy()
Lt['tb'] = pd.cut(Lt.sec_left, tedges, labels=False); Lt['mb'] = pd.cut(Lt.margin, medges, labels=False)
late = {'tedges': tedges, 'medges': medges, 'qf': [], 'kill': [], 'dur': [], 'shift': [], 'f3': [], 'qf_dur': float(Lt[Lt.qf == 1].dur.mean())}
PRIOR = 50     # possessions of shrinkage toward the neighbouring overall rate for thin cells
for tb in range(3):
    row = {k: [] for k in ('qf', 'kill', 'dur', 'shift', 'f3')}
    for mb in range(len(medges) - 1):
        c = Lt[(Lt.tb == tb) & (Lt.mb == mb)]
        n = len(c)
        qf = (c.qf.sum() + PRIOR * 0.03) / (n + PRIOR)
        rest = c[c.qf == 0]
        kill = (rest.kill.sum() + PRIOR * 0.02) / (len(rest) + PRIOR)
        r2 = rest[rest.kill == 0]
        dur = (r2.dur.sum() + PRIOR * base_dur) / (len(r2) + PRIOR)
        pp = base_ppp * (r2.pts.sum() + PRIOR * base_ppp) / (r2.exp.sum() + PRIOR * base_ppp)
        f3 = (r2.fga3.sum() + PRIOR * base_f3) / (r2.fga.sum() + PRIOR)
        row['qf'].append(round(float(qf), 4)); row['kill'].append(round(float(kill), 4)); row['dur'].append(round(float(dur), 2))
        row['shift'].append(round(float(np.log(pp / base_ppp) / sens), 4)); row['f3'].append(round(float(f3), 4))
    for k in row: late[k].append(row[k])
out['late'] = late
H = P[P.sec_left <= 3]
out['heave'] = {'p_score': float((H.pts > 0).mean()), 'pts_if': float(H[H.pts > 0].pts.mean()), 'dur': float(H.dur.clip(0, 3).mean())}
json.dump(out, open('pipeline/engine_params.json', 'w'), indent=1)
print(f'ENGINE PARAMS {y0}-{y1}: {len(P)} possessions; normal ppp {base_ppp:.4f} (formula {p0:.4f}), '
      f'3PA share {base_f3:.3f}, mean length {base_dur:.1f}s; heave scores {out["heave"]["p_score"]:.3f}')
