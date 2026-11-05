"""
Held-out joint-calibration test for same-game parlays (SGPs).
Usage (repo root): python3 validation/sgp.py <season> [v2]   e.g. 2026 (= 2025-26, the test season); v2 = the retuned
props (validation/props_<Y>_base_v2.parquet, settings in validation/props2_tuned.json)
Needs validation/rows_<season>_base.bin (+ .json) from
    PROPS=1 ROWS=validation/rows_<Y>_base.bin node validation/sim_games.js validation/inputs_<Y>_base.json validation/sim_<Y>_rows.json 4000
and validation/props_<Y>_base.parquet from validation/props.py. Same rules as validation/props_final.py: every game and
player-game rated on earlier games only; availability = the players who actually played; minutes from earlier games.

Joint model ("per-sim"): in each simulated game s, the player's points are negative binomial with mean
share x (team points in s), dispersion from pipeline/settings.json (chosen on 2024-25). Instead of drawing one random
points total per simulation, each simulation contributes the exact chance he goes over given that simulation's team
points (same model, less noise). So P(player over AND team wins) = mean over s of [P(over | s) x win_s], and the
single-leg price equals the prop model's own. Baseline ("legs multiplied"): the same two single-leg probabilities,
multiplied. The only thing that differs is the correlation, so this tests the correlation and nothing else.

Combos (one row per player-game with a book-style points line, or per game for the controls):
  A  player points over (book line) + own team wins (OT included)
  B  player points over (book line) + game over (model's fair total: mean rounded down + 0.5)
  C  control: home covers the model's fair spread + game over
  C2 control: home wins + game over
  D  info:    the team's top two projected scorers both over (teammates; the engine ties them only through team points)
  E  info:    each team's top projected scorer both over (opponents)
  F  info, 3 legs: player points over + own team wins + game over
  G  info, 3 legs: the team's top two projected scorers both over + their team wins
  For 3-leg combos the baseline multiplies all three legs; "leg B" in the conditional check is the other two together.
Pass rule (from the plan): keep the per-sim joint for a combo only if its log loss beats legs-multiplied on 2025-26.
A paired game-clustered bootstrap gives a 95% interval for the improvement, so noise can be told from signal.
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import nbinom

Y = int(sys.argv[1]); V2 = len(sys.argv) > 2 and sys.argv[2] == 'v2'
S = json.load(open('pipeline/settings.json'))
PHI = S['props']['phi']['pts']
DISP = json.load(open('validation/props2_tuned.json'))['dispersion']['pts'] if V2 else {'mode': 'fixed', 'phi': PHI}
phi_of = (lambda m: max(DISP['c'] * m, 0.3)) if DISP['mode'] == 'prop' else (lambda m: DISP['phi'])
PCOL = 'mnew' if V2 else f'm{PHI}'
SUF = '_v2' if V2 else ''
meta = json.load(open(f'validation/rows_{Y}_base.bin.json'))
N = meta['N']
R = np.fromfile(f'validation/rows_{Y}_base.bin', dtype=np.int16).reshape(len(meta['game_ids']), N, 2).astype(float)
gidx = {g: i for i, g in enumerate(meta['game_ids'])}
sims = {s['game_id']: s for s in json.load(open(f'validation/sim_{Y}_rows.json'))}
games = pd.read_parquet(f'validation/games_{Y}_base.parquet').set_index('game_id')
pl = pd.read_parquet(f'data/derived/players_{Y}.parquet')[['game_id', 'pid', 'team']]
pl['pid'] = pl.pid.astype('int64')
pr = pd.read_parquet(f'validation/props_{Y}_base{SUF}.parquet', columns=['game_id', 'pid', 'stat', 'kind', 'line', 'actual', 'mean', PCOL])
pr = pr[(pr.stat == 'pts') & (pr.kind == 'book')].merge(pl, on=['game_id', 'pid'], how='left')
pr = pr[pr.game_id.isin(gidx) & pr.team.notna()].copy()


def ET(s, side):
    """Expected team points exactly as validation/props.py computed it (from the clipped 60-170 pmf)."""
    v = np.asarray(s['p' + side]); return float(np.dot(np.arange(60, 60 + len(v)), v))


rows, ctrl = [], []
for gid, gp in pr.groupby('game_id'):
    i = gidx[gid]; g = games.loc[gid]
    h, a = R[i, :, 0], R[i, :, 1]
    marg, tot = h - a, h + a
    Lt = np.floor(tot.mean()) + 0.5
    Ls = np.floor(-marg.mean()) + 0.5                      # home's fair spread line (cover when margin + Ls > 0)
    over_s = (tot > Lt).astype(float)
    am, at = g.home_pts - g.away_pts, g.home_pts + g.away_pts
    ctrl.append(dict(game_id=gid, kind='C', pA=float(((marg + Ls) > 0).mean()), pB=float(over_s.mean()),
                     joint=float((((marg + Ls) > 0) * over_s).mean()), yA=int(am + Ls > 0), yB=int(at > Lt)))
    ctrl.append(dict(game_id=gid, kind='C2', pA=float((marg > 0).mean()), pB=float(over_s.mean()),
                     joint=float(((marg > 0) * over_s).mean()), yA=int(am > 0), yB=int(at > Lt)))
    per_player = []
    for r in gp.itertuples():
        side = 'h' if r.team == g.home else 'a'
        T = h if side == 'h' else a
        share = r.mean / ET(sims[gid], side)
        m = np.maximum(share * T, 1e-6)
        f = phi_of(r.mean)
        p_s = nbinom.sf(np.floor(r.line), f, f / (f + m))
        win_s = (marg > 0).astype(float) if side == 'h' else (marg < 0).astype(float)
        won = int(am > 0) if side == 'h' else int(am < 0)
        yA = int(r.actual > r.line)
        rows.append(dict(game_id=gid, pid=r.pid, kind='A', mean=r.mean, pA=float(p_s.mean()), pB=float(win_s.mean()),
                         joint=float((p_s * win_s).mean()), yA=yA, yB=won, prop_model=getattr(r, PCOL)))
        rows.append(dict(game_id=gid, pid=r.pid, kind='B', mean=r.mean, pA=float(p_s.mean()), pB=float(over_s.mean()),
                         joint=float((p_s * over_s).mean()), yA=yA, yB=int(at > Lt), prop_model=getattr(r, PCOL)))
        rows.append(dict(game_id=gid, pid=r.pid, kind='F', mean=r.mean, pA=float(p_s.mean()), pB=float((win_s * over_s).mean()),
                         legs3=float(p_s.mean() * win_s.mean() * over_s.mean()), joint=float((p_s * win_s * over_s).mean()),
                         yA=yA, yB=int(won and at > Lt)))
        per_player.append((r.team, r.mean, p_s, yA, win_s, won))
    by_team = {}
    for t, mu, p_s, y, ws, wn in per_player: by_team.setdefault(t, []).append((mu, p_s, y, ws, wn))
    tops = {}
    for t, L in by_team.items():
        L.sort(key=lambda x: -x[0])
        tops[t] = L[0]
        if len(L) >= 2:
            (m1, p1, y1, ws, wn), (m2, p2, y2, _, _) = L[0], L[1]
            rows.append(dict(game_id=gid, kind='D', mean=min(m1, m2), pA=float(p1.mean()), pB=float(p2.mean()),
                             joint=float((p1 * p2).mean()), yA=y1, yB=y2))
            rows.append(dict(game_id=gid, kind='G', mean=min(m1, m2), pA=float(p1.mean()), pB=float((p2 * ws).mean()),
                             legs3=float(p1.mean() * p2.mean() * ws.mean()), joint=float((p1 * p2 * ws).mean()), yA=y1, yB=int(y2 and wn)))
    if len(tops) == 2:
        (m1, p1, y1, _, _), (m2, p2, y2, _, _) = tops.values()
        rows.append(dict(game_id=gid, kind='E', mean=min(m1, m2), pA=float(p1.mean()), pB=float(p2.mean()),
                         joint=float((p1 * p2).mean()), yA=y1, yB=y2))
df = pd.DataFrame(rows + ctrl)
df['legs'] = df.legs3.fillna(df.pA * df.pB) if 'legs3' in df else df.pA * df.pB
df['y'] = df.yA * df.yB


def ll_rows(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


rng = np.random.default_rng(7)
LABEL = {'A': 'Player points over + own team wins', 'B': 'Player points over + game over',
         'C': 'Control: home covers fair spread + game over', 'C2': 'Control: home wins + game over',
         'D': 'Info: top two teammates both over', 'E': 'Info: both teams\' top scorers over',
         'F': '3 legs: player points over + own team wins + game over', 'G': '3 legs: top two teammates both over + team wins'}


def evaluate(x):
    lj, lg = ll_rows(x.y.values, x.joint.values), ll_rows(x.y.values, x.legs.values)
    d = pd.Series(lg - lj).groupby(x.game_id.values).sum()          # per-game improvement (positive = joint better)
    G = len(d); vals = d.values
    boot = np.array([vals[rng.integers(0, G, G)].sum() for _ in range(2000)]) / len(x)
    # conditional check: chance of leg A given leg B happened / did not, predicted vs actual
    pAgB = x.joint / x.pB.clip(1e-6); pAgnB = (x.pA - x.joint) / (1 - x.pB).clip(1e-6)
    hb, mb = x.yB == 1, x.yB == 0
    bins = pd.cut(x.joint, [0, .1, .2, .3, .4, .5, .7, 1])
    return {'n': int(len(x)), 'games': int(G),
            'logloss_joint': round(float(lj.mean()), 5), 'logloss_legs': round(float(lg.mean()), 5),
            'improvement': round(float(lg.mean() - lj.mean()), 5),
            'improvement_95ci': [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)],
            'hit_rate': round(float(x.y.mean()), 4), 'pred_joint': round(float(x.joint.mean()), 4), 'pred_legs': round(float(x.legs.mean()), 4),
            'lift_pred': round(float(x.joint.sum() / x.legs.sum()), 3), 'lift_actual': round(float(x.y.sum() / x.legs.sum()), 3),
            'A_given_B': {'n': int(hb.sum()), 'pred': round(float(pAgB[hb].mean()), 4), 'act': round(float(x.yA[hb].mean()), 4)},
            'A_given_notB': {'n': int(mb.sum()), 'pred': round(float(pAgnB[mb].mean()), 4), 'act': round(float(x.yA[mb].mean()), 4)},
            'legA': {'pred': round(float(x.pA.mean()), 4), 'act': round(float(x.yA.mean()), 4)},
            'legB': {'pred': round(float(x.pB.mean()), 4), 'act': round(float(x.yB.mean()), 4)},
            'bins': [[str(b), int(len(z)), round(float(z.joint.mean()), 3), round(float(z.legs.mean()), 3), round(float(z.y.mean()), 3)]
                     for b, z in x.groupby(bins, observed=True)]}


out = {'season': Y, 'props': 'v2 (retuned)' if V2 else 'v1', 'dispersion_pts': DISP, 'sims_per_game': N, 'combos': {}}
chk = df[df.kind == 'A']
out['check_single_leg_vs_prop_model'] = round(float(np.abs(chk.pA - chk.prop_model).mean()), 5)
for k in ('A', 'B', 'C', 'C2', 'D', 'E', 'F', 'G'):
    x = df[df.kind == k]
    res = {'label': LABEL[k], 'all': evaluate(x)}
    if k in ('A', 'B', 'D', 'E', 'F', 'G'):
        res['main_scorers_15plus'] = evaluate(x[x['mean'] >= 15])
    if k == 'A':
        res['team_favoured'] = evaluate(x[x.pB >= .5]); res['team_underdog'] = evaluate(x[x.pB < .5])
    res['pass'] = bool(res['all']['improvement'] > 0) if k in ('A', 'B') else None
    out['combos'][k] = res
json.dump(out, open(f'validation/sgp_{Y}{SUF}.json', 'w'), indent=1)
df.to_parquet(f'validation/sgp_rows_{Y}{SUF}.parquet', index=False)

# readable log
L = [f'SGP joint-calibration test, season ending {Y} ({N} sims per game, props {"v2 retuned" if V2 else "v1"}, points dispersion {DISP})',
     f'single-leg check: per-sim points-over price vs the prop model, mean abs diff {out["check_single_leg_vs_prop_model"]}', '']
for k, res in out['combos'].items():
    for sub in [s for s in ('all', 'main_scorers_15plus', 'team_favoured', 'team_underdog') if s in res]:
        e = res[sub]
        L.append(f'{k} {res["label"]} [{sub}] n={e["n"]} games={e["games"]}')
        L.append(f'   log loss joint {e["logloss_joint"]:.5f} vs legs {e["logloss_legs"]:.5f}  improvement {e["improvement"]:+.5f} '
                 f'(95% {e["improvement_95ci"][0]:+.5f} to {e["improvement_95ci"][1]:+.5f})')
        L.append(f'   hit {e["hit_rate"]:.3f}  pred joint {e["pred_joint"]:.3f}  legs {e["pred_legs"]:.3f}  lift pred {e["lift_pred"]:.3f} actual {e["lift_actual"]:.3f}')
        L.append(f'   P(A | B hit)  pred {e["A_given_B"]["pred"]:.3f} act {e["A_given_B"]["act"]:.3f} (n {e["A_given_B"]["n"]});  '
                 f'P(A | B missed) pred {e["A_given_notB"]["pred"]:.3f} act {e["A_given_notB"]["act"]:.3f} (n {e["A_given_notB"]["n"]})')
        if sub == 'all':
            L.append('   bins [joint range, n, pred joint, legs, actual]: ' + '; '.join(f'{b[0]} {b[1]} {b[2]}/{b[3]}/{b[4]}' for b in e['bins']))
    if res['pass'] is not None: L.append(f'   PASS: {res["pass"]}')
    L.append('')
open(f'validation/sgp_{Y}{SUF}.log', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
