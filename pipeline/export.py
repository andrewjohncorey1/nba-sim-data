"""
Fit current ratings and write build/ratings.json for the app and pipeline/predict.js.

Usage (repo root): python3 pipeline/export.py <season>
Contents: team component ratings and pace (pipeline/ratings.py, settings from pipeline/settings.json), engine
settings (pipeline/engine_params.json + calibration), player impact and minutes for availability
(pipeline/players.py), per-minute player rates for props, rest / back-to-back effects that passed testing, and the
held-out test tables shown in the app.
"""
import sys, json, os
import numpy as np, pandas as pd
import ratings as RT, players as PS, matchup as MU

Y = int(sys.argv[1])
S = json.load(open('pipeline/settings.json'))
E = json.load(open('pipeline/engine_params.json'))
P = S['model']
now = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize() + pd.Timedelta(days=1)   # every game up to today
TG = RT.load_tg(range(Y - 3, Y + 1))
R = RT.fit(TG, now, P['ratings'])
PL = PS.load(range(Y - 3, Y + 1))
prior = json.load(open('pipeline/player_prior.json'))
PR = PS.fit(PL, now, P['players'], prior=prior)
low = PR[PR.wmin < 300]
repl = [float(np.average(low.O, weights=low.wmin + 1)), float(np.average(low.D, weights=low.wmin + 1))]
pm = PS.projected_minutes(PL, now)
last_game = PL[PL['min'] > 0].date.max()

# ---------- per-minute rates for props (recency-weighted, shrunk toward the position average) ----------
pp = S['props']
x = PL[(PL['min'] > 0) & (PL.date < now)].copy()
x['w'] = PS.weights(x.date, now, pp['half_life'])
x = x[x.w > 1e-3]
x['posg'] = x.pos.fillna('F').str[0].map({'G': 'G', 'F': 'F', 'C': 'C'}).fillna('F')
stats = ['pts', 'reb', 'ast', 'fg3m', 'fgm']
agg = x.assign(wmin=x.w * x['min'], **{f'w_{s}': x.w * x[s] for s in stats}).groupby('pid')
A = agg[['wmin'] + [f'w_{s}' for s in stats]].sum()
pos = x.sort_values('date').groupby('pid').posg.last()
posavg = {}
for g, xx in x.groupby('posg'):
    posavg[g] = {s: float((xx.w * xx[s]).sum() / (xx.w * xx['min']).sum()) for s in stats}
rates = {}
PV2 = pp.get('v2')          # tuned settings (validation/tune_props2.py): own long-run rate as the shrinkage target
if PV2:
    def wsum(h):
        xx = PL[(PL['min'] > 0) & (PL.date < now)]; w = PS.weights(xx.date, now, h); xx = xx.assign(w=w)[w > 1e-3]
        return xx.assign(wmin=xx.w * xx['min'], **{f'w_{s}': xx.w * xx[s] for s in stats}).groupby('pid')[['wmin'] + [f'w_{s}' for s in stats]].sum()
    AH = {h: wsum(h) for h in {PV2['rate'][s]['hl'] for s in stats} | {720}}
    for pid in AH[720].index:
        pa = posavg.get(pos.get(pid, 'F'), posavg['F']); out = {}
        for s in stats:
            rc = PV2['rate'][s]; L = AH[720].loc[pid]
            tgt = (L[f'w_{s}'] + 300 * pa[s]) / (L.wmin + 300) if rc['target'] == 'own' else pa[s]
            Ah = AH[rc['hl']]
            out[s] = (Ah.at[pid, f'w_{s}'] + rc['K'] * tgt) / (Ah.at[pid, 'wmin'] + rc['K']) if pid in Ah.index else tgt
        rates[pid] = out
    tgt = None
    if PV2['min'].get('target') == 'season':     # prior toward the player's own season average minutes (else last season's)
        played = PL[(PL['min'] > 0) & (PL.date < now)]
        tgt = played[played.season == Y].groupby('pid')['min'].mean().combine_first(played[played.season == Y - 1].groupby('pid')['min'].mean())
    pmp = PS.projected_minutes(PL, now, hl_games=PV2['min']['hl'], prior_games=PV2['min']['pw'], prior_min=14.0, prior_target=tgt)
else:
    for pid, r in A.iterrows():
        pa = posavg.get(pos.get(pid, 'F'), posavg['F'])
        rates[pid] = {s: (r[f'w_{s}'] + pp['prior_min'] * pa[s]) / (r.wmin + pp['prior_min']) for s in stats}
    pmp = pm

# ---------- rosters: players who appeared for the team this season (most recent team), else last season ----------
cur = PL[PL.season == Y].sort_values('date')
last_team = cur.groupby('pid').team.last()
prev = PL[PL.season == Y - 1].sort_values('date').groupby('pid').team.last()
names = PL.sort_values('date').groupby('pid').name.last()
posn = PL.sort_values('date').groupby('pid').pos.last()
# chance a healthy player gets in the game: played / (played + coach's-decision DNPs) over his last 15 team games
recent = PL[(PL.date < now) & (PL.season >= Y - 1)].sort_values('date')
recent = recent[(recent['min'] > 0) | (recent.reason.str.upper().str.startswith("COACH"))]
recent = recent.groupby('pid').tail(15)
pplay = ((recent['min'] > 0).groupby(recent.pid).sum() + 1) / (recent.groupby('pid').size() + 1)
teams = []
for t in RT.TEAMS:
    sb = PS.team_shares(PL, t, now, P['ratings']['half_life'])
    ids = list(last_team[last_team == t].index)
    roster = []
    for pid in ids:
        rr = rates.get(pid, posavg['F'])
        roster.append(dict(pid=int(pid), name=names.get(pid, str(pid)), pos=posn.get(pid) or '',
                           O=round(float(PR.O.get(pid, repl[0])), 3), D=round(float(PR.D.get(pid, repl[1])), 3),
                           pm=round(float(pm.proj.get(pid, 12.0)), 1), pmp=round(float(pmp.proj.get(pid, 12.0)), 1), pp=round(float(pplay.get(pid, 0.5)), 3),
                           last=str(pm['last'].get(pid).date()) if pid in pm.index else None,
                           r={k: round(float(v), 4) for k, v in rr.items()}))
    roster.sort(key=lambda p: -p['pm'])
    sbar = [[int(pid), round(float(v), 4), round(float(PR.O.get(pid, repl[0])), 3), round(float(PR.D.get(pid, repl[1])), 3)]
            for pid, v in sb.items() if v > 1e-4]
    teams.append(dict(abbr=t, name=S['team_names'][t][0], roster=roster, sbar=sbar))

# previous-season roster lookup (names -> pid) for injury-list matching of players with no games yet this season
pool = {int(pid): dict(name=names.get(pid, ''), team=last_team.get(pid, prev.get(pid)),
                       O=round(float(PR.O.get(pid, repl[0])), 3), D=round(float(PR.D.get(pid, repl[1])), 3),
                       pm=round(float(pm.proj.get(pid, 12.0)), 1), pmp=round(float(pmp.proj.get(pid, 12.0)), 1), pp=round(float(pplay.get(pid, 0.5)), 3), pos=posn.get(pid) or '',
                       r={k: round(float(v), 4) for k, v in rates.get(pid, posavg['F']).items()})
        for pid in PR.index if (PR.wmin.get(pid, 0) > 0)}
# minutes distribution by role for the props (validation/minutes_dist.py; 11 of the 21 stored quantiles)
mdist = None
if os.path.exists('pipeline/minutes_dist.json'):
    _m = json.load(open('pipeline/minutes_dist.json'))
    mdist = {'edges': _m['edges'], 'q': [q[0::2] for q in _m['q']], 'fit_on': _m['fit_on']}
mdistc = None          # the same by final-margin bucket (validation/minutes_dist.py fitc; validation/sgp3.py)
if os.path.exists('pipeline/minutes_distc.json'):
    _m = json.load(open('pipeline/minutes_distc.json'))
    mdistc = {'edges': _m['edges'], 'medges': _m['medges'], 'share': _m['share'], 'q': [[q[0::2] for q in row] for row in _m['q']], 'fit_on': _m['fit_on']}
# shot-zone matchup for points props (pipeline/matchup.py): players' zone mix and teams' zone defense as of today
zone = {}
if pp.get('zone_lambda'):
    for y in (Y - 1, Y):
        if y == Y or not os.path.exists(f'data/derived/zones_{y}.parquet'):
            if os.path.exists(f'data/raw/play_by_play_{y}.parquet'): MU.zones(y)
    zmix, zdef, zlg = MU.asof(MU.player_games([y for y in (Y - 1, Y) if os.path.exists(f'data/derived/zones_{y}.parquet')]),
                              MU.team_poss([Y - 1, Y]), now)
    zone = dict(zdef={t: [round(float(v), 4) for v in zdef.loc[t]] if t in zdef.index else [1, 1, 1, 1] for t in RT.TEAMS},
                zlg=[round(float(v), 4) for v in zlg])
    for t in teams:
        for p in t['roster']:
            if p['pid'] in zmix.index: p['zm'] = [round(float(v), 4) for v in zmix.loc[p['pid']]]
    for pid, p in pool.items():
        if pid in zmix.index: p['zm'] = [round(float(v), 4) for v in zmix.loc[pid]]
# recent form for points props (validation/props_adj.py 'hot'): points and minutes over each player's last 3 games
if pp.get('hot_g'):
    l3 = PL[(PL['min'] > 0) & (PL.date < now)].sort_values('date').groupby('pid').tail(3).groupby('pid')[['pts', 'min']].sum()
    for t in teams:
        for p in t['roster']:
            if p['pid'] in l3.index: p['l3'] = [round(float(l3.at[p['pid'], 'pts']), 1), round(float(l3.at[p['pid'], 'min']), 1)]
    for pid, p in pool.items():
        if pid in l3.index: p['l3'] = [round(float(l3.at[pid, 'pts']), 1), round(float(l3.at[pid, 'min']), 1)]
cal = json.load(open('pipeline/calib.json')) if os.path.exists('pipeline/calib.json') else {}
out = dict(asOf=str(last_game.date()), generated=pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), season=Y,
           teams=RT.TEAMS, names={t: S['team_names'][t][0] for t in RT.TEAMS},
           comp={k: {'mu': R[k]['mu'], 'h': R[k]['h'], 'O': [round(v, 5) for v in R[k]['O']], 'D': [round(v, 5) for v in R[k]['D']]}
                 for k in RT.COMPS},
           pace={'mu': R['pace']['mu'], 'P': [round(v, 5) for v in R['pace']['P']]},
           engine={**E, **cal}, avail=dict(alpha=P['ratings']['alpha'], repl=repl, ppp0=E['base_ppp'], sens=E['ppp_sens']),
           rest=S.get('rest_effects', {}), props=dict(**S.get('props', {}), posavg=posavg, mdist=mdist, mdistc=mdistc, **zone),
           rosters=teams, pool=pool, tests=S.get('tests', {}), notes=S.get('notes', {}))
json.dump(out, open('build/ratings.json', 'w'), separators=(',', ':'))
nr = {t['abbr']: t for t in teams}
print(f'EXPORT ratings as of {out["asOf"]} ({R["n_games"]} weighted games); {sum(len(t["roster"]) for t in teams)} rostered players, '
      f'{len(pool)} in the player pool; ratings.json {os.path.getsize("build/ratings.json") // 1024} KB')
