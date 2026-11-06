"""
Props candidate adjustments, each tested on the props harness (validation/props_minutes.py: minutes model, scoring
plays for points, negative binomial for rebounds), so only the one change moves.

Usage (repo root): python3 validation/props_adj.py tune 2025 | test 2026

  usage    teammates out: the model already re-divides a team's points among the players who play (shares are
           normalized), so a missing star's shots go to everyone in proportion. Candidate: high-usage teammates take
           more of them. M = missing rotation players' usual points / team points (players averaging 20+ minutes in
           their last 10 team games who sit tonight); mult_i = 1 + g M (u_i / u_team - 1), u = points per minute.
  hot      recent form: points per minute over his last 3 games against his rate, mult = 1 + g w (z - 1),
           w = minutes / (minutes + 60)
  fouls    foul-prone players' minutes: the spread of his minutes ratios scaled by 1 + d z_pf (his fouls per minute,
           standardized, last 30 games)
  rebmatch rebounds: his offensive / defensive rebound mix against the rebounds this opponent leaves (offensive
           boards it allows, defensive boards its misses give), relative to the league, mult = (his / team avg)^l
Each multiplier keeps the team's expected total (usage, hot, rebmatch are re-normalized within the team-game).
Book-style line (season average rounded down + 0.5, 5+ games) and the fixed lines; tune on 2024-25, test on 2025-26.
"""
import sys, json
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import props_minutes as PM, tune_props2 as T

HLD = 180.0


def box(seasons):
    out = []
    for y in seasons:
        B = pd.read_parquet(f'data/derived/players_{y}.parquet'); B['game_id'] = B.game_id.astype(str); out.append(B)
    B = pd.concat(out, ignore_index=True); B['date'] = pd.to_datetime(B.date); return B


def features(P, Y):
    D = P['D'].copy(); D['date'] = pd.to_datetime(D.date); D['game_id'] = D.game_id.astype(str)
    B = box([Y - 1, Y]); Bp = B[B['min'] > 0]
    tg = pd.concat([pd.read_parquet(f'data/derived/teamgames_{y}.parquet') for y in (Y - 1, Y)]); tg['game_id'] = tg.game_id.astype(str)
    gd = pd.concat([pd.read_parquet(f'data/derived/games_{y}.parquet')[['game_id', 'date']] for y in (Y - 1, Y)]); gd['game_id'] = gd.game_id.astype(str)
    tg['game_id'] = tg.game_id.astype(str); tg = tg.merge(gd, on='game_id'); tg['date'] = pd.to_datetime(tg.date)
    opp_of = B.drop_duplicates(['game_id', 'team']).set_index(['game_id', 'team']).opp
    D['opp'] = [opp_of.get((g, t)) for g, t in zip(D.game_id, D.team)]
    n = len(D); F = {k: np.full(n, np.nan) for k in ('M', 'z3', 'w3', 'zpf', 'wo', 'fo', 'fd')}
    played = set(zip(Bp.game_id, Bp.pid))
    for d, idx in D.groupby('date').groups.items():
        k = D.index.get_indexer(idx); rows = D.loc[idx]
        H = Bp[Bp.date < d]
        # missing rotation players' usual points, per team-game
        last10 = H.sort_values('date').groupby('pid').tail(10)
        avgm = last10.groupby('pid')['min'].mean(); avgp = last10.groupby('pid').pts.mean()
        lastteam = H.sort_values('date').groupby('pid').team.last()
        rot = avgm[avgm >= 20].index
        tp = H.groupby(['team']).apply(lambda x: x.groupby('game_id').pts.sum().tail(10).mean())
        Ms = {}
        for (g, t), _ in rows.groupby(['game_id', 'team']):
            miss = [p for p in rot if lastteam.get(p) == t and (g, p) not in played]
            Ms[(g, t)] = sum(avgp.get(p, 0) for p in miss) / max(tp.get(t, 110), 1)
        F['M'][k] = [Ms[(g, t)] for g, t in zip(rows.game_id, rows.team)]
        # last 3 games
        l3 = H.sort_values('date').groupby('pid').tail(3).groupby('pid')[['pts', 'min']].sum()
        F['z3'][k] = l3.reindex(rows.pid).pts.values; F['w3'][k] = l3.reindex(rows.pid)['min'].values
        # fouls per minute, last 30 games
        l30 = H.sort_values('date').groupby('pid').tail(30).groupby('pid')[['pf', 'min']].sum()
        F['zpf'][k] = ((l30.pf + 3) / (l30['min'] + 36)).reindex(rows.pid).values
        # rebound mix and the opponent's rebounds left, recency weighted
        w = 0.5 ** ((d - H.date).dt.days.values / HLD)
        rb = pd.DataFrame({'o': H.oreb.values * w, 'd': H.dreb.values * w, 'pid': H.pid.values}).groupby('pid').sum()
        lo = rb.o.sum() / (rb.o.sum() + rb.d.sum())
        wo = (rb.o + 20 * lo) / (rb.o + rb.d + 20); F['wo'][k] = wo.reindex(rows.pid).fillna(lo).values
        Ht = tg[tg.date < d]; wt = 0.5 ** ((d - Ht.date).dt.days.values / HLD)
        # as a defense (opp column = the defense): offensive rebounds it allows per possession; as an offense: misses
        # it leaves for the other team (missed FG + missed last FT - own offensive rebounds) per possession
        orb = (Ht.oreb_fg + Ht.oreb_ft).values
        x = pd.DataFrame({'team': Ht.team.values, 'opp': Ht.opp.values, 'poss': Ht.poss.values * wt, 'orb': orb * wt,
                          'left': (Ht.miss_fg.values + Ht.miss_ftlast.values - orb) * wt})
        allow_o = x.groupby('opp')[['orb', 'poss']].sum(); give_d = x.groupby('team')[['left', 'poss']].sum()
        lo_pp = x.orb.sum() / x.poss.sum(); ld_pp = x.left.sum() / x.poss.sum()
        fo = ((allow_o.orb + 1000 * lo_pp) / (allow_o.poss + 1000)) / lo_pp; fd = ((give_d.left + 1000 * ld_pp) / (give_d.poss + 1000)) / ld_pp
        F['fo'][k] = fo.reindex(rows.opp).fillna(1).values; F['fd'][k] = fd.reindex(rows.opp).fillna(1).values
    return D, F


def renorm(mult, mu, D):
    key = D.game_id.values + D.side.values; s = pd.Series(mu)
    return mult * (s.groupby(key).transform('sum') / (s * mult).groupby(key).transform('sum')).values


def mults(D, F, mu_pts, mu_reb, rate_pts, kind, g):
    if kind == 'usage':
        key = D.game_id.values + D.side.values; mins = mu_pts / np.maximum(rate_pts, 1e-6)
        ut = (pd.Series(mu_pts).groupby(key).transform('sum') / pd.Series(mins).groupby(key).transform('sum')).values
        m = 1 + g * F['M'] * (rate_pts / ut - 1); return renorm(np.clip(m, 0.5, 2), mu_pts, D)
    if kind == 'hot':
        z = np.where(F['w3'] > 0, (F['z3'] / np.maximum(F['w3'], 1)) / np.maximum(rate_pts, 1e-3), 1.0)
        w = np.nan_to_num(F['w3']) / (np.nan_to_num(F['w3']) + 60)
        m = 1 + g * w * (np.clip(np.nan_to_num(z, nan=1.0), 0.3, 3) - 1); return renorm(m, mu_pts, D)
    if kind == 'rebmatch':
        x = F['wo'] * F['fo'] + (1 - F['wo']) * F['fd']; key = D.game_id.values + D.side.values; s = pd.Series(mu_reb)
        avg = ((s * x).groupby(key).transform('sum') / s.groupby(key).transform('sum')).values
        return renorm((x / avg) ** g, mu_reb, D)


def lrow(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(P, cfg, ch, mu, rel, market, mult=None, relw=None):
    D = P['D']; ok = (D.n_season >= 5).values
    k = 'pts' if market == 'pts' else 'reb'
    nodes, m = mu[k][0], mu[k][1]
    if mult is not None: nodes, m = nodes * mult[:, None], m * mult
    r = rel if relw is None else relw
    b = D[f'b_{k}'].values[ok]; a = D[f'a_{k}'].values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED[k]]; y = [(a > L).astype(float) for L in lines]
    if k == 'pts':
        m2 = dict(mu); m2['pts'] = (nodes, m); pr = PM.cp_price(P, cfg, m2, r, ch['pts']['beta'], lines, ok)
    else:
        c = ch['reb']; pr = [PM.price(nodes[ok][:, PM.NI], m[ok], r[ok], c['beta'], c['f'], L) for L in lines]
    return ok, y, pr, PM.tiers_of(b, m[ok], a, pr[0], y[0])


GRID = {'usage': [-1.0, -0.5, 0.5, 1.0, 2.0], 'hot': [0.05, 0.1, 0.25, 0.5], 'fouls': [-0.4, -0.2, 0.2, 0.4], 'rebmatch': [0.5, 1.0, 1.5, 2.5]}


def run(Y, picks=None):
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); T5 = json.load(open('validation/props5_tuned.json'))
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json')); cfg, ch = T5['cfg'], T5['choice']
    mu, rel = PM.build(P, cfg, MD, ch['minutes']['er']); D, F = features(P, Y)
    alloc = T.project_minutes(P, cfg['min']); rate = mu['pts'][1] / np.maximum(alloc, 1)
    base = {mk: evaluate(P, cfg, ch, mu, rel, mk) for mk in ('pts', 'reb')}
    rng = np.random.default_rng(2); L = [f'props candidates, season ending {Y}: log loss gain vs current (positive = better), book line / all lines']; res = {}
    zpf = (F['zpf'] - np.nanmean(F['zpf'])) / np.nanstd(F['zpf']); zpf = np.clip(np.nan_to_num(zpf), -2, 2)
    for kind, grid in GRID.items():
        mk = 'reb' if kind == 'rebmatch' else 'pts'
        for g in (grid if picks is None else [picks[kind]]):
            if kind == 'fouls':
                rw = 1 + (rel - 1) * (1 + g * zpf)[:, None]; rw = np.maximum(rw, 0.02); rw = rw / rw.mean(1, keepdims=True)
                ok, y, pr, tiers = evaluate(P, cfg, ch, mu, rel, mk, relw=rw)
            else:
                ok, y, pr, tiers = evaluate(P, cfg, ch, mu, rel, mk, mult=mults(D, F, mu['pts'][1], mu['reb'][1], rate, kind, g))
            _, y0, p0, t0 = base[mk]; gid = D.game_id.values[ok]; e = {}
            for lab, J in (('book', [0]), ('all', range(len(y)))):
                d = np.concatenate([lrow(y0[j], p0[j]) - lrow(y[j], pr[j]) for j in J]); gg = np.concatenate([gid] * len(J))
                s = pd.Series(d).groupby(gg).sum().values; n = len(s)
                bt = np.array([s[rng.integers(0, n, n)].sum() for _ in range(1000)]) / len(d)
                e[lab] = [round(float(d.mean()), 5), round(float(np.percentile(bt, 2.5)), 5), round(float(np.percentile(bt, 97.5)), 5)]
            e['tiers'] = tiers; res[f'{kind} {g}'] = e
            L.append(f'{kind:8s} {g:+.2f} ({mk}): book {e["book"][0]:+.5f} [{e["book"][1]:+.5f},{e["book"][2]:+.5f}]  all {e["all"][0]:+.5f} [{e["all"][1]:+.5f},{e["all"][2]:+.5f}]')
            print(L[-1], flush=True)
    if picks is None:
        pk = {kind: max(GRID[kind], key=lambda g: res[f'{kind} {g}']['all'][0] + res[f'{kind} {g}']['book'][0]) for kind in GRID}
        L.append('picked on 2024-25: ' + json.dumps(pk)); json.dump(pk, open('validation/props_adj_pick_2025.json', 'w'))
        print(L[-1])
    L.append('missing-teammate share M: games with M > 0.15: ' + str(int((F['M'] > 0.15).sum())) + ' player-games')
    json.dump(res, open(f'validation/props_adj_{Y}.json', 'w'), indent=1)
    open(f'validation/props_adj_{Y}.log', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    Y = int(sys.argv[2]); run(Y, None if sys.argv[1] == 'tune' else json.load(open('validation/props_adj_pick_2025.json')))
