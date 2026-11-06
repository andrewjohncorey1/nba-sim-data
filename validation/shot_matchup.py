"""
Matchup test for points props: does the opponent's defense move some players more than others?

Usage (repo root):
  python3 validation/shot_matchup.py zones <season>       player-game shot zones from the play-by-play -> validation/player_zones_<Y>.parquet
  python3 validation/shot_matchup.py tune 2025            scan the strength on 2024-25
  python3 validation/shot_matchup.py test 2026 <zone_lambda> <pos_lambda>    held-out 2025-26

So far a player's points = his share of the team's points, whatever the opponent; the opponent's defense only moves
the team total. Two candidate matchup terms, each re-dividing the team's points (the team total is unchanged):
  zone  his points mix (rim / mid-range / three / free throws, from his earlier games, shrunk to the league mix)
        against the opponent's points allowed per possession in each zone relative to the league (earlier games,
        recency weighted, shrunk). A shooter facing a defense that gives up few threes gets a bigger cut than his
        center. mult_i = (sum_z w_iz f_z / team average of the same)^lambda.
  pos   defense vs position: the opponent's points allowed per possession to guards / forwards / centers relative to
        the league, same form.
Priced exactly as validation/props_minutes.py (minutes model, scoring plays) so only the mean moves. Book-style lines
(season average rounded down + 0.5, 5+ games) and the fixed lines; tune on 2024-25, test once on 2025-26.
"""
import sys, json, re
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd

from matchup import Z, HL, K_DEF, zones as _zones, player_games as _pg, team_poss, asof


def zones(Y):
    out = _zones(Y, f'validation/player_zones_{Y}.parquet'); print(f'zones {Y}: {len(out)} player-games')


def player_games(seasons):
    return _pg(seasons, 'validation/player_zones_{y}.parquet')


def features(P, Y):
    """For every player-game row of the props data: his zone mix and position, the opponent's zone and position factors,
    each from games before that date only."""
    D = P['D'].copy(); D['date'] = pd.to_datetime(D.date); D['game_id'] = D.game_id.astype(str)
    PG = player_games([Y - 1, Y]); TP = team_poss([Y - 1, Y])
    opp_of = PG.drop_duplicates(['game_id', 'team']).set_index(['game_id', 'team']).opp
    D['opp'] = [opp_of.get((g, t)) for g, t in zip(D.game_id, D.team)]
    pos_of = PG.drop_duplicates(['game_id', 'pid']).set_index(['game_id', 'pid']).pg
    D['pg'] = [pos_of.get((g, p), 'F') for g, p in zip(D.game_id, D.pid)]
    W = {k: np.zeros((len(D), 4)) for k in ('mix', 'fz')}; FP = np.ones(len(D))
    for d, idx in D.groupby('date').groups.items():
        rows = D.loc[idx]; k = D.index.get_indexer(idx)
        mix, fz, lg_share = asof(PG, TP, d)
        W['mix'][k] = mix.reindex(rows.pid).fillna(pd.Series(lg_share, index=Z)).values
        W['fz'][k] = fz.reindex(rows.opp).fillna(1.0).values
        # defense vs position (scan only): points allowed per possession to G / F / C relative to the league
        H = PG[PG.date < d]; w = 0.5 ** ((d - H.date).dt.days.values / HL); Hp = TP[TP.date < d]; wp = 0.5 ** ((d - Hp.date).dt.days.values / HL)
        n = pd.Series(Hp.poss.values * wp).groupby(Hp.opp.values).sum().reindex(rows.opp).fillna(0).values
        ap = pd.Series(H.pts.values * w).groupby([H.opp.values, H.pg.values]).sum()
        lgp = pd.Series(H.pts.values * w).groupby(H.pg.values).sum() / (Hp.poss.values * wp).sum()
        a = np.array([ap.get((o, g), 0.0) for o, g in zip(rows.opp, rows.pg)]); lp = lgp.reindex(rows.pg).values
        FP[k] = (a + K_DEF * lp) / (n + K_DEF) / lp
    return D, W['mix'], W['fz'], FP


def multipliers(D, mu_pts, mix, fz, fp, lz, lp):
    """Re-divide each team's expected points: mult_i = (x_i / team-points-weighted average of x)^lambda."""
    xz = (mix * fz).sum(1); out = np.ones(len(D))
    for lam, x in ((lz, xz), (lp, fp)):
        if not lam: continue
        key = D.game_id.values + D.side.values
        s = pd.Series(mu_pts); avg = (s * x).groupby(key).transform('sum') / s.groupby(key).transform('sum')
        out *= (x / avg.values) ** lam
    return out


def setup(Y):
    import props_minutes as PM
    P = pd.read_pickle(f'validation/props2_pre_{Y}.pkl'); T5 = json.load(open('validation/props5_tuned.json'))
    MD = json.load(open(f'validation/minutes_dist_for_{Y}.json')); cfg, ch = T5['cfg'], T5['choice']
    mu, rel = PM.build(P, cfg, MD, ch['minutes']['er'])
    D, mix, fz, fp = features(P, Y)
    return P, PM, cfg, ch, mu, rel, D, mix, fz, fp


def price_all(P, PM, cfg, ch, mu, rel, mult):
    import tune_props2 as T
    D = P['D']; ok = (D.n_season >= 5).values
    m2 = {k: v for k, v in mu.items()}; m2['pts'] = (mu['pts'][0] * mult[:, None], mu['pts'][1] * mult)
    b = D.b_pts.values[ok]; a = D.a_pts.values[ok]
    lines = [np.floor(b) + 0.5] + [np.full(ok.sum(), L) for L in T.FIXED['pts']]
    pr = PM.cp_price(P, cfg, m2, rel, ch['pts']['beta'], lines, ok); y = [(a > L).astype(float) for L in lines]
    return ok, lines, pr, y, PM.tiers_of(b, m2['pts'][1][ok], a, pr[0], y[0])


def lrow(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tune(Y):
    P, PM, cfg, ch, mu, rel, D, mix, fz, fp = setup(Y)
    L = [f'shot-zone and position matchup scan, season ending {Y} (log loss: book line / all lines)']; res = {}
    for name, grid in (('zone', [0, 0.5, 1, 1.5, 2]), ('pos', [0, 0.5, 1, 1.5])):
        for lam in grid:
            mult = multipliers(D, mu['pts'][1], mix, fz, fp, lam if name == 'zone' else 0, lam if name == 'pos' else 0)
            ok, lines, pr, y, tiers = price_all(P, PM, cfg, ch, mu, rel, mult)
            book = lrow(y[0], pr[0]).mean(); allll = np.mean(np.concatenate([lrow(a, b) for a, b in zip(y, pr)]))
            res[f'{name} {lam}'] = {'book': round(float(book), 5), 'all': round(float(allll), 5), 'mult_sd': round(float(mult[ok].std()), 4)}
            L.append(f'{name:4s} lambda {lam:.1f}: book {book:.5f} all {allll:.5f}  (spread of multipliers sd {mult[ok].std():.3f}, 5%-95% {np.percentile(mult[ok], 5):.3f}-{np.percentile(mult[ok], 95):.3f})')
            print(L[-1], flush=True)
    json.dump(res, open(f'validation/shot_matchup_tune_{Y}.json', 'w'), indent=1)
    open(f'validation/shot_matchup_tune_{Y}.log', 'w').write('\n'.join(L) + '\n')


def test(Y, lz, lp):
    P, PM, cfg, ch, mu, rel, D, mix, fz, fp = setup(Y); rng = np.random.default_rng(4)
    base = price_all(P, PM, cfg, ch, mu, rel, np.ones(len(D)))
    L = [f'shot-zone / position matchup test, season ending {Y}: zone lambda {lz}, position lambda {lp} (chosen on 2024-25)']; res = {}
    for name, a, b in (('zone', lz, 0), ('pos', 0, lp), ('both', lz, lp)):
        if (name == 'both' and not (lz and lp)) or (name != 'both' and not (a or b)): continue
        mult = multipliers(D, mu['pts'][1], mix, fz, fp, a, b); ok, lines, pr, y, tiers = price_all(P, PM, cfg, ch, mu, rel, mult)
        g = D.game_id.values[ok]
        for lab, j in (('book line', [0]), ('all lines', list(range(len(lines))))):
            d0 = np.concatenate([lrow(base[3][k], base[2][k]) for k in j]); d1 = np.concatenate([lrow(y[k], pr[k]) for k in j])
            gg = np.concatenate([g] * len(j)); s = pd.Series(d0 - d1).groupby(gg).sum().values; n = len(s)
            bt = np.array([s[rng.integers(0, n, n)].sum() for _ in range(2000)]) / len(d0); lo, hi = np.percentile(bt, [2.5, 97.5])
            res[f'{name} {lab}'] = {'before': round(float(d0.mean()), 5), 'after': round(float(d1.mean()), 5), 'gain': round(float((d0 - d1).mean()), 5),
                                   'ci': [round(float(lo), 5), round(float(hi), 5)]}
            L.append(f'{name:4s} {lab:9s}: {d0.mean():.5f} -> {d1.mean():.5f}  gain {(d0 - d1).mean():+.5f} [{lo:+.5f},{hi:+.5f}]')
        L.append('      tiers [season avg, mean, actual, pred over, actual over] before: ' + ' | '.join(str(t) for t in base[4]))
        L.append('      after: ' + ' | '.join(str(t) for t in tiers)); res[f'{name} tiers'] = tiers
        print('\n'.join(L[-4:]), flush=True)
    json.dump(res, open(f'validation/shot_matchup_test_{Y}.json', 'w'), indent=1)
    open(f'validation/shot_matchup_test_{Y}.log', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'zones': zones(int(a[2]))
    elif a[1] == 'tune': tune(int(a[2]))
    else: test(int(a[2]), float(a[3]), float(a[4]))
