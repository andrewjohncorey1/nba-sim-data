"""
Minutes distribution by role (props step 4, "minutes model").

A player's minutes on a night = his projected minutes (the step-2 projection, fitted to the team's 240 among the
players who played) x a ratio r. The distribution of r is learned per projected-minutes band (role) from earlier
seasons only: a starter has a stable role with occasional short nights (early exits, foul trouble, blowouts); a bench
player's minutes swing far more.

Usage (repo root):
  python3 validation/minutes_dist.py rows <season>     player-game rows (projection, actual minutes) -> validation/minutes_rows_<Y>.parquet
  python3 validation/minutes_dist.py fit <target>      fit on seasons 2023..target-1 -> validation/minutes_dist_for_<target>.json
"""
import sys, json
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import players as PS

EDGES = [0, 12, 18, 24, 30, 34, 48]
NQ = 21
STEP2_MIN = dict(hl_games=4, prior_games=0.5, prior_min=14.0)     # validation/props2_tuned.json minutes


def rows(Y):
    PL = PS.load(range(Y - 2, Y + 1))
    cur = PL[(PL.season == Y) & (PL['min'] > 0)]
    out = []
    for d, gd in cur.groupby('date'):
        pm = PS.projected_minutes(PL, d, **STEP2_MIN)
        nh = PL[(PL.date < d) & (PL['min'] > 0)].groupby('pid').size()
        for (gid, team), tp in gd.groupby(['game_id', 'team']):
            proj = pm.proj.reindex(tp.pid).fillna(12.0).values
            alloc = PS.allocate(proj)
            out.append(pd.DataFrame(dict(season=Y, date=d, game_id=gid, team=team, pid=tp.pid.values, alloc=alloc,
                                         min=tp['min'].values, nhist=nh.reindex(tp.pid).fillna(0).values)))
    R = pd.concat(out, ignore_index=True)
    R.to_parquet(f'validation/minutes_rows_{Y}.parquet', index=False)
    print(f'rows {Y}: {len(R)} player-games')


def fit(target):
    R = pd.concat([pd.read_parquet(f'validation/minutes_rows_{y}.parquet') for y in range(2023, target)], ignore_index=True)
    R = R[R.nhist >= 5]
    R['r'] = (R['min'] / R.alloc).clip(0, None)
    R['band'] = pd.cut(R.alloc, EDGES, labels=False, include_lowest=True)
    q = (np.arange(NQ) + 0.5) / NQ
    out = {'fit_on': f'2023..{target - 1}', 'edges': EDGES, 'q': [], 'mean': [], 'n': []}
    for b in range(len(EDGES) - 1):
        x = R[R.band == b]
        top = 48 / max(EDGES[b], 6)                                   # no more than 48 minutes
        rq = np.clip(np.quantile(x.r, q), 0.02, top)
        out['q'].append([round(float(v), 4) for v in rq]); out['mean'].append(round(float(rq.mean()), 4)); out['n'].append(int(len(x)))
    json.dump(out, open(f'validation/minutes_dist_for_{target}.json', 'w'), indent=1)
    print(f'fit for {target}: ' + '; '.join(f'band {EDGES[b]}-{EDGES[b + 1]}: n {out["n"][b]}, mean {out["mean"][b]}, 5%/50%/95% '
                                          f'{out["q"][b][1]}/{out["q"][b][10]}/{out["q"][b][19]}' for b in range(len(EDGES) - 1)))


MEDGES = [0, 10, 20, 200]      # |final margin| buckets: 0-9, 10-19, 20+


def fitc(target):
    """Minutes ratio quantiles by role band AND final-margin bucket (blowouts), seasons 2023..target-1."""
    R = pd.concat([pd.read_parquet(f'validation/minutes_rows_{y}.parquet') for y in range(2023, target)], ignore_index=True)
    G = pd.concat([pd.read_parquet(f'data/derived/games_{y}.parquet')[['game_id', 'home_pts', 'away_pts']] for y in range(2023, target)])
    R = R[R.nhist >= 5].merge(G, on='game_id')
    R['r'] = (R['min'] / R.alloc).clip(0, None); R['am'] = (R.home_pts - R.away_pts).abs()
    R['band'] = pd.cut(R.alloc, EDGES, labels=False, include_lowest=True)
    R['mb'] = pd.cut(R.am, MEDGES, labels=False, right=False)
    q = (np.arange(NQ) + 0.5) / NQ
    out = {'fit_on': f'2023..{target - 1}', 'edges': EDGES, 'medges': MEDGES, 'q': [], 'n': [], 'share': []}
    for b in range(len(EDGES) - 1):
        top = 48 / max(EDGES[b], 6); row, nn = [], []
        for m in range(len(MEDGES) - 1):
            x = R[(R.band == b) & (R.mb == m)]
            row.append([round(float(v), 4) for v in np.clip(np.quantile(x.r, q), 0.02, top)]); nn.append(int(len(x)))
        out['q'].append(row); out['n'].append(nn)
    out['share'] = [round(float((R.mb == m).mean()), 4) for m in range(len(MEDGES) - 1)]
    json.dump(out, open(f'validation/minutes_distc_for_{target}.json', 'w'), indent=1)
    for b in range(len(EDGES) - 1):
        print(f'band {EDGES[b]}-{EDGES[b + 1]}: median ratio by margin 0-9 / 10-19 / 20+: ' +
              ' / '.join(f'{out["q"][b][m][10]:.3f}' for m in range(3)) + '   5%: ' + ' / '.join(f'{out["q"][b][m][1]:.3f}' for m in range(3)))


if __name__ == '__main__':
    {'rows': rows, 'fit': fit, 'fitc': fitc}[sys.argv[1]](int(sys.argv[2]))
