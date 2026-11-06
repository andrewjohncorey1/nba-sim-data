"""
Shot-zone matchup for points props (validation/shot_matchup.py; kept Nov 6, 2026).

A player's points mix (rim / mid-range / three / free throws, recency weighted, shrunk to the league mix) against
the opponent's points allowed per possession in each zone relative to the league (as a defense, shrunk to 1). The
engine re-divides each team's expected points with mult_i = (sum_z w_iz f_z / team average)^lambda (props.zone_lambda).
zones(Y) builds data/derived/zones_<Y>.parquet (made field goals by zone per player-game) from the play-by-play.
"""
import json, re
import numpy as np, pandas as pd

S = json.load(open('pipeline/settings.json'))
TID = {int(k): v for k, v in S['team_ids'].items()}
FOOT = re.compile(r'(\d+)-foot')
RIMWORDS = ('Layup', 'Dunk', 'Tip', 'Finger Roll', 'Putback', 'Alley Oop')
Z = ['rim', 'mid', 'three', 'ft']
HL = 180.0                  # days, recency half-life for both the player's mix and the opponent's defense
K_MIX, K_DEF = 60.0, 1000.0  # shrinkage: points of league mix for a player; possessions of league average for a defense


def zones(Y, path=None):
    """Made field goals by zone for every shooter and game (same zone rules as pipeline/prep.py)."""
    cols = ['game_id', 'type_text', 'text', 'scoring_play', 'score_value', 'shooting_play', 'points_attempted', 'team_id',
            'coordinate_x', 'coordinate_y', 'home_team_id', 'away_team_id', 'season_type', 'athlete_id_1']
    p = pd.read_parquet(f'data/raw/play_by_play_{Y}.parquet', columns=cols)
    p = p[p.home_team_id.isin(TID) & p.away_team_id.isin(TID) & p.season_type.isin([2, 3, 5])].copy()
    t = p.type_text.fillna(''); is_ft = t.str.startswith('Free Throw'); txt = p.text.fillna('').str.lower()
    cx, cy = p.coordinate_x.astype(float), p.coordinate_y.astype(float)
    cd = np.sqrt((cx.abs() - 41.75) ** 2 + cy ** 2); feet = txt.str.extract(FOOT)[0].astype(float)
    three = (txt.str.contains('three point') | (p.score_value == 3) | (feet >= 24)
             | ((cx.abs() < 60) & ((cd >= 23.6) | ((cy.abs() >= 21.8) & (cx.abs() >= 32)))))
    pa = np.where(p.points_attempted.isin([2, 3]), p.points_attempted, np.where(three, 3, 2))
    pa = np.where(p.shooting_play.fillna(False) & ~is_ft, pa, 0)
    fga = p.shooting_play.fillna(False) & ~is_ft & np.isin(pa, [2, 3]) & ~t.str.startswith('No Shot')
    ok = cx.abs() < 60
    dist = np.where(ok, cd, np.nan); dist = np.where(np.isnan(dist), feet, dist)
    rimtype = t.apply(lambda s: any(w in s for w in RIMWORDS))
    dist = np.where(np.isnan(dist), np.where(rimtype, 1.0, 12.0), dist)
    zone = np.where(pa == 3, 3, np.where(dist <= 4.0, 1, 2))
    made = p.scoring_play.fillna(False)
    E = pd.DataFrame(dict(game_id=p.game_id.astype(str), pid=pd.to_numeric(p.athlete_id_1, errors='coerce'), zone=zone, made=made))[fga.values]
    E = E[E.pid.notna()]; E['pid'] = E.pid.astype(int)
    out = E[E.made].groupby(['game_id', 'pid', 'zone']).size().unstack(fill_value=0).rename(columns={1: 'fgm1', 2: 'fgm2', 3: 'fgm3'}).reset_index()
    out.to_parquet(path or f'data/derived/zones_{Y}.parquet', index=False); return out


def player_games(seasons, zpath='data/derived/zones_{y}.parquet'):
    """Points by zone per player-game (box free throws), with date, team, opponent, position group."""
    out = []
    for y in seasons:
        B = pd.read_parquet(f'data/derived/players_{y}.parquet'); B = B[B['min'] > 0].copy(); B['game_id'] = B.game_id.astype(str)
        Zn = pd.read_parquet(zpath.format(y=y))
        B = B.merge(Zn, on=['game_id', 'pid'], how='left').fillna({'fgm1': 0, 'fgm2': 0, 'fgm3': 0})
        B['rim'], B['mid'], B['three'], B['ft'] = 2 * B.fgm1, 2 * B.fgm2, 3 * B.fgm3, B.ftm
        B['pg'] = B.pos.fillna('F').str[-1].map({'G': 'G', 'F': 'F', 'C': 'C'}).fillna('F')
        out.append(B[['game_id', 'date', 'team', 'opp', 'pid', 'pg', 'pts'] + Z])
    R = pd.concat(out, ignore_index=True); R['date'] = pd.to_datetime(R.date); return R


def team_poss(seasons):
    out = []
    for y in seasons:
        tg = pd.read_parquet(f'data/derived/teamgames_{y}.parquet'); g = pd.read_parquet(f'data/derived/games_{y}.parquet')[['game_id', 'date']]
        tg = tg.merge(g, on='game_id'); tg['game_id'] = tg.game_id.astype(str); out.append(tg[['game_id', 'date', 'team', 'opp', 'poss']])
    R = pd.concat(out, ignore_index=True); R['date'] = pd.to_datetime(R.date); return R


def asof(PG, TP, d):
    """As of date d (games before d only): every player's points mix by zone (shrunk to the league mix) and every
    team's points allowed per possession by zone relative to the league, as a defense (shrunk to 1)."""
    H = PG[PG.date < d]; w = 0.5 ** ((d - H.date).dt.days.values / HL)
    lg_z = (H[Z].values * w[:, None]).sum(0); lg_share = lg_z / lg_z.sum()
    pm = pd.DataFrame(H[Z].values * w[:, None], columns=Z).assign(pid=H.pid.values).groupby('pid').sum()
    mix = (pm + K_MIX * lg_share).div(pm.sum(1) + K_MIX, axis=0)
    Hp = TP[TP.date < d]; wp = 0.5 ** ((d - Hp.date).dt.days.values / HL)
    poss_d = pd.Series(Hp.poss.values * wp).groupby(Hp.opp.values).sum()
    lg_pp = lg_z / (Hp.poss.values * wp).sum()
    az = pd.DataFrame(H[Z].values * w[:, None], columns=Z).groupby(H.opp.values).sum()
    n = poss_d.reindex(az.index).fillna(0).values[:, None]
    fz = pd.DataFrame((az.values + K_DEF * lg_pp) / (n + K_DEF) / lg_pp, index=az.index, columns=Z)
    return mix, fz, lg_share
