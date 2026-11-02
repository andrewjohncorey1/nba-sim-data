"""
Rest, back-to-back and travel features for every scheduled game (past and future), from the schedules.

features(seasons) -> DataFrame per game: rest_h, rest_a (days since the team's previous game, capped at 4; 4 for a
season opener), b2b_h, b2b_a, trav_a / trav_h (miles from the previous game's arena), tz_a / tz_h (time zones
moved since the previous game, signed: + = moved east).
Game location = the home team's arena (neutral-site games: the home team's arena is used too; few games).
"""
import json, math, os
import numpy as np, pandas as pd

S = json.load(open('pipeline/settings.json'))
TID = {int(k): v for k, v in S['team_ids'].items()}
AR = S['arenas']


def miles(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(h))


def features(seasons):
    rows = []
    for y in seasons:
        s = pd.read_parquet(f'data/raw/nba_schedule_{y}.parquet') if os.path.exists(f'data/raw/nba_schedule_{y}.parquet') else None
        if s is None:
            g = pd.read_parquet(f'data/derived/games_{y}.parquet')
            s = pd.DataFrame(dict(game_id=g.game_id, st=g.start, home=g.home, away=g.away))
        else:
            s = s[s.home_id.isin(TID) & s.away_id.isin(TID) & s.season_type.isin([2, 3, 5])]
            s = pd.DataFrame(dict(game_id=s.game_id.astype(int), st=pd.to_datetime(s.start_date, utc=True),
                                  home=s.home_id.map(TID), away=s.away_id.map(TID)))
        s['season'] = y
        rows.append(s.drop_duplicates('game_id'))
    s = pd.concat(rows, ignore_index=True)
    s['day'] = s.st.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    long = pd.concat([s.assign(team=s.home, side='h'), s.assign(team=s.away, side='a')]).sort_values(['team', 'st'])
    long['loc'] = long.home
    long['prev_day'] = long.groupby(['team', 'season']).day.shift()
    long['prev_loc'] = long.groupby(['team', 'season']).loc.shift()
    long['rest'] = ((long.day - long.prev_day).dt.days).fillna(4).clip(upper=4)
    long['b2b'] = (long.rest == 1).astype(int)
    long['trav'] = [0.0 if pd.isna(p) else miles(AR[p], AR[l]) for p, l in zip(long.prev_loc, long['loc'])]
    long['tz'] = [0 if pd.isna(p) else AR[l][2] - AR[p][2] for p, l in zip(long.prev_loc, long['loc'])]
    out = s[['game_id', 'st', 'home', 'away', 'season']].copy()
    for sd in ('h', 'a'):
        x = long[long.side == sd].set_index('game_id')[['rest', 'b2b', 'trav', 'tz']].add_suffix('_' + sd)
        out = out.join(x, on='game_id')
    return out
