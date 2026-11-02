"""
Turn the ESPN play-by-play and box scores (sportsdataverse hoopR releases) into the tables the model uses.

Usage (repo root): python3 pipeline/prep.py <season_end_year> [<season_end_year> ...]
Reads  data/raw/{play_by_play,team_box,player_box,nba_schedule}_<Y>.parquet
Writes data/derived/games_<Y>.parquet     one row per game (teams, score, quarter scores, OT, neutral site)
       data/derived/teamgames_<Y>.parquet one row per game and offense: possessions and every component count
       data/derived/poss_<Y>.parquet      one row per possession (start time, margin, duration, outcome)
       data/derived/players_<Y>.parquet   one row per player and game from the box score (incl. DNP and reason)

Event rules (ESPN play-by-play):
- Field goal attempt: shooting_play with points_attempted 2 or 3 (free throws excluded). Zone: three if 3 points;
  otherwise rim if the shot was within 4 feet of the hoop (from coordinates, else the "N-foot" text, else layup/dunk/tip
  types), mid-range for the rest.
- Free throws: "x of 2" / "x of 3" trips (incl. flagrant and clear path), "1 of 1" (and-one), technical (kept apart:
  they are not part of a possession).
- Turnover: any type with "Turnover" plus Traveling (team turnovers included), not "No Turnover".
- Offensive-rebound chances: every missed field goal and every missed last free throw of a trip; the next rebound
  event says who got it. Rebounds after a missed non-final free throw are ignored (dead-ball team rebounds).
- Possessions: the offense is the team shooting / turning it over / rebounding offensively, or the team getting the
  defensive rebound. A possession starts when the offense changes or a period starts.
"""
import sys, json, re
import numpy as np, pandas as pd

S = json.load(open('pipeline/settings.json'))
TID = {int(k): v for k, v in S['team_ids'].items()}
FOOT = re.compile(r'(\d+)-foot')
RIMWORDS = ('Layup', 'Dunk', 'Tip', 'Finger Roll', 'Putback', 'Alley Oop')


def clock_sec(v):
    v = str(v)
    if ':' in v:
        m, s = v.split(':'); return int(m) * 60 + float(s)
    try: return float(v)
    except ValueError: return 0.0


def per_len(p): return 720 if p <= 4 else 300
def per_start(p): return (p - 1) * 720 if p <= 4 else 2880 + (p - 5) * 300


def prep(Y):
    cols = ['game_id', 'game_play_number', 'sequence_number', 'type_text', 'text', 'scoring_play', 'score_value', 'shooting_play',
            'points_attempted', 'team_id', 'period_number', 'clock_display_value', 'coordinate_x', 'coordinate_y',
            'home_team_id', 'away_team_id', 'home_score', 'away_score', 'season_type', 'game_date', 'athlete_id_1']
    p = pd.read_parquet(f'data/raw/play_by_play_{Y}.parquet', columns=cols)
    p = p[p.home_team_id.isin(TID) & p.away_team_id.isin(TID) & p.season_type.isin([2, 3, 5])].copy()
    p['seq'] = pd.to_numeric(p.sequence_number, errors='coerce')
    p = p.sort_values(['game_id', 'period_number', 'seq', 'game_play_number']).reset_index(drop=True)
    t = p.type_text.fillna('')
    p['sec'] = p.clock_display_value.map(clock_sec)
    p['el'] = p.period_number.map(per_start) + p.period_number.map(per_len) - p.sec
    is_ft = t.str.startswith('Free Throw')
    # points_attempted is empty before 2024-25: a shot is a three when the text says so
    txt = p.text.fillna('').str.lower()
    cx, cy = p.coordinate_x.astype(float), p.coordinate_y.astype(float)
    cd = np.sqrt((cx.abs() - 41.75) ** 2 + cy ** 2)
    feet = txt.str.extract(FOOT)[0].astype(float)
    three = (txt.str.contains('three point') | (p.score_value == 3) | (feet >= 24)
             | ((cx.abs() < 60) & ((cd >= 23.6) | ((cy.abs() >= 21.8) & (cx.abs() >= 32)))))
    pa = np.where(p.points_attempted.isin([2, 3]), p.points_attempted, np.where(three, 3, 2))
    p['points_attempted'] = np.where(p.shooting_play.fillna(False) & ~is_ft, pa, 0)
    fga = p.shooting_play.fillna(False) & ~is_ft & p.points_attempted.isin([2, 3]) & ~t.str.startswith('No Shot')
    made = p.scoring_play.fillna(False)
    x, y = p.coordinate_x.astype(float), p.coordinate_y.astype(float)
    ok = x.abs() < 60
    dist = np.where(ok, np.sqrt((x.abs() - 41.75) ** 2 + y ** 2), np.nan)
    tf = p.text.fillna('').str.extract(FOOT)[0].astype(float)
    dist = np.where(np.isnan(dist), tf, dist)
    rimtype = t.apply(lambda s: any(w in s for w in RIMWORDS))
    dist = np.where(np.isnan(dist), np.where(rimtype, 1.0, 12.0), dist)
    zone = np.where(p.points_attempted == 3, 3, np.where(dist <= 4.0, 1, 2))
    tech = is_ft & t.str.contains('Technical')
    ft = is_ft & ~tech
    m = t.str.extract(r'(\d) of (\d)')
    ftn, ftof = m[0].astype(float), m[1].astype(float)
    and1 = ft & (ftof == 1)
    trip_start = ft & (ftn == 1) & (ftof >= 2)
    ft_last = ft & (ftn == ftof)
    tov = (t.str.contains('Turnover') & (t != 'No Turnover')) | (t == 'Traveling')
    oreb, dreb = t == 'Offensive Rebound', t == 'Defensive Rebound'
    E = pd.DataFrame(dict(g=p.game_id.values, tm=p.team_id.fillna(-1).astype(int).values, per=p.period_number.values,
                          sec=p.sec.values, el=p.el.values, fga=fga.values, made=made.values, zone=zone, dist=dist,
                          pts=p.score_value.fillna(0).astype(int).values, ft=ft.values, tech=tech.values, and1=and1.values,
                          trip=trip_start.values, trip3=(trip_start & (ftof == 3)).values, ftlast=ft_last.values,
                          tov=tov.values, oreb=oreb.values, dreb=dreb.values, h=p.home_team_id.values, a=p.away_team_id.values,
                          stype=p.season_type.values, date=p.game_date.values))

    poss_rows, tg = [], {}
    for gid, G in E.groupby('g', sort=False):
        h, a = int(G.h.iloc[0]), int(G.a.iloc[0])
        def cnt(team):
            k = (gid, team)
            if k not in tg:
                tg[k] = dict(game_id=gid, team=team, opp=a if team == h else h, home=int(team == h), poss=0, opps=0, tov=0,
                             fga1=0, fga2=0, fga3=0, fgm1=0, fgm2=0, fgm3=0, and1_1=0, and1_2=0, and1_3=0, trips=0, trips3=0,
                             fta=0, ftm=0, tech_pts=0, oreb_fg=0, miss_fg=0, oreb_ft=0, miss_ftlast=0, pts=0, sec_off=0.0)
            return tg[k]
        cnt(h); cnt(a)
        score = {h: 0, a: 0}
        off, pstart, pper, cur = None, 0.0, 1, None
        pending_reb = None          # (team, 'fg'|'ft') waiting for a rebound
        last_made = None            # (team, zone, el) for and-one attribution
        rows = G.itertuples(index=False)

        def close(end_el):
            nonlocal cur
            if cur is not None:
                cur['dur'] = max(0.0, end_el - cur['start'])
                cnt(cur['off'])['sec_off'] += cur['dur']
                poss_rows.append(cur); cur = None

        def open_(team, start_el, per):
            nonlocal cur, off
            close(start_el)
            off = team
            plen = per_len(per); pend = per_start(per) + plen
            cur = dict(game_id=gid, off=team, per=per, start=start_el, sec_left=pend - start_el,
                       margin=score[team] - score[h if team == a else a], pts=0, fga=0, fga3=0, trips=0, tov=0, oreb=0,
                       first_trip=-1, end='')
            cnt(team)['poss'] += 1

        last_el = 0.0
        for r in rows:
            if r.per != pper:                               # period change: close at period end
                close(per_start(pper) + per_len(pper)); off = None; pper = r.per; pending_reb = None
            tm = r.tm
            if tm not in (h, a):
                continue
            if r.tech:
                c = cnt(tm); c['tech_pts'] += r.pts; score[tm] += r.pts; continue
            act = r.fga or r.ft or r.tov or r.oreb or r.dreb
            if not act:
                continue
            if r.oreb or r.dreb:
                if pending_reb is not None:
                    shooter, kind = pending_reb
                    if tm == shooter:
                        c = cnt(shooter); c['oreb_fg' if kind == 'fg' else 'oreb_ft'] += 1
                        if cur is not None and cur['off'] == shooter: cur['oreb'] += 1
                    pending_reb = None
                else:
                    # dead-ball team rebound (e.g. after a missed first free throw): no possession change
                    continue
                newoff = tm
            else:
                newoff = tm
            if newoff != off:
                open_(newoff, per_start(r.per) if off is None else (r.el if r.dreb else last_el), r.per)
            c = cnt(tm)
            if r.fga:
                z = int(r.zone); c[f'fga{z}'] += 1; c['opps'] += 1; cur['fga'] += 1; cur['fga3'] += int(z == 3)
                if r.made:
                    c[f'fgm{z}'] += 1; c['pts'] += r.pts; score[tm] += r.pts; cur['pts'] += r.pts; last_made = (tm, z, r.el); cur['end'] = 'make'
                else:
                    c['miss_fg'] += 1; pending_reb = (tm, 'fg'); cur['end'] = 'miss'
            elif r.ft:
                c['fta'] += 1
                if r.trip:
                    c['trips'] += 1; c['trips3'] += int(r.trip3); c['opps'] += 1; cur['trips'] += 1
                    if cur['first_trip'] < 0 and cur['fga'] == 0: cur['first_trip'] = r.el - cur['start']
                if r.and1 and last_made is not None and last_made[0] == tm and abs(last_made[2] - r.el) < 1e-6:
                    c[f'and1_{last_made[1]}'] += 1
                if r.made:
                    c['ftm'] += 1; c['pts'] += r.pts; score[tm] += r.pts; cur['pts'] += r.pts
                    if r.ftlast: cur['end'] = 'make'
                elif r.ftlast:
                    c['miss_ftlast'] += 1; pending_reb = (tm, 'ft'); cur['end'] = 'miss'
            elif r.tov:
                c['tov'] += 1; c['opps'] += 1; cur['tov'] = 1; cur['end'] = 'tov'
            last_el = r.el
        close(per_start(pper) + per_len(pper))
    TG = pd.DataFrame(tg.values())
    P = pd.DataFrame(poss_rows)
    for d in (TG, P):
        for c in ('team', 'opp', 'off'):
            if c in d: d[c] = d[c].map(TID)

    # ---------- games ----------
    s = pd.read_parquet(f'data/raw/nba_schedule_{Y}.parquet')
    s = s[s.home_id.isin(TID) & s.away_id.isin(TID) & s.season_type.isin([2, 3, 5])]
    def lines(v):
        try: return [float(q) for q in re.findall(r"'value': ([0-9.]+)", str(v))]
        except Exception: return []
    gm = pd.DataFrame(dict(game_id=s.game_id.astype(int), season=Y, stype=s.season_type, date=pd.to_datetime(s.game_date),
                           start=pd.to_datetime(s.start_date, utc=True), home=s.home_id.map(TID), away=s.away_id.map(TID),
                           neutral=s.neutral_site.fillna(False).astype(int), city=s.venue_address_city, state=s.venue_address_state,
                           done=s.status_type_completed.fillna(False).astype(bool),
                           home_pts=pd.to_numeric(s.home_score, errors='coerce'), away_pts=pd.to_numeric(s.away_score, errors='coerce'),
                           hq=s.home_linescores.map(lines), aq=s.away_linescores.map(lines)))
    gm = gm.drop_duplicates('game_id')
    gm['n_per'] = gm.hq.map(len)
    for i in range(4):
        gm[f'h{i + 1}'] = gm.hq.map(lambda v: v[i] if len(v) > i else np.nan)
        gm[f'a{i + 1}'] = gm.aq.map(lambda v: v[i] if len(v) > i else np.nan)
    gm['ot'] = (gm.n_per - 4).clip(lower=0)
    gm = gm.drop(columns=['hq', 'aq'])
    gm.loc[~gm.done, ['home_pts', 'away_pts']] = np.nan
    gm['has_pbp'] = gm.game_id.isin(TG.game_id)

    # ---------- player box ----------
    b = pd.read_parquet(f'data/raw/player_box_{Y}.parquet')
    b = b[b.team_id.isin(TID) & b.season_type.isin([2, 3, 5])]
    pl = pd.DataFrame(dict(game_id=b.game_id.astype(int), date=pd.to_datetime(b.game_date), team=b.team_id.map(TID),
                           opp=b.opponent_team_id.map(TID), home=(b.home_away == 'home').astype(int),
                           pid=b.athlete_id.astype('Int64'), name=b.athlete_display_name, pos=b.athlete_position_abbreviation,
                           starter=b.starter.fillna(False).astype(int), dnp=b.did_not_play.fillna(False).astype(int),
                           reason=b.reason.fillna(''), active=b.active.fillna(False).astype(int),
                           min=b.minutes.fillna(0).astype(float), pts=b.points.fillna(0), reb=b.rebounds.fillna(0),
                           oreb=b.offensive_rebounds.fillna(0), dreb=b.defensive_rebounds.fillna(0), ast=b.assists.fillna(0),
                           fg3m=b.three_point_field_goals_made.fillna(0), fg3a=b.three_point_field_goals_attempted.fillna(0),
                           fgm=b.field_goals_made.fillna(0), fga=b.field_goals_attempted.fillna(0),
                           ftm=b.free_throws_made.fillna(0), fta=b.free_throws_attempted.fillna(0), tov=b.turnovers.fillna(0),
                           stl=b.steals.fillna(0), blk=b.blocks.fillna(0), pf=b.fouls.fillna(0),
                           pm=pd.to_numeric(b.plus_minus.astype(str).str.replace('+', '', regex=False), errors='coerce').fillna(0)))
    pl = pl[pl.pid.notna()].drop_duplicates(['game_id', 'pid'])

    # ---------- checks against the team box scores ----------
    tb = pd.read_parquet(f'data/raw/team_box_{Y}.parquet')
    tb = tb[tb.team_id.isin(TID)]
    tb = pd.DataFrame(dict(game_id=tb.game_id.astype(int), team=tb.team_id.map(TID), b_fga=tb.field_goals_attempted,
                           b_3pa=tb.three_point_field_goals_attempted, b_fta=tb.free_throws_attempted, b_tov=tb.total_turnovers,
                           b_oreb=tb.offensive_rebounds, b_pts=tb.team_score))
    c = TG.merge(tb, on=['game_id', 'team'])
    c['fga'] = c.fga1 + c.fga2 + c.fga3
    msg = []
    for k, bk in (('fga', 'b_fga'), ('fga3', 'b_3pa'), ('fta', 'b_fta'), ('tov', 'b_tov'), ('oreb_fg', 'b_oreb')):
        v = c[k] if k != 'fta' else c['fta']
        msg.append(f'{k} {v.sum() / c[bk].sum():.3f}')
    allpts = c.pts + c.tech_pts
    msg.append(f'points match {np.mean(allpts == c.b_pts):.3f}')
    for d, n in ((gm, 'games'), (TG, 'teamgames'), (P, 'poss'), (pl, 'players')):
        d.to_parquet(f'data/derived/{n}_{Y}.parquet', index=False)
    print(f'PREP {Y}: {gm.done.sum()} finished games ({TG.game_id.nunique()} with play-by-play), {len(P)} possessions; '
          f'pbp/box ratios: ' + ', '.join(msg))


if __name__ == '__main__':
    for y in sys.argv[1:]: prep(int(y))
