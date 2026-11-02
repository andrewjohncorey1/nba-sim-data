"""
Build the slate of upcoming games (today and tomorrow, Eastern time) with player availability and rest.

Usage (repo root): python3 pipeline/slate.py <season> <stamp>
Reads  data/raw/nba_schedule_<season>.parquet, build/ratings.json, build/injuries.json (RotoWire, may be empty)
Writes build/slate.json

Availability: players RotoWire lists as Out, OFS or Doubt are out; Ques / Prob / GTD are shown and treated as
playing. When the lineups page does not cover a team, nobody is marked out ('no report'), except players who have
missed the team's last 5 games and every game in the last 10 days (long-term absences).
"""
import sys, os, json, re, unicodedata
import numpy as np, pandas as pd
import schedule_feats as SF

Y = int(sys.argv[1]); stamp = sys.argv[2]
D = json.load(open('build/ratings.json'))
S = json.load(open('pipeline/settings.json'))
INJ = json.load(open('build/injuries.json')) if os.path.exists('build/injuries.json') else {}
TID = {int(k): v for k, v in S['team_ids'].items()}
OUT = ('Out', 'OFS', 'Doubt')


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


rost = {t['abbr']: t for t in D['rosters']}
pool_by_name = {}
for pid, p in D['pool'].items():
    pool_by_name.setdefault(norm(p['name']), []).append(int(pid))


def find(name, team):
    n = norm(name)
    for p in rost[team]['roster']:
        if norm(p['name']) == n: return p['pid'], True
    c = pool_by_name.get(n, [])
    if c: return c[0], False
    # last name + first initial on the same roster
    parts = str(name).split()
    if len(parts) >= 2:
        ln = norm(parts[-1]); fi = norm(parts[0])[:1]
        for p in rost[team]['roster']:
            q = str(p['name']).split()
            if len(q) >= 2 and norm(q[-1]) == ln and norm(q[0])[:1] == fi: return p['pid'], True
    return None, False


s = pd.read_parquet(f'data/raw/nba_schedule_{Y}.parquet')
s = s[s.home_id.isin(TID) & s.away_id.isin(TID)]
s['start'] = pd.to_datetime(s.start_date, utc=True)
now = pd.Timestamp.now(tz='UTC')
et = now.tz_convert('America/New_York')
lo = et.normalize().tz_convert('UTC')
hi = (et.normalize() + pd.Timedelta(days=2)).tz_convert('UTC') + pd.Timedelta(hours=6)
s = s[(s.start >= lo) & (s.start < hi) & (s.status_type_state == 'pre')].sort_values('start')
F = SF.features([Y]).set_index('game_id')
games, rep = [], {'covered': 0, 'out': 0, 'unmatched': []}
for r in s.itertuples():
    h, a = TID[r.home_id], TID[r.away_id]
    gid = int(r.game_id)
    g = dict(id=str(gid), home=h, away=a, start=r.start.strftime('%Y-%m-%dT%H:%M:%SZ'), neutral=int(bool(r.neutral_site)),
             playoff=int(r.season_type == 3), venue=r.venue_full_name)
    if gid in F.index:
        f = F.loc[gid]
        g['rest'] = {'h': int(f.rest_h), 'a': int(f.rest_a), 'b2b_h': int(f.b2b_h), 'b2b_a': int(f.b2b_a)}
    for side, t in (('home', h), ('away', a)):
        info = {'source': 'no report', 'out': [], 'ques': [], 'extra': [], 'lineup': None, 'starters': []}
        ij = INJ.get(t)
        if ij:
            info['source'] = 'RotoWire ' + ij['lineup'].lower() + ' lineup'; info['lineup'] = ij['lineup']; rep['covered'] += 1
            for nm, st in ij['status'].items():
                pid, on = find(nm, t)
                if pid is None: rep['unmatched'].append(f'{t}: {nm}'); continue
                if st in OUT: info['out'].append(pid)
                else: info['ques'].append([pid, st])
            for nm in ij['starters']:
                pid, on = find(nm, t)
                if pid is None: rep['unmatched'].append(f'{t}: {nm}'); continue
                info['starters'].append(pid)
                if not on and pid not in info['out']: info['extra'].append(pid)     # new arrival not yet on the roster
        else:
            for p in rost[t]['roster']:
                if p.get('last') and (pd.Timestamp(et.date()) - pd.Timestamp(p['last'])).days > 10 and p['pm'] >= 15:
                    info['out'].append(p['pid'])
        rep['out'] += len(info['out'])
        g[side + '_avail'] = info
    games.append(g)
json.dump(dict(generated=now.strftime('%Y-%m-%dT%H:%M:%SZ'), games=games), open('build/slate.json', 'w'), separators=(',', ':'))
print(f'SLATE {len(games)} games (today and tomorrow ET); lineups page covered {rep["covered"]} of {2 * len(games)} team-games, '
      f'{rep["out"]} players out' + (f'; NOTE names not matched: {", ".join(rep["unmatched"][:10])}' if rep['unmatched'] else ''))
