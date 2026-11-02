"""
Attach a Pinnacle snapshot (parseforge/pinnacle-odds-scraper, data/apify/odds/latest.json) to the slate.

Pinnacle's NBA game lines include overtime. The spread and total are what the anchoring fits; the moneyline is kept
as an independent check. Only the main game lines are used.

Usage (repo root): python3 pipeline/odds.py <stamp>
Writes build/odds.json {game_id: {...}} oriented to the slate's home team, and data/odds/<stamp>.csv
"""
import json, os, re, sys, unicodedata
import pandas as pd

stamp = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H%M')
SRC = 'data/apify/odds/latest.json'
slate = json.load(open('build/slate.json'))
S = json.load(open('pipeline/settings.json'))


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9 ]', '', s).strip()


abbr = {norm(n): ab for ab, ns in S['team_names'].items() for n in ns}


def fair(sel, designation):
    for s in sel or []:
        if s.get('designation') == designation:
            return s.get('fairProbability'), s.get('american'), s.get('points')
    return None, None, None


rows = json.load(open(SRC)) if os.path.exists(SRC) and os.path.getsize(SRC) > 2 else []
idx = {(g['away'], g['home']): g for g in slate['games']}
out, log = {}, []
for r in rows:
    if 'NBA' not in (r.get('leagueName') or '') or (r.get('units') or 'Regular') != 'Regular':
        continue
    if (r.get('leagueName') or '').strip() != 'NBA':
        continue
    h, a = abbr.get(norm(r.get('homeTeam'))), abbr.get(norm(r.get('awayTeam')))
    flip = False
    g = idx.get((a, h))
    if g is None and (h, a) in idx: g, flip = idx[(h, a)], True
    if g is None: continue
    o = {'snap': r.get('scrapedAt'), 'matchup': r.get('matchupId')}
    ml = r.get('moneyline') or {}
    fh, amh, _ = fair(ml.get('selections'), 'home'); fa, ama, _ = fair(ml.get('selections'), 'away')
    if fh is not None and fa is not None:
        o.update({'home_fair': fa if flip else fh, 'ml_home': ama if flip else amh, 'ml_away': amh if flip else ama, 'ml_limit': ml.get('maxStake')})
    t = r.get('total') or {}
    fo, amo, _ = fair(t.get('selections'), 'over'); fu, amu, _ = fair(t.get('selections'), 'under')
    if fo is not None and t.get('points') is not None:
        o.update({'tot_line': float(t['points']), 'over_fair': fo, 'over': amo, 'under': amu, 'tot_limit': t.get('maxStake')})
    sp = r.get('spread') or {}
    sh, amsh, ph = fair(sp.get('selections'), 'home'); sa, amsa, pa = fair(sp.get('selections'), 'away')
    if sh is not None and ph is not None:
        o.update({'sp_home_line': float(pa if flip else ph), 'sp_home_fair': sa if flip else sh,
                  'sp_home': amsa if flip else amsh, 'sp_away': amsh if flip else amsa, 'sp_limit': sp.get('maxStake')})
    if not any(k in o for k in ('home_fair', 'tot_line', 'sp_home_line')): continue
    out[g['id']] = o
    log.append({'snap_utc': o['snap'], 'game_id': g['id'], 'start': g['start'], 'home': g['home'], 'away': g['away'],
                **{k: o.get(k) for k in ('sp_home_line', 'sp_home_fair', 'sp_home', 'sp_away', 'sp_limit', 'tot_line', 'over_fair',
                                         'over', 'under', 'tot_limit', 'home_fair', 'ml_home', 'ml_away', 'ml_limit')}})
json.dump(out, open('build/odds.json', 'w'), separators=(',', ':'))
if log:
    os.makedirs('data/odds', exist_ok=True)
    pd.DataFrame(log).to_csv(f'data/odds/{stamp}.csv', index=False)
print(f'ODDS {len(out)} of {len(slate["games"])} slate games priced by Pinnacle '
      f'({sum("sp_home_line" in v for v in out.values())} spread, {sum("tot_line" in v for v in out.values())} total); {len(rows)} rows in the pull')
