"""Inject ratings, slate (with Pinnacle prices and anchoring) and the scorecard into the template -> build/nba-sim.html"""
import json, os
tpl = open('pipeline/app_template.html').read()
data = open('build/ratings.json').read(); json.loads(data)
slate = json.load(open('build/slate.json')) if os.path.exists('build/slate.json') else {'games': [], 'generated': ''}
odds = json.load(open('build/odds.json')) if os.path.exists('build/odds.json') else {}
market = json.load(open('build/market.json')) if os.path.exists('build/market.json') else {}
for g in slate['games']:
    if g['id'] in odds: g['odds'] = odds[g['id']]
    m = market.get(g['id'])
    if m:
        g['mkt'] = m
        g['fair'] = {'m': m['fm'], 'a': m.get('fa', m['fm'])}
score = open('build/scorecard.json').read() if os.path.exists('build/scorecard.json') else '{"n_scored":0,"n_pending":0}'
json.loads(score)
engine = open('pipeline/engine.js').read().split("if (typeof module !== 'undefined')")[0]
html = (tpl.replace('__ENGINE__', engine).replace('__DATA__', data)
        .replace('__SLATE__', json.dumps(slate, separators=(',', ':'))).replace('__SCORE__', score))
assert all(k not in html for k in ('__DATA__', '__SLATE__', '__ENGINE__', '__SCORE__'))
open('build/nba-sim.html', 'w').write(html)
print(f'wrote build/nba-sim.html ({len(html) // 1024} KB)')
