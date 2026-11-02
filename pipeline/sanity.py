"""
Sanity checks on build/ratings.json before publishing; compares with build/published_ratings.json.
Exit 0 = safe to publish, 1 = something looks wrong (PROBLEM lines printed).
"""
import json, math, os, sys
sys.path.insert(0, 'pipeline')
new = json.load(open('build/ratings.json'))
old = json.load(open('build/published_ratings.json')) if os.path.exists('build/published_ratings.json') else None
problems, notes = [], []
E = new['engine']
ex = lambda v: 1 / (1 + math.exp(-v))


def ppp(D, o, d):
    i, j = D['teams'].index(o), D['teams'].index(d)
    c = {k: ex(D['comp'][k]['mu'] + D['comp'][k]['O'][i] - D['comp'][k]['D'][j]) for k in D['comp']}
    sh = [(1 - c['s3']) * c['s1'], (1 - c['s3']) * (1 - c['s1']), c['s3']]; m = [c['m1'], c['m2'], c['m3']]
    fp = sum(sh[z] * m[z] * ((2 if z < 2 else 3) + E['and1'][z] * c['ft']) for z in range(3))
    fc = sum(sh[z] * (1 - m[z]) for z in range(3)) * c['oreb']
    e = (1 - c['tov']) * (c['trip'] * (2 + E['trip3']) * c['ft'] + (1 - c['trip']) * fp)
    cont = (1 - c['tov']) * (c['trip'] * (1 - c['ft']) * E['oreb_ft'] + (1 - c['trip']) * fc)
    return e / (1 - cont)


def net(D, t):
    o = [ppp(D, t, x) for x in D['teams'] if x != t]; d = [ppp(D, x, t) for x in D['teams'] if x != t]
    return 100 * (sum(o) - sum(d)) / len(o)


if len(new['teams']) != 30: problems.append(f'{len(new["teams"])} teams (expected 30)')
lg = 100 * sum(ppp(new, t, 'BOS') for t in new['teams'] if t != 'BOS') / 29
nets = {t: net(new, t) for t in new['teams']}
for t, v in nets.items():
    if abs(v) > 20: problems.append(f'{t}: net rating {v:+.1f} per 100 outside +/-20')
pace = math.exp(new['pace']['mu']) 
if not 90 <= pace <= 110: problems.append(f'league pace {pace:.1f} outside [90, 110]')
if not 100 <= lg <= 130: problems.append(f'league points per 100 {lg:.1f} outside [100, 130]')
nr = sum(len(t['roster']) for t in new['rosters'])
if nr < 300: problems.append(f'only {nr} rostered players')
if old:
    if new['asOf'] < old['asOf']: problems.append(f'data date went backwards {old["asOf"]} -> {new["asOf"]}')
    on = {t: net(old, t) for t in old['teams']}
    mv = sorted(((nets[t] - on[t], t) for t in nets if t in on), key=lambda x: -abs(x[0]))
    notes.append('biggest rating moves (net points per 100): ' + ', '.join(f'{a} {d:+.1f}' for d, a in mv[:3]))
    if mv and abs(mv[0][0]) > 6: problems.append(f'{mv[0][1]} net rating moved {mv[0][0]:+.1f} in one update')
    lo = 100 * sum(ppp(old, t, 'BOS') for t in old['teams'] if t != 'BOS') / 29
    if abs(lg - lo) > 3: problems.append(f'league scoring level moved {lg - lo:+.1f} per 100 in one update')
for n in notes: print('NOTE', n)
for p in problems[:20]: print('PROBLEM', p)
print('RESULT', 'FAIL' if problems else 'PASS')
sys.exit(1 if problems else 0)
