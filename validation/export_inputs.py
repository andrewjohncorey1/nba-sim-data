"""Rolling held-out engine inputs for a season (every game rated on earlier games only, availability from the players
who played, minutes projected from earlier games). Usage: python3 validation/export_inputs.py <season> [<variant>]
variants: base (tuned settings), noavail (alpha 0), rest (adds the back-to-back effect), restm (back-to-back on the
margin only), restall (all rest/travel terms)
Writes validation/inputs_<season>_<variant>.json and validation/games_<season>.parquet (actual results)."""
import sys, os, json
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import rolling as RO, ratings as RT, players as PS, schedule_feats as SF
from avail import Avail
Y = int(sys.argv[1]); VARS = (sys.argv[2] if len(sys.argv) > 2 else 'base').split(',')
S = json.load(open('pipeline/settings.json')); prm0 = dict(S['model']['ratings'])
TG = RT.load_tg(range(Y - 3, Y + 1)); PL = PS.load(range(Y - 3, Y + 1))
F = SF.features([Y]).set_index('game_id')
RE = json.load(open('validation/rest_effects.json'))
for var in VARS:
    prm = dict(prm0)
    if var == 'noavail': prm['alpha'] = 0.0
    av = Avail(PL, S['model']['players'], prm['half_life'], prm['alpha'], RO.E['ppp_sens'], RO.E['base_ppp'])
    def adj(r, R):
        sh, sa = av(r, R)
        if var == 'restm' and r.game_id in F.index:
            f = F.loc[r.game_id]; net = (RE['own_b2b']['est'] - RE['dfn_b2b']['est']) / 2      # margin-only: no effect on the total
            ph = net * f['b2b_h'] - net * f['b2b_a']; pa = -ph
            k = lambda p: np.log(max(0.5, 1 + p / 100 / RO.E['base_ppp'])) / RO.E['ppp_sens']
            sh += k(ph); sa += k(pa)
        if var in ('rest', 'restall') and r.game_id in F.index:
            f = F.loc[r.game_id]; ph = pa = 0.0
            terms = [('b2b', 'b2b_{}', lambda v: v)]
            if var == 'restall':
                terms += [('rest3', 'rest_{}', lambda v: float(v >= 3)), ('trav', 'trav_{}', lambda v: v / 1000),
                          ('tzE', 'tz_{}', lambda v: max(v, 0)), ('tzW', 'tz_{}', lambda v: max(-v, 0))]
            for nm, col, fn in terms:
                ph += RE['own_' + nm]['est'] * fn(f[col.format('h')]) + RE['dfn_' + nm]['est'] * fn(f[col.format('a')])
                pa += RE['own_' + nm]['est'] * fn(f[col.format('a')]) + RE['dfn_' + nm]['est'] * fn(f[col.format('h')])
            k = lambda p: np.log(max(0.5, 1 + p / 100 / RO.E['base_ppp'])) / RO.E['ppp_sens']
            sh += k(ph); sa += k(pa)
        return sh, sa
    df = RO.predict_season(Y, prm, TG, adj=adj if var != 'noavail' else None)
    X = [dict(game_id=int(r.game_id), h={k: float(v) for k, v in r.ch.items()}, a={k: float(v) for k, v in r.ca.items()}, pace=float(r.pace)) for r in df.itertuples()]
    json.dump(X, open(f'validation/inputs_{Y}_{var}.json', 'w'))
    df.drop(columns=['ch', 'ca']).to_parquet(f'validation/games_{Y}_{var}.parquet', index=False)
    print(var, Y, json.dumps({k: round(v, 4) for k, v in RO.score_closed(df).items()}))
