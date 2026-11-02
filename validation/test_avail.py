"""Does the player-availability adjustment improve held-out predictions? 2024-25 (tuning season), closed form.
Usage: python3 validation/test_avail.py <season> <alpha,alpha,...> [half_life_p lam_p]"""
import sys, os, json
sys.path.insert(0, 'validation'); sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import rolling as RO, ratings as RT, players as PS
from avail import Avail
Y = int(sys.argv[1]); alphas = [float(a) for a in sys.argv[2].split(',')]
hp = float(sys.argv[3]) if len(sys.argv) > 3 else 400; lp = float(sys.argv[4]) if len(sys.argv) > 4 else 20
tp = json.load(open('validation/tuned_params.json'))['ratings'] if os.path.exists('validation/tuned_params.json') else \
    dict(half_life=90, K_other=3, K_make=5, K_ft=20, K_pace=3)
TG = RT.load_tg(range(Y - 3, Y + 1)); PL = PS.load(range(Y - 3, Y + 1))
E = RO.E
for a in alphas:
    av = Avail(PL, dict(half_life_p=hp, lam_p=lp), tp['half_life'], a, E['ppp_sens'], E['base_ppp'])
    df = RO.predict_season(Y, tp, TG, adj=av)
    s = RO.score_closed(df)
    print(json.dumps({'season': Y, 'alpha': a, 'half_life_p': hp, 'lam_p': lp, **{k: round(v, 4) for k, v in s.items()}}), flush=True)
