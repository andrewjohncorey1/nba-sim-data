"""Score the 2025-26 player-prop test at the dispersions chosen on 2024-25 (validation/props_tune_2025.log), for the
model and for each baseline. A market passes when the model beats both baselines on its book-style lines and on the
fixed lines together. Writes validation/props_test_2026.json."""
import json
import numpy as np, pandas as pd
tune = json.load(open('validation/props_tune_2025.log'))
df = pd.read_parquet('validation/props_2026_base.parquet')
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))))
out = {}
for k in ('pts', 'reb', 'ast', 'fg3m', 'pra'):
    ph = tune['results'][f'{k} book']['phi']
    x = df[df.stat == k]
    row = {'phi': ph}
    for kind, xx in [('all', x), ('book', x[x.kind == 'book']), ('fixed', x[x.kind == 'fixed'])]:
        row[kind] = {'n': len(xx), 'model': round(ll(xx.y, xx[f"m{ph['model']}"]), 4), 'season_avg': round(ll(xx.y, xx[f"s{ph['season_avg']}"]), 4),
                     'recent_10': round(ll(xx.y, xx[f"r{ph['recent_10']}"]), 4), 'pred_over': round(float(xx[f"m{ph['model']}"].mean()), 3),
                     'act_over': round(float(xx.y.mean()), 3)}
    a = row['all']; row['pass'] = bool(a['model'] < a['season_avg'] and a['model'] < a['recent_10'])
    row['mae'] = {'model': round(float(np.mean(np.abs(x[x.kind == 'book'].actual - x[x.kind == 'book']['mean']))), 3),
                  'season_avg': round(float(np.mean(np.abs(x[x.kind == 'book'].actual - x[x.kind == 'book'].base))), 3)}
    out[k] = row
    print(k, json.dumps(row))
json.dump(out, open('validation/props_test_2026.json', 'w'), indent=1)
