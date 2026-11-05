"""Score the 2025-26 player-prop test at the dispersions chosen on 2024-25 (validation/props_2025_base.parquet), for the
model and for each baseline. A market passes when the model beats both baselines on its book-style lines and on the
fixed lines together. Writes validation/props_test_2026.json."""
import json, sys
SUF = '_v2' if len(sys.argv) > 1 and sys.argv[1] == 'v2' else ''      # v2 = retuned minutes, rates and dispersion
import numpy as np, pandas as pd
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))))
GRID = [3, 5, 8, 12, 20, 40]
tune_df = pd.read_parquet(f'validation/props_2025_base{SUF}.parquet')
tune = {}
for k, x in tune_df.groupby('stat'):            # dispersion per stat and per method, chosen on all 2024-25 lines
    tune[k] = {n: min(GRID, key=lambda f: ll(x.y, x[f'{c}{f}'])) for c, n in (('m', 'model'), ('s', 'season_avg'), ('r', 'recent_10'))}
    xx = x; tune[k + '_2025'] = {n: round(ll(xx.y, xx[f"{c}{tune[k][n]}"]), 4) for c, n in (('m', 'model'), ('s', 'season_avg'), ('r', 'recent_10'))}
print('2024-25 (tuning):', json.dumps(tune))
df = pd.read_parquet(f'validation/props_2026_base{SUF}.parquet')
out = {'tuning_2025': tune}
for k in ('pts', 'reb', 'ast', 'fg3m', 'pra'):
    ph = tune[k]
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
json.dump(out, open(f'validation/props_test_2026{SUF}.json', 'w'), indent=1)
