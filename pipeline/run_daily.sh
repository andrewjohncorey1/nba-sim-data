#!/usr/bin/env bash
# Usage (repo root): pipeline/run_daily.sh <lineups_dataset_ids,comma-separated|-> [<pinnacle_dataset_id>|-]
# Pulls the current season from the GitHub releases, rebuilds derived tables and ratings, reads the RotoWire lineups
# page(s) and the Pinnacle snapshot from Apify datasets, logs pre-game predictions, scores finished games, runs sanity
# checks and builds build/nba-sim.html. Publishing and git push are done by the caller.
set -euo pipefail
LU="${1:--}"; ODS="${2:-}"; [ "$ODS" = "-" ] && ODS=""
cd "$(dirname "$0")/.."
SEASON=$(python3 -c "import json;print(json.load(open('pipeline/settings.json'))['season'])")
STAMP=$(date -u +%Y-%m-%dT%H%M)
API="https://api.apify.com/v2/datasets"
count() { curl -sSf "${API}/$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["itemCount"])'; }
mkdir -p data/injuries data/apify/odds build
export PYTHONPATH=pipeline

pipeline/fetch_data.sh "$SEASON"
python3 pipeline/prep.py "$SEASON"

: > data/injuries/latest.md
if [ "$LU" != "-" ]; then
  for ds in ${LU//,/ }; do
    if [ "$(count "$ds")" -gt 0 ]; then
      curl -sSf "${API}/${ds}/items?format=json&clean=true&fields=markdown" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[0].get("markdown","") if d else "")' >> data/injuries/latest.md
    fi
  done
fi
python3 pipeline/injuries.py "$STAMP"
python3 pipeline/export.py "$SEASON"
python3 pipeline/slate.py "$SEASON" "$STAMP"

if [ -n "$ODS" ] && [ "$(count "$ODS")" -gt 0 ]; then
  curl -sSf -o data/apify/odds/latest.json "${API}/${ODS}/items?format=json&clean=true"
else
  echo '[]' > data/apify/odds/latest.json
fi
python3 pipeline/odds.py "$STAMP"
node pipeline/predict.js "$STAMP" "$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
python3 pipeline/score.py "$SEASON"
python3 pipeline/sanity.py || true
python3 pipeline/build_app.py
