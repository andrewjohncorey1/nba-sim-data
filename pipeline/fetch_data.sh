#!/usr/bin/env bash
# Download play-by-play, box scores and schedules from the sportsdataverse-data GitHub releases (hoopR; free; refreshed
# daily, usually between 5 and 7:30 AM ET).
# Usage (repo root): pipeline/fetch_data.sh <current_season_end_year> [also_previous=0]
# Only the current season is re-downloaded on a normal run; earlier seasons' derived tables are committed in data/derived.
set -euo pipefail
Y="$1"; PREV="${2:-0}"
B=https://github.com/sportsdataverse/sportsdataverse-data/releases/download
mkdir -p data/raw
get(){ curl -sSfL --retry 2 -m 600 -o "data/raw/$2.parquet.tmp" "$B/$1/$2.parquet" && mv "data/raw/$2.parquet.tmp" "data/raw/$2.parquet"; }
seasons="$Y"; [ "$PREV" = "1" ] && seasons="$((Y-5)) $((Y-4)) $((Y-3)) $((Y-2)) $((Y-1)) $Y"
for y in $seasons; do
  get espn_nba_pbp "play_by_play_$y"
  get espn_nba_schedules "nba_schedule_$y"
  get espn_nba_team_boxscores "team_box_$y"
  get espn_nba_player_boxscores "player_box_$y"
done
python3 - "$Y" <<'PY'
import sys, pandas as pd
y = sys.argv[1]
p = pd.read_parquet(f'data/raw/play_by_play_{y}.parquet', columns=['game_id', 'game_date'])
s = pd.read_parquet(f'data/raw/nba_schedule_{y}.parquet', columns=['game_id', 'status_type_completed'])
print(f'FETCH season {y}: play-by-play through {p.game_date.max()} ({p.game_id.nunique()} games); schedule {len(s)} games, {int(s.status_type_completed.sum())} final')
PY
