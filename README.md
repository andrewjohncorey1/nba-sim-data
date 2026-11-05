# NBA Game Simulator: data and pipeline

Data, rating model and build pipeline behind the NBA Game Simulator app (link in `DAILY.md`).
Same layout and conventions as `nhl-sim-data` and `tennis-sim-data`.

## How the model works
1. **Possessions** (`pipeline/prep.py`). Every ESPN play-by-play event (sportsdataverse hoopR releases, free) is turned
   into per-game counts for each offense and a table of possessions. Shots are split by zone: rim (within 4 feet),
   mid-range, three. Checks against the box scores in every season 2021-22 to 2026-27: field-goal attempts, threes and
   turnovers match within 1.2%, points match in 99.6-100% of games. Offensive rebounds include team rebounds (they
   keep the possession alive, so the engine needs them), about 1.3x the box-score count.
2. **Team ratings** (`pipeline/ratings.py`). Nine components, each `logit P = mu + home + Off[offense] - Def[defense]`,
   ridge-penalized and weighted by game age: turnover per opportunity, free-throw trip instead of a shot, three-point
   share, rim share of twos, make rate at the rim / mid-range / three, offensive rebound on a miss, free-throw %
   (offense only). Pace: log possessions per 48 = mu + Pace[home] + Pace[away]. Half-life 90 days; ridge strength
   3 games of league-average data for shot mix, turnovers, trips and rebounds, 5 for make rates, 20 for free-throw %,
   3 for pace (all chosen on 2024-25, `validation/tune.py`). The closed-form points per possession from these
   components reproduces the league's actual rate (1.1503 vs 1.1498).
3. **Player availability** (`pipeline/players.py`). Each player gets an offensive and defensive impact (points per 100
   possessions on court): a ridge regression of each team-game's offensive rating on the minutes shares of the ten
   players involved, pulled toward a box-score prior (fit once on 2021-22..2023-24, `pipeline/player_prior.json`).
   A team's adjustment for a game = sum over players of (tonight's share of the floor - the share behind the team
   rating, same 90-day weights) x impact. Tonight's minutes: each available player's recent minutes when he plays
   (half-life 8 games) x the chance he gets in (played vs coach's-decision DNPs, last 15 games), fitted to 240 with
   regulars protected (`allocate`). The adjustment moves the offense's make rates by the equivalent amount. Daily
   source: RotoWire's lineups page (Out, OFS and Doubtful count as out; Questionable, Probable and GTD as in).
4. **Game engine** (`pipeline/engine.js`, shared by the app, the pipeline and the tests). Each simulation plays
   possession by possession on a running clock: turnover, free-throw trip (2 or 3 shots), or a shot from one of the
   three zones with its make rate and and-ones; misses can be rebounded by the offense. Possession lengths are drawn
   from measured lengths for how the previous possession ended, scaled to the matchup's pace. Measured league
   behaviour (`pipeline/engine_params.py`, 2021-22..2024-25, about 1.05 million possessions): score effects by
   margin (adjusted for team quality), the last 150 seconds of the 4th quarter and overtime (intentional fouls,
   running the clock out, hurried threes, playing for the last shot) and end-of-period heaves. Ties go to 5-minute
   overtimes. Before anything else, both offenses' make rates are shifted so the simulation's mean margin and total
   equal the ratings' own closed-form expectation (`selfAnchor`): the ratings already contain how teams play ahead and
   behind, and the score effects would otherwise count that twice. Three calibration constants were set on the
   2023-24 held-out simulation (`pipeline/calib.json`): possession-length scale 1.058, total offset 2.15 points,
   first/second-half split 0.038.
5. **Markets** from the same simulations: moneyline, spread and alternate spreads, totals, team totals, each half and
   quarter (spread, total, win), winning-margin bands, player props. Game lines include overtime; the 2nd half includes
   overtime, the 4th quarter does not.

## Pinnacle anchoring
For games Pinnacle prices, `anchorToMarket` shifts both offenses' make rates (a common amount for the total, a split
for the spread) until the simulation reproduces Pinnacle's no-vig cover probability at its spread and over probability
at its total; everything else is priced from that simulation. Pinnacle's moneyline is not used in the fit and is kept
as the independent check. The app's "Model only" toggle shows the unanchored numbers.
First check (Nov 2, 2026, all 15 games): anchored moneyline within 1.1 points of Pinnacle's no-vig moneyline on
average, 4.0 at most. Pulled once a day (afternoon run) to save Apify credit.

## Held-out results
Every game rated only on earlier games (refit daily; player impacts weekly; availability = the players who actually
played, which is public before tip-off; minutes projected from earlier games only). 2024-25 was used to choose
settings; 2025-26 was untouched until the end. Log loss, lower is better.

| 2025-26 (test) | Model | Point-differential baseline | Home-rate baseline | DraftKings close |
|---|---|---|---|---|
| Moneyline log loss | 0.5854 | 0.6011 | 0.6871 | 0.5778 (no-vig, 1,057 games; model 0.5877 on those) |
| Final margin RMSE (points) | 14.27 | 14.68 | 16.46 | 14.27 (model 14.48 on those 1,062 games) |
| Total RMSE (points) | 18.91 | 20.20 (league average) | | 18.32 (model 18.74 on those 1,061 games) |
| Spread at DraftKings' line | 0.7063 | | 0.6931 (50/50) | |
| Total at DraftKings' line | 0.7097 | | 0.6931 (50/50) | |

| Fixed lines, log loss | 2025-26 model | baseline | 2024-25 model | baseline |
|---|---|---|---|---|
| Home -7.5 / -3.5 | 0.5703 / 0.5942 | 0.6586 / 0.6924 | 0.5796 / 0.5976 | 0.6611 / 0.6923 |
| Home +3.5 / +7.5 | 0.5662 / 0.5253 | 0.6645 / 0.6080 | 0.5681 / 0.5096 | 0.6582 / 0.5861 |
| Total over 215.5 / 225.5 / 235.5 | 0.5101 / 0.6457 / 0.6277 | 0.5494 / 0.6848 / 0.6688 | 0.5549 / 0.6507 / 0.5920 | 0.6043 / 0.6944 / 0.6382 |
| Team total over 114.5 (home / away) | 0.6369 / 0.6197 | 0.6923 / 0.6958 | 0.6319 / 0.6241 | 0.6957 / 0.6840 |
| 1st-half total over 113.5 | 0.6681 | 0.6906 | 0.6714 | 0.6931 |
| 1st half won by home team | 0.6413 | 0.6926 | 0.6526 | 0.6930 |

Baselines for fixed lines: the share of games in the three earlier seasons that went over or covered.

What this says:
- The model beats every simple baseline on every market tested, in both seasons.
- It does **not** beat the market. At DraftKings' closing spread and total its probabilities do worse than a coin flip
  (0.706 and 0.710 vs 0.693): where it disagrees with the close it is wrong more often than right. Model-only prices
  are not an edge on their own; on games Pinnacle prices, use the anchored numbers.
- Game shape on 2025-26: possessions 99.5 vs 100.0 per team, points per game 228.2 vs 230.2 (2.0 low; 2.0 low in
  2024-25 too, after the offset set on 2023-24, where it had been 2.2 high), first-half share of points 50.0% vs 50.3%.
  Overtime is under-produced: 2.8% simulated vs 4.5-5.0% actual. Real NBA endings cluster at ties more than the
  engine's late-game rules reproduce. Anchoring to Pinnacle's total absorbs the overtime points; the moneyline in
  near-even games and the "tie" columns in the period table are the numbers most affected.
- Calibration (2025-26, model home win %, predicted / actual): 16 / 14, 29 / 22, 43 / 40, 57 / 61, 72 / 78, 85 / 88.
  Slightly underconfident on clear favourites.

## Model additions (each kept only if it improved held-out predictions)
Closed-form expected margin and total (fast), every game rated on earlier games; moneyline via a normal approximation.

| Addition | 2024-25 (tuning) | 2025-26 (test) | Kept |
|---|---|---|---|
| Player availability (impact x minutes) | margin RMSE 14.31 -> 14.02, moneyline 0.6106 -> 0.5959 | 14.66 -> 14.26, 0.6021 -> 0.5831 | yes |
| Back-to-back, margin only (-2.6 per 100 for the tired team) | 14.02 -> 13.97, 0.5959 -> 0.5930 | 14.26 -> 14.26, 0.5831 -> 0.5844 | no: did not hold up on 2025-26 |
| Back-to-back on offense and defense separately | margin 14.06 -> 14.00 but total RMSE 17.78 -> 17.83 | | no |
| Rest of 3+ days, travel distance, time zones | margin 14.003 -> 13.996, total worse; none significant on 2021-24 (`validation/rest_effects.json`) | | no |
| Home court | in every component; home offense +1.9 per 100 (se 0.3) on 2021-24 | | yes |
| Recency half-life 45 / 60 / 90 / 120 / 180 days | margin 14.18 / 14.12 / 14.06 / 14.05 / 14.04, total 17.71 / 17.72 / 17.78 / 17.86 / 18.00 | | 90 |
| Matching the simulation to the ratings' closed form (2023-24 calibration season) | moneyline 0.6053 -> 0.6005, total RMSE 24.8 -> 18.6 | | yes |
| First/second-half split calibration (set on 2023-24) | 1st-half share 49.5% -> 50.0% (actual 50.0-50.3%) | 1st-half total log loss 0.6808 -> 0.6681 | yes |

Tuning log: `validation/tune_2025.log` (with availability), `validation/tune_2025_noavail.log` (without);
variants: `validation/variants_2025_closedform.log`, `validation/export_2025.log`, `validation/export_2026.log`.

## Player props
Points, assists, threes made and points+rebounds+assists for every available player (`playerProps` in
`pipeline/engine.js`, tested in `validation/props.py` and `validation/props_final.py`):
- **Rates** per minute (recency-weighted, half-life 180 days, shrunk toward the position average with 300 minutes of
  prior) x **minutes** when he plays (as in the availability model).
- **From the simulation**: a player's mean = his expected stat / the team's sum of player expectations x the team
  quantity in each simulated game (points; made field goals for assists; threes made; rebounds), so when a teammate
  is out his usage is shared out, and pace and game script carry through. Priced with a negative binomial mixed over
  the simulated team distribution (Pinnacle-anchored when priced). Lines shown: the player's mean rounded down + 0.5.
- Dispersions chosen on 2024-25 (points 5, assists 12, threes 20, PRA 8).

Held-out test, 2025-26 (every player-game rated on earlier games; players who played; lines: book-style = season
average rounded down + 0.5, plus fixed lines; baselines price the same lines from the season-to-date or last-10
average with their own best dispersion). Log loss over all lines, lower is better:

| 2025-26 | Lines | Model | Season average | Last 10 games | Predicted / actual over | Kept |
|---|---|---|---|---|---|---|
| Points (book lines alone: 0.6915 vs 0.6959 / 0.6927) | 154,308 | 0.3616 | 0.3809 | 0.3773 | 23.5% / 25.5% | yes |
| Assists | 128,590 | 0.3891 | 0.4012 | 0.4025 | 28.4% / 28.8% | yes |
| Threes made | 128,590 | 0.4482 | 0.4583 | 0.4690 | 31.2% / 31.0% | yes |
| Points + rebounds + assists | 154,308 | 0.4014 | 0.4337 | 0.4241 | 31.4% / 32.1% | yes |
| Rebounds | 128,590 | 0.4254 | 0.4381 | 0.4378 | 34.3% / 29.1% | no |

Rebounds beat both baselines on log loss but their means run high (the simulated team rebounds include team
rebounds, which players don't get credited) and their average error is worse than the season average (1.97 vs 1.93),
so they stay off until that scaling is fixed and retested. Points run slightly low on overs (23.5% vs 25.5%).
On book-style lines alone the edge over the baselines is small for points (0.6915 vs 0.6927), which is what a real
book line would look like; the model has not been tested against actual sportsbook prop lines, which are not collected.

## Same-game parlays: joint-calibration test (not in the app)
`validation/sgp.py`, results in `validation/sgp_2025.log` / `sgp_2026.log` (and `.json`). Same held-out rules as the
prop test, same simulations (the engine can now keep every simulated final score: `simulateGame(..., keepRows)`, which
draws no extra random numbers). Joint model: in each simulated game the player's points are negative binomial around
his share x that game's team points. Baseline: the same two single-leg prices multiplied. "Gap" = chance of leg A when
leg B hit minus when it missed, predicted vs actual (the correlation itself, mostly free of single-leg bias).

| 2025-26 (2024-25 in brackets) | Log loss gain vs legs multiplied (95% interval) | Gap predicted | Gap actual |
|---|---|---|---|
| A: player points over + own team wins | +0.0013 (+0.0009 to +0.0018) [+0.0015] | +2.9 [+4.1] | +3.2 [+4.8] |
| A, projected 15+ points | +0.0088 [+0.0084] | +4.8 [+6.4] | +7.5 [+6.4] |
| B: player points over + game over | +0.0036 (+0.0021 to +0.0052) [+0.0051] | +7.4 [+7.5] | +8.7 [+8.8] |
| B, projected 15+ points | +0.0124 [+0.0160] | +8.2 [+8.5] | +13.7 [+12.8] |
| C (control): home covers + game over | -0.0000 (-0.0009 to +0.0009) | -0.3 [-0.0] | +4.5 [-1.2] |
| C2 (control): home wins + game over | +0.0001 (-0.0005 to +0.0007) | -1.9 [+0.6] | +0.3 [+0.0] |
| D (info): top two teammates both over | +0.0045 (not trusted, see below) | +5.5 [+5.5] | +1.8 [+0.1] |
| E (info): both teams' top scorers over | +0.0010 | +1.7 [+1.0] | +8.2 [+2.8] |

- A and B pass the rule (beat legs multiplied on 2025-26, intervals above zero, same in 2024-25), and the correlation
  matches reality for players overall. For stars it is understated, most for points + game over (+8 vs +13).
- Side + total really is close to independent (C, C2): the engine's "no correlation" is right within noise.
- Teammates: the engine says +5.5, reality +0.1 to +1.8. Teammates' overs are nearly independent, not negative; the
  engine overstates it. Opposing stars: the engine says +1 to +2, reality +3 to +8 (shared game pace and environment).
- **Blocker found by this test:** single-leg points props are compressed toward the middle. On book-style lines the
  model's over chance averages 36% for players projected 15+ points but they go over 49%; bench players run the other
  way. By season average, 25+ scorers: projected 23.0, actual 26.8 (2025-26). The log-loss gains for stars and for
  D/E above are mostly this bias, not correlation. `validation/props_diag.py` (2024-25, 20-25 point scorers): per-minute
  rates are right before shrinkage; shrinkage toward the position average costs 1.0 point, the minutes projection
  (pulled toward a 14-minute prior) costs 1.4, rescaling to team points 0.3. The same bias is in the live app's points
  props. Fix and retest single legs before building SGP pricing.

### Props retuned (Nov 5, 2026; not live yet) and the joint test rerun
`validation/tune_props2.py` (fast harness, reproduces `validation/props.py` within 0.003 log loss), tuned on 2024-25
only, then the official `validation/props.py` with `PROPS_V2=validation/props2_tuned.json` on both seasons:
- Minutes: half-life 4 games (was 8), prior weight a sixth of before. Rates: shrink toward the player's own long-run
  rate (half-life 720 days) with 50 minutes of prior (rebounds 150), half-life 90 days; was 300 minutes toward the
  position average, half-life 180. Points and PRA: dispersion proportional to the player's mean (phi = 0.4 x mean);
  rebounds, assists and threes keep one dispersion (proportional lost on 2024-25).
- 2025-26, log loss over all lines, old -> new (season-average baseline): points 0.3616 -> 0.3473 (0.3809),
  assists 0.3891 -> 0.3786 (0.4012), threes 0.4482 -> 0.4373 (0.4583), PRA 0.4014 -> 0.3843 (0.4337),
  rebounds 0.4254 -> 0.4215 (0.4381; overs still 5 points high, stays off). `validation/props_test_2026_v2.log`.
- Bench players fixed (points, season average 0-5: over predicted 46% vs 48% actual; was 65%). Stars improved but
  still low: 20+ point scorers, book-line over 38% predicted vs 44% actual (was 31-33%). About half is the mean
  (3% low: minutes 0.6 short, team points 1 low) and half the shape (model spread 8.9 points vs 7.8 actual).
  Props stay paused in the app until this is fixed. The code is in place behind `props.v2` / `props.disp` in
  settings (`pipeline/export.py` rates and prop minutes `pmp`, `playerProps` dispersion), off by default.

Joint test on the retuned props (`validation/sgp_2025_v2.log`, `sgp_2026_v2.log`), 2025-26 [2024-25], gap in points:

| Combo | Log loss gain vs legs (95%) | Gap model | Gap actual |
|---|---|---|---|
| A: points over + own team wins | +0.0019 (+0.0014 to +0.0023) [+0.0021] | +2.5 [+3.7] | +3.2 [+4.8] |
| A, projected 15+ | +0.0051 | +5.4 [+7.4] | +7.8 [+6.7] |
| B: points over + game over | +0.0041 (+0.0027 to +0.0057) [+0.0057] | +7.2 [+7.3] | +8.7 [+8.8] |
| B, projected 15+ | +0.0079 | +10.1 [+10.1] | +13.5 [+13.5] |
| F (3 legs): points over + team wins + game over | +0.0037 (+0.0024 to +0.0049) [+0.0046] | +5.6 [+7.1] | +7.4 [+8.7] |
| G (3 legs): top two teammates over + team wins | +0.0098 | +7.1 [+8.9] | +5.0 [+3.1] |
| D: top two teammates both over | +0.0024 | +7.3 [+7.4] | +3.5 [+1.8] |
| E: both teams' top scorers over | +0.0007 | +2.1 [+0.6] | +9.4 [+4.9] |
| C, C2 controls (side + total) | about 0, intervals span 0 | about 0 | about 0 (noise) |

- Player + team legs (A, B, F) pass in both seasons; the engine captures roughly 70-85% of the real correlation, a
  little less for stars. Teammates are overstated about 2-4x; opposing stars understated about 4x. Both point to the
  same two gaps: no shared game-level swing (pace or shooting environment hitting both teams) and no competition
  for shots between teammates (each player is drawn independently given his team's points).

### Props step 3 (Nov 5, 2026; not live): star calibration
`validation/tune_props2.py tune3/test3`, `validation/points_shape.py` (logs `props3_*`, `points_shape_*`). Tuned on
2024-25, tested on 2025-26. Book-style line over % for the top fifth of scorers (season average 16+), predicted vs actual:

| 2025-26 top fifth | Step 2 | Step 3 | Actual |
|---|---|---|---|
| Points | 38.6% | 40.1% | 44.2% |
| Assists | 40.4% | 40.6% | 43.7% |
| Threes | 37.9% | 37.6% | 39.6% |
| Points + rebounds + assists | 42.3% | 43.6% | 44.9% |

- Step 3 = two-parameter dispersion (phi = phi10 x (mean/10)^g; won in every market on 2024-25) and a 1.5% team-points
  calibration for points-based props. Minutes: a finer search kept step 2. All-lines log loss 2025-26: points
  0.3473 -> 0.3460, assists 0.3786 -> 0.3782, threes 0.4374 -> 0.4370, PRA 0.3842 -> 0.3836.
- Fitting the spread on book-style lines only does not move stars (2024-25: 43.6% -> 43.4%).
- Points as scoring plays (free throws, twos, threes in the player's own mix, compound Poisson, gamma night factor):
  2024-25 all lines 0.3483 -> 0.3473, book lines 0.6662 -> 0.6624, stars 43.6% -> 44.0% (actual 47.1%). Better, kept as
  a candidate.
- Why stars stay low: their actual mean sits slightly below the line yet they go over 47%, so their distribution has a
  short-game tail on the left (early exits, foul trouble, sitting out blowouts); every shape tried leans right. A
  left-skewed night factor moves stars to 46.7% (2024-25) but, with one size for everyone, worsens overall log loss
  (0.3473 -> 0.3517). Next: a per-player minutes distribution (stable role plus occasional short nights; bench
  minutes swing far more), which also matters for SGPs (blowouts change star minutes).

## Layout
- `pipeline/fetch_data.sh`: play-by-play, box scores and schedules from the sportsdataverse-data GitHub releases
  (refreshed daily around 5 AM ET). Raw files are not committed.
- `data/derived/`: per-season games, team-game counts, possessions and player box scores (`pipeline/prep.py`).
- `data/injuries/`: what RotoWire listed each run. `data/odds/`: Pinnacle snapshots. `data/predictions/`: the model's
  pre-game numbers per run (games and player props), scored by `pipeline/score.py`. Never edit these logs.
- `data/market_hist/`: DraftKings closing lines from ESPN's game pages (2025-26), used only for the held-out comparison.
- `build/published_ratings.json`: ratings currently live in the app.
- `validation/`: tuning, held-out tests and logs.
- `DAILY.md`: runbook for the twice-daily update.
