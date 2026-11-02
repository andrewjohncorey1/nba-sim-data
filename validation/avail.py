"""Player-availability adjustment for rolling tests: players who actually played are treated as the known pre-game
availability (inactive lists are public before tip-off); minutes are projected from earlier games only."""
import sys, os, json
sys.path.insert(0, 'pipeline')
import numpy as np, pandas as pd
import players as PS

PRIOR = json.load(open('validation/player_prior.json')) if os.path.exists('validation/player_prior.json') else None


FITS, DAYS, SBAR = {}, {}, {}      # shared across instances (tuning reuses them)


class Avail:
    def __init__(self, PL, prm_p, half_life_team, alpha, sens, ppp0, refit_days=7):
        self.PL, self.prm, self.hl, self.alpha, self.sens, self.ppp0 = PL, prm_p, half_life_team, alpha, sens, ppp0
        self.refit_days = refit_days
        self.played = PL[PL['min'] > 0].groupby(['game_id', 'team']).pid.apply(list).to_dict()

    def _state(self, d):
        d = pd.Timestamp(d)
        key = d - pd.Timedelta(days=(d - pd.Timestamp('2000-01-03')).days % self.refit_days)
        fk = (key, tuple(sorted(self.prm.items())))
        if fk not in FITS:
            R = PS.fit(self.PL, key, self.prm, prior=PRIOR)
            low = R[R.wmin < 300]
            repl = (float(np.average(low.O, weights=low.wmin + 1)), float(np.average(low.D, weights=low.wmin + 1)))
            FITS[fk] = (R, repl)
        if d not in DAYS: DAYS[d] = PS.projected_minutes(self.PL, d)
        return FITS[fk], DAYS[d]

    def delta(self, game_id, team, d):
        (R, repl), pm = self._state(d)
        sk = (team, pd.Timestamp(d), self.hl)
        if sk not in SBAR: SBAR[sk] = PS.team_shares(self.PL, team, d, self.hl)
        ids = self.played.get((game_id, team), [])
        am = pm.proj.reindex(ids).fillna(12.0)
        return PS.availability_delta(R, SBAR[sk], am, repl)

    def __call__(self, r, Rt):
        if self.alpha == 0: return 0.0, 0.0
        oh, dh = self.delta(r.game_id, r.home, r.date)
        oa, da = self.delta(r.game_id, r.away, r.date)
        f = lambda pts100: np.log(max(0.5, 1 + pts100 / 100 / self.ppp0)) / self.sens
        return f(self.alpha * (oh - da)), f(self.alpha * (oa - dh))
