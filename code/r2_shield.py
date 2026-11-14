"""
Step 7: the emergency brake for the second reactor. Same rule as shield.py (reactor 1):

    A proposed feed-valve opening OP is SAFE if, for every model in the brake's model set,
      1) holding OP for the next 50 s keeps T <= MAT (373 K), and
      2) after that, the backup action (feed valve shut, coolant valve stays at its setting)
         keeps T <= MAT until MTSR <= MAT, i.e. until even a total cooling failure could no
         longer push the reactor past MAT (Stoessel criterion).
    Otherwise the brake returns the largest safe opening (bisection). OP = 0 is the fallback.

Differences from reactor 1: the jacket temperature is a state (it is simulated too), and the
brake may shut the valve faster than the AI's +/-5 % per step limit (an emergency shut-off).
The robust ('careful') brake checks the worst-case corners of the +/-20 % uncertainty set.

Step 7d (coolant valve as a second control, filter2): the step check holds the REQUESTED coolant
opening OPJ for 50 s; the backup then uses
    backup_OPJ = 100.0   option A: the emergency procedure opens the coolant fully
    backup_OPJ = "hold"  option B: the coolant stays where it was (no emergency opening)
If the request is unsafe, the brake first cuts the feed (removes the heat source) and only if no
feed at all is safe at the requested coolant does it open the coolant (bisection again).
With OPJ = None everywhere (the default) the brake behaves exactly as in Step 7.

Step 7o (delay-aware brake, cf_delay_steps > 0; default 0 = off): an extra test for every model.
The cooling fails at the START of this step; the feed stays at the proposed OP for cf_delay_steps
steps (operators need 100 s to react, as in the cooling-failure test), then it is shut, and the
reactor is followed with no coolant flow until MTSR <= MAT (safe from then on) or for cf_tail_steps
(2 h, the test's horizon). T must stay <= MAT throughout. The jacket's water still absorbs heat but
no fresh coolant flows (OPJ = 0), exactly as in the test. Without this test the brake assumed the
feed stops at the instant of a failure; controllers running close to the limit then passed MAT by
up to 1.2 K in the cooling-failure test (Step 7d/7m). cf_models picks the models for this extra test:
None = the brake's worst-case corners ('robust', very cautious), [nominal] = the design model only.
"""
from __future__ import annotations

from typing import Sequence

from r2_model import IT, R2Params, advance, mtsr


class R2Shield:
    def __init__(self, models: Sequence[R2Params], backoff: float = 0.0,
                 horizon_s: float = 7200.0, backup_h: float = 5.0, bisection_iters: int = 8,
                 backup_OPJ: float | str | None = None, cf_delay_steps: int = 0,
                 cf_tail_steps: int = 144, cf_models: Sequence[R2Params] | None = None):
        self.models = list(models)
        self.cf_delay_steps, self.cf_tail_steps = cf_delay_steps, cf_tail_steps
        self.cf_models = list(cf_models) if cf_models else None   # None: the brake's own (corner) models
        self.backup_OPJ = backup_OPJ      # None: model default (80 %), number: fixed, "hold": keep
        self.p0 = self.models[0]
        self.T_lim = self.p0.MAT - backoff
        self.H = int(horizon_s / self.p0.dt_control)
        self.backup_h = backup_h          # RK4 step during the backup check (checked in r2_test_shield)
        self.iters = bisection_iters      # 8 halvings: opening resolved to within 0.4 %
        self._order = list(range(len(self.models)))
        self.n_interventions = 0

    def _backup_opj(self, OPJ):
        if self.backup_OPJ == "hold":
            return OPJ
        return self.backup_OPJ           # None -> model default, or a fixed opening (e.g. 100)

    def _safe_one(self, s, OP: float, p: R2Params, OPJ: float | None = None) -> bool:
        lim = self.T_lim
        y = advance(s, OP, p, OPJ=OPJ)
        if y[IT] > lim:
            return False
        OPJ_b = self._backup_opj(OPJ)
        for _ in range(self.H):
            if mtsr(y, p) <= lim:
                return True
            y = advance(y, 0.0, p, h=self.backup_h, OPJ=OPJ_b)
            if y[IT] > lim:
                return False
        return False                     # not provably safe within the horizon -> unsafe

    def _survives_cooling_failure(self, s, OP: float, p: R2Params) -> bool:
        """Step 7o: cooling fails now, feed held at OP for the operator delay, then shut; no coolant flow."""
        lim, y = self.T_lim, s
        for _ in range(self.cf_delay_steps):
            y = advance(y, OP, p, OPJ=0.0)
            if y[IT] > lim:
                return False
        for _ in range(self.cf_tail_steps):
            if mtsr(y, p) <= lim:
                return True
            y = advance(y, 0.0, p, h=self.backup_h, OPJ=0.0)
            if y[IT] > lim:
                return False
        return True                      # stayed below MAT for the whole 2 h test horizon

    def is_safe(self, s, OP: float, OPJ: float | None = None) -> bool:
        for j, i in enumerate(self._order):
            if not self._safe_one(s, OP, self.models[i], OPJ):
                if j:                     # check the model that failed first next time (same answer, faster)
                    self._order.insert(0, self._order.pop(j))
                return False
        if self.cf_delay_steps:           # Step 7o: also survive a cooling failure with the operator delay
            return all(self._survives_cooling_failure(s, OP, p) for p in (self.cf_models or self.models))
        return True

    def filter(self, s, OP_proposed: float) -> float:
        if OP_proposed <= 0.0 or self.is_safe(s, OP_proposed):
            return OP_proposed
        self.n_interventions += 1
        if not self.is_safe(s, 0.0):
            return 0.0
        lo, hi = 0.0, OP_proposed
        for _ in range(self.iters):
            m = 0.5 * (lo + hi)
            if self.is_safe(s, m):
                lo = m
            else:
                hi = m
        return lo

    def filter2(self, s, OP_req: float, OPJ_req: float) -> tuple[float, float]:
        """Two valves (Step 7d). Returns (feed opening, coolant opening) actually applied."""
        if OP_req <= 0.0 and self.is_safe(s, 0.0, OPJ_req):
            return 0.0, OPJ_req
        if OP_req > 0.0 and self.is_safe(s, OP_req, OPJ_req):
            return OP_req, OPJ_req
        self.n_interventions += 1
        if self.is_safe(s, 0.0, OPJ_req):          # 1) cut the feed, keep the requested coolant
            lo, hi = 0.0, OP_req
            for _ in range(self.iters):
                m = 0.5 * (lo + hi)
                if self.is_safe(s, m, OPJ_req):
                    lo = m
                else:
                    hi = m
            return lo, OPJ_req
        if not self.is_safe(s, 0.0, 100.0):        # 2) nothing is provably safe: feed shut, full cooling
            return 0.0, 100.0
        lo, hi = OPJ_req, 100.0                    # 3) feed shut, smallest coolant opening that is safe
        for _ in range(self.iters):
            m = 0.5 * (lo + hi)
            if self.is_safe(s, 0.0, m):
                hi = m
            else:
                lo = m
        return 0.0, hi


def robust_models(p: R2Params, mismatch: float = 0.2):
    """Worst-case corners: high reaction heat, low cooling; both kinetic extremes."""
    return [p.with_mismatch(k_factor=kf, dH_factor=1 + mismatch, UA_factor=1 - mismatch)
            for kf in (1 - mismatch, 1 + mismatch)]
