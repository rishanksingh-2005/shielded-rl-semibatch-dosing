"""
Safety shield (prototype for Part 3) and the 'shield-only' reference controller.

Shield rule (look-ahead with a backup policy):
    A proposed feed F is SAFE if, for every model in the shield's model set,
      1) applying F for one control interval keeps T <= T_lim, and
      2) after that, the backup policy (feed stopped) keeps T <= T_lim until the
         reactor is provably safe.
    "Provably safe" = MTSR <= T_lim: even if all accumulated anhydride reacted with
    zero cooling, the temperature limit could not be exceeded (Stoessel criterion).
    If the proposal is unsafe, the shield returns the largest safe feed (bisection);
    F = 0 is always the fallback.

With one nominal model the shield inherits that model's errors. A ROBUST shield checks
the worst-case corners of the uncertainty set (+/- mismatch in k, dH, UA).
Assumption (to be revisited): the reactor state, including accumulated anhydride, is
measured (e.g. by inline IR / reaction calorimetry).
"""
from __future__ import annotations

import itertools
from typing import Sequence

from fast_model import FastModel
from reactor_model import IT, ReactorParams, advance, mtsr


class Shield:
    def __init__(self, models: Sequence[ReactorParams], backoff: float = 0.0,
                 horizon_s: float = 1800.0, bisection_iters: int = 10, fast: bool = True):
        self.models = list(models)
        # fast=True: same arithmetic on plain floats (fast_model.py), about 3x quicker.
        # fast=False: the original NumPy reference path (kept for verification).
        self.fast = fast and all(m.water_order == 0.0 for m in self.models)
        self.fms = [FastModel(m) for m in self.models] if self.fast else None
        self._order = list(range(len(self.models)))  # check the last failing model first
        self.p0 = self.models[0]
        self.T_lim = self.p0.T_max - backoff
        self.H = int(horizon_s / self.p0.dt_control)
        self.iters = bisection_iters
        self.n_interventions = 0

    def _safe_one(self, x, F, p: ReactorParams) -> bool:
        y = advance(x, F, p, p.dt_control)
        if y[IT] > self.T_lim:
            return False
        for _ in range(self.H):
            if mtsr(y, p) <= self.T_lim:
                return True
            y = advance(y, 0.0, p, p.dt_control, substeps=4)
            if y[IT] > self.T_lim:
                return False
        return False  # not provably safe within the horizon -> treat as unsafe

    def _safe_one_fast(self, s, F, fm: FastModel) -> bool:
        T_lim, dt = self.T_lim, self.p0.dt_control
        y = fm.advance(s, F, dt)
        if y[3] > T_lim:
            return False
        for _ in range(self.H):
            if fm.mtsr(y) <= T_lim:
                return True
            y = fm.advance(y, 0.0, dt, 4)
            if y[3] > T_lim:
                return False
        return False

    def is_safe(self, x, F) -> bool:
        # Safe only if safe for EVERY model. The order of checking does not change the
        # answer; checking the model that failed last time first just finds 'unsafe' sooner.
        if not self.fast:
            return all(self._safe_one(x, F, p) for p in self.models)
        s = tuple(float(v) for v in x)
        for j, i in enumerate(self._order):
            if not self._safe_one_fast(s, F, self.fms[i]):
                if j:
                    self._order.insert(0, self._order.pop(j))
                return False
        return True

    def filter(self, x, F_proposed: float) -> float:
        if self.is_safe(x, F_proposed):
            return F_proposed
        self.n_interventions += 1
        if not self.is_safe(x, 0.0):
            return 0.0
        lo, hi = 0.0, F_proposed
        for _ in range(self.iters):
            m = 0.5 * (lo + hi)
            if self.is_safe(x, m):
                lo = m
            else:
                hi = m
        return lo


def robust_models(p: ReactorParams, mismatch: float = 0.2):
    """Worst-case corners: high heat release, low cooling; both kinetic extremes."""
    return [p.with_mismatch(k_factor=kf, dH_factor=1 + mismatch, UA_factor=1 - mismatch)
            for kf in (1 - mismatch, 1 + mismatch)]


class ShieldedPolicy:
    """Any base policy (callable t, x -> F) passed through the shield."""

    def __init__(self, base, shield: Shield):
        self.base, self.shield = base, shield

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.shield.n_interventions = 0

    def __call__(self, t, x):
        return self.shield.filter(x, float(self.base(t, x)))


def max_feed_with_shield(shield: Shield):
    """'Shield-only' controller: always ask for full pump speed, let the shield cut it."""
    F_max = shield.p0.F_max_mL_min
    return ShieldedPolicy(lambda t, x: F_max, shield)
