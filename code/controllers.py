"""
Baseline dosing controllers ("normal methods") for the semi-batch reactor.

1. FixedTiming  - constant feed rate for a fixed dosing time (industrial recipe practice).
2. PIDDosing    - feed rate manipulated by a PI(D) loop on reactor temperature
                  (temperature-controlled dosing), with anti-windup.

Both are tuned on the NOMINAL model only, exactly as an engineer would do before the plant
is built. Robustness is then tested on mismatched 'true plants'.

Tuning objective for both: shortest batch time subject to
    peak reactor temperature <= T_max - backoff      (nominal model)
`backoff` (K) is the safety margin the designer keeps. Same backoff will be used for the
RL agent and its shield, so the comparison is fair.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from reactor_model import IT, ReactorParams, simulate


# ------------------------------------------------------------------ Fixed timing
@dataclass
class FixedTiming:
    F_mL_min: float

    def reset(self):
        pass

    def __call__(self, t, x):
        return self.F_mL_min


def tune_fixed_timing(p: ReactorParams, backoff: float = 0.0, tol_min: float = 0.05,
                      check_plants=None):
    """Bisection on dosing time: shortest constant-rate dose with T_peak <= T_max - backoff.
    check_plants: optional list of plants on which the limit must hold (robust tuning)."""
    T_lim_C = p.T_max - 273.15 - backoff
    plants = [p] + list(check_plants or [])

    def peak(t_dose_min):
        c = FixedTiming(p.V_A_total_mL / t_dose_min)
        return max(simulate(c, q).T_peak_C for q in plants), None

    lo = p.V_A_total_mL / p.F_max_mL_min  # fastest the pump allows
    if peak(lo)[0] <= T_lim_C:
        return FixedTiming(p.F_max_mL_min), lo
    hi = lo
    while peak(hi)[0] > T_lim_C:
        hi *= 1.5
    while hi - lo > tol_min:
        mid = 0.5 * (lo + hi)
        if peak(mid)[0] <= T_lim_C:
            hi = mid
        else:
            lo = mid
    return FixedTiming(p.V_A_total_mL / hi), hi


# ------------------------------------------------------------------ PID
@dataclass
class PIDDosing:
    T_sp_C: float          # temperature setpoint
    Kp: float              # (mL/min) per K
    Ti: float              # integral time, s (math.inf = P only)
    Td: float = 0.0        # derivative time, s
    F_max: float = 20.0
    dt: float = 10.0

    def reset(self):
        self.I = 0.0
        self.e_prev = None

    def __call__(self, t, x):
        e = self.T_sp_C - (x[IT] - 273.15)        # positive = colder than setpoint -> dose more
        d = 0.0 if (self.e_prev is None or self.Td == 0) else (e - self.e_prev) / self.dt
        self.e_prev = e
        I_new = self.I + (e * self.dt / self.Ti if math.isfinite(self.Ti) else 0.0)
        u = self.Kp * (e + I_new + self.Td * d)
        # anti-windup: only accept the integral update if output is not saturated in that direction
        if (u > self.F_max and e > 0) or (u < 0 and e < 0):
            u = self.Kp * (e + self.I + self.Td * d)
        else:
            self.I = I_new
        return min(max(u, 0.0), self.F_max)


def tune_pid(p: ReactorParams, backoff: float = 0.0, verbose: bool = False, check_plants=None):
    """Grid search over setpoint, gains on the nominal model.
    check_plants: optional list of plants on which the limit must also hold (robust tuning)."""
    T_lim_C = p.T_max - 273.15 - backoff
    others = list(check_plants or [])
    best = None
    sp_grid = np.arange(T_lim_C - 12.0, T_lim_C + 0.01, 1.0)
    Kp_grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 4.0]
    Ti_grid = [300.0, 600.0, 1200.0, 2400.0, math.inf]
    Td_grid = [0.0, 60.0, 180.0]
    for sp, Kp, Ti, Td in itertools.product(sp_grid, Kp_grid, Ti_grid, Td_grid):
        c = PIDDosing(float(sp), Kp, Ti, Td, F_max=p.F_max_mL_min, dt=p.dt_control)
        r = simulate(c, p)
        if r.batch_time_s is None or r.T_peak_C > T_lim_C:
            continue
        if best is not None and r.batch_time_s >= best[1]:
            continue
        if any(simulate(c, q).T_peak_C > T_lim_C for q in others):
            continue
        if best is None or r.batch_time_s < best[1]:
            best = (c, r.batch_time_s)
    if best is None:
        raise RuntimeError("no safe PID tuning found on the grid")
    if verbose:
        c = best[0]
        print(f"PID best: sp={c.T_sp_C:.1f} C Kp={c.Kp} Ti={c.Ti} Td={c.Td} batch={best[1]/60:.1f} min")
    return best[0], best[1]
