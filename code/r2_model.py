"""
Step 7: second reactor, the benchmark semi-batch reactor of Kummer et al. 2020 / Fricz et al. 2026.

    A + B -> C   (exothermic, second order)

Reagent A is fed through valve V1 into a jacketed reactor that holds B. This is the
reactor used by the closest paper (Fricz et al. 2026, Comput. Chem. Eng. 212, 109704,
Appendix A), so our results can be put next to theirs.

Model (Fricz 2026, Eqs. A.1-A.9; SI units with kJ and kW):
    r       = k0 exp(-E/(R T)) cA cB                         [kmol/(m3 s)]
    dV/dt   = F
    dnA/dt  = F c_in - V r
    dnB/dt  = -V r ,  dnC/dt = V r
    dT/dt   = [F rho_cp (T_in - T) + (-dH) V r - UA (T - TJ)] / (V rho_cp)
    dTJ/dt  = [FJ rho_cpJ (TJ_in - TJ) + UA (T - TJ)] / (VJ rho_cpJ)
    UA      = UA0 (1 + V_dos / V0)
    F       = F_max * OP/100 ,  FJ = FJ_max * OP_J/100       (linear valves)

All numbers come from Fricz 2026 Tables A.5 and A.6 unless marked otherwise.
The model is written on plain Python floats (no NumPy): the brake simulates it thousands
of times per decision, and plain floats are about 5-10x faster for a 7-element state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

# state tuple indices
IV, IA, IB, IC, IT, ITJ, IFED = range(7)
STATE_NAMES = ["V_m3", "nA_kmol", "nB_kmol", "nC_kmol", "T_K", "TJ_K", "nA_fed_kmol"]


@dataclass(frozen=True)
class R2Params:
    # ---- kinetics and thermochemistry (Fricz 2026, Table A.5) ----
    k0: float = 1.465e7            # m3/(kmol s)
    E_R: float = 8500.0            # K   (E_A / R_g)
    dH: float = -3.5e5             # kJ/kmol
    rho_cp: float = 4800.0         # kJ/(m3 K), reactor contents
    rho_cpJ: float = 4183.0        # kJ/(m3 K), jacket water
    UA0: float = 1.85              # kW/K
    V0: float = 0.5                # m3, initial liquid volume
    VJ: float = 0.41               # m3, jacket volume
    MAT: float = 373.0             # K, Maximum Allowable Temperature = our hard limit

    # ---- operation (Fricz 2026, Table A.6) ----
    nB0: float = 5.0               # kmol
    nA_feed: float = 5.0           # kmol to feed (= nB0)
    c_in: float = 5.0              # kmol/m3, feed concentration of A
    F_max: float = 1.4e-4          # m3/s
    FJ_max: float = 1.1e-3         # m3/s
    T0: float = 340.0              # K, initial reactor temperature
    TJ0: float = 298.0             # K
    T_in: float = 298.0            # K, feed temperature
    TJ_in: float = 298.0           # K, coolant inlet temperature

    # ---- not stated in Fricz 2026 text ----
    # Coolant valve during feeding: ASSUMPTION 80 %. Fricz 2026 says the jacket flow is
    # kept "at a pre-defined setpoint" in the feeding phase but gives no value; the same
    # group's open-access paper on this reactor family (Kummer, Varga, Nagy 2020,
    # Comput. Chem. Eng. 141, 106998, Sec. 2.1) uses 80 % of the cooling capacity.
    OPJ: float = 80.0              # %
    # Temperature setpoint 363 K: read from Fricz 2026 Fig. 4 (dashed line), and consistent
    # with the reward at t = 0 (-23 = -|340 - 363|).
    T_sp: float = 363.0            # K

    # ---- simulation choices (ours) ----
    dt_control: float = 50.0       # s, action interval (Fricz 2026)
    h: float = 2.0                 # s, RK4 step (checked against LSODA in r2_test_model.py)
    t_final_max: float = 4 * 3600.0  # s, episode limit (Fricz 2026: 4 h)

    def with_mismatch(self, k_factor=1.0, dH_factor=1.0, UA_factor=1.0):
        """A 'true reactor' whose reaction speed, reaction heat and cooling differ from the model."""
        return replace(self, k0=self.k0 * k_factor, dH=self.dH * dH_factor, UA0=self.UA0 * UA_factor)


def initial_state(p: R2Params) -> tuple:
    return (p.V0, 0.0, p.nB0, 0.0, p.T0, p.TJ0, 0.0)


def rate(s, p: R2Params) -> float:
    """Reaction rate r [kmol/(m3 s)]."""
    V = s[IV]
    cA = s[IA] / V if s[IA] > 0.0 else 0.0
    cB = s[IB] / V if s[IB] > 0.0 else 0.0
    return p.k0 * math.exp(-p.E_R / s[IT]) * cA * cB


def UA(s, p: R2Params) -> float:
    return p.UA0 * (1.0 + (s[IV] - p.V0) / p.V0)


def rhs(s, F: float, FJ: float, p: R2Params):
    V, T, TJ = s[IV], s[IT], s[ITJ]
    Vr = V * rate(s, p)
    ua = UA(s, p)
    q_cool = ua * (T - TJ)
    dT = (F * p.rho_cp * (p.T_in - T) + (-p.dH) * Vr - q_cool) / (V * p.rho_cp)
    dTJ = (FJ * p.rho_cpJ * (p.TJ_in - TJ) + q_cool) / (p.VJ * p.rho_cpJ)
    return (F, F * p.c_in - Vr, -Vr, Vr, dT, dTJ, F * p.c_in)


def _rk4(s, F, FJ, p, h):
    k1 = rhs(s, F, FJ, p)
    s2 = tuple(a + 0.5 * h * b for a, b in zip(s, k1))
    k2 = rhs(s2, F, FJ, p)
    s3 = tuple(a + 0.5 * h * b for a, b in zip(s, k2))
    k3 = rhs(s3, F, FJ, p)
    s4 = tuple(a + h * b for a, b in zip(s, k3))
    k4 = rhs(s4, F, FJ, p)
    return tuple(a + h / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4))


def advance(s, OP: float, p: R2Params, dt: float | None = None, OPJ: float | None = None,
            h: float | None = None):
    """Move the reactor forward dt seconds with feed valve OP (%) and coolant valve OPJ (%) held."""
    dt = p.dt_control if dt is None else dt
    h = p.h if h is None else h
    F = p.F_max * OP / 100.0
    FJ = p.FJ_max * (p.OPJ if OPJ is None else OPJ) / 100.0
    n = max(1, int(round(dt / h)))
    hh = dt / n
    for _ in range(n):
        s = _rk4(s, F, FJ, p, hh)
    return s


def max_OP_for_remaining(s, p: R2Params, dt: float | None = None) -> float:
    """Largest valve opening that does not feed more than the planned charge in this interval."""
    dt = p.dt_control if dt is None else dt
    remaining = max(p.nA_feed - s[IFED], 0.0)
    return min(100.0, 100.0 * remaining / (p.c_in * p.F_max * dt))


def feeding_done(s, p: R2Params) -> bool:
    return s[IFED] >= p.nA_feed - 1e-9


# ---------------- safety indicators ----------------
def adiabatic_rise(s, p: R2Params) -> float:
    """K rise if all A that can still react (limited by B) reacted with no cooling."""
    n = min(max(s[IA], 0.0), max(s[IB], 0.0))
    return n * (-p.dH) / (s[IV] * p.rho_cp)


def mtsr(s, p: R2Params) -> float:
    """Stoessel MTSR: temperature reached if the cooling failed now (reactor contents only;
    ignoring the jacket's heat capacity is conservative)."""
    return s[IT] + adiabatic_rise(s, p)


def mdc(s, p: R2Params) -> float:
    """Modified Dynamic Condition (Kummer & Varga 2019) as used by Fricz 2026, Eq. (14)-(15):
        MDC = cA cB E/(R T^2) - (cA + cB)/beta - cA cB/(T - TJ),  beta = -dH/rho_cp.
    MDC > 0 means the reactor has potential for runaway."""
    V, T, TJ = s[IV], s[IT], s[ITJ]
    cA, cB = max(s[IA], 0.0) / V, max(s[IB], 0.0) / V
    beta = -p.dH / p.rho_cp
    dT = T - TJ
    last = cA * cB / dT if abs(dT) > 1e-9 else 0.0
    return cA * cB * p.E_R / T ** 2 - (cA + cB) / beta - last
