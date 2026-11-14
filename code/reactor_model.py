"""
Virtual semi-batch reactor: hydrolysis of acetic anhydride.

    (CH3CO)2O + H2O  ->  2 CH3COOH        (exothermic)

Acetic anhydride (A) is dosed into a jacketed reactor that initially holds water (W).
Product is acetic acid (P).

Model equations (per second, SI units except where noted)
---------------------------------------------------------
    r        = k(T) * nA * (C_W / C_W_ref)^water_order        [mol/s]
    k(T)     = k_ref * exp(-Ea/R * (1/T - 1/T_ref))           [1/s]

    dnA/dt   = F_mol - r
    dnW/dt   = -r
    dnP/dt   = 2 r
    dnFed/dt = F_mol                                         (book-keeping only)

    C_tot dT/dt = F_mol*Cp_A*(T_feed - T)      sensible heat of the cold feed
                + (-dH_r) * r                  heat released by reaction
                - F_mol * dH_mix               heat of mixing (endothermic if > 0)
                - UA * (T - T_jacket)          jacket cooling

    C_tot    = nA*Cp_A + nW*Cp_W + nP*Cp_P      [J/K]
    V        = sum(n_i * M_i / rho_i)          ideal (additive) volumes [L]

The default water_order = 0 is the pseudo-first-order rate law as reported in the
kinetic source (water in large excess). water_order = 1 is available as a sensitivity
option (it accounts for dilution of water as the batch proceeds).

Every number in `ReactorParams` carries its source or is marked ASSUMPTION.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

R_GAS = 8.314462618  # J/(mol K), CODATA


def C_to_K(t_c: float) -> float:
    return t_c + 273.15


def K_to_C(t_k: float) -> float:
    return t_k - 273.15


@dataclass(frozen=True)
class ReactorParams:
    # ---------------- Kinetics ----------------
    # Pseudo-first-order rate constant 0.129 1/min at 20 C (lowest temperature reported).
    # Source: "Characterization of reaction enthalpy and kinetics in a microscale flow
    # platform", React. Chem. Eng. 2020, DOI 10.1039/D0RE00304B (MIT DSpace 1721.1/128541).
    # Abstract: "0.129 min-1 up to 0.522 min-1 for temperatures between 20 C and 56.3 C".
    k_ref: float = 0.129 / 60.0            # 1/s
    T_ref: float = C_to_K(20.0)            # K
    # Ea from the two end points of that same study (two-point Arrhenius, DERIVED):
    #   Ea = R ln(0.522/0.129) / (1/293.15 - 1/329.45) = 30.9 kJ/mol
    # Cross-check: Magritek NMR application note reports 6.79 kcal/mol = 28.4 kJ/mol.
    Ea: float = R_GAS * math.log(0.522 / 0.129) / (1 / 293.15 - 1 / 329.45)  # J/mol
    water_order: float = 0.0               # 0 = rate law as reported (pseudo-1st order)

    # ---------------- Thermochemistry ----------------
    # Reaction enthalpy -63 +/- 3.0 kJ/mol (same study, measured).
    # Cross-checks: Hess's law -58.2 kJ/mol (same study SI), NIST -56.6, Syrris -60.86.
    dH_r: float = -63.0e3                  # J/mol
    # Heat of mixing: the study measured +8.8 kJ/mol (endothermic) for a 3:1 Ac2O:AcOH
    # feed solution, not pure Ac2O. Default 0 -> CONSERVATIVE (ignoring an endothermic
    # term over-predicts temperature). Kept as a sensitivity option.
    dH_mix: float = 0.0                    # J/mol of A fed

    # Liquid heat capacities at 20 C, J/(mol K). Source: same study, SI Table S1.
    Cp_A: float = 168.2
    Cp_W: float = 75.3
    Cp_P: float = 120.5

    # Molar masses g/mol (IUPAC atomic weights) and densities g/mL near 20-25 C.
    M_A: float = 102.09
    M_W: float = 18.015
    M_P: float = 60.052
    rho_A: float = 1.082                   # Syrris tech note (also standard handbook value)
    rho_W: float = 0.998                   # water, 20 C
    rho_P: float = 1.049                   # acetic acid, handbook value

    # ---------------- Reactor & utilities (ASSUMPTIONS, 1 L lab scale) ----------------
    V_water0_mL: float = 400.0             # ASSUMPTION: initial water charge
    n_A_total: float = 3.0                 # ASSUMPTION: total anhydride to dose, mol (~283 mL)
                                           #   final volume ~0.68 L in a 1 L vessel
    UA: float = 3.0                        # ASSUMPTION: W/K, jacketed glass 1 L reactor
                                           #   (U ~ 50-150 W/m2K x A ~ 0.03-0.04 m2 -> 1.5-6 W/K)
    T_jacket: float = C_to_K(20.0)         # ASSUMPTION: thermostat coolant temperature
    T_feed: float = C_to_K(20.0)           # ASSUMPTION: feed at room temperature
    T0: float = C_to_K(20.0)               # ASSUMPTION: initial reactor temperature
    F_max_mL_min: float = 20.0             # ASSUMPTION: dosing pump maximum (full pump speed
                                           #   is deliberately UNSAFE: peak ~74 C)
    # Hard safety limit. 50 C keeps the model INSIDE the temperature range where the
    # kinetics were measured (20-56.3 C), so no extrapolation happens at the limit.
    T_max: float = C_to_K(50.0)

    # ---------------- Batch definition ----------------
    conversion_target: float = 0.99        # batch done when 99% of total charge has reacted
    t_final_max: float = 4 * 3600.0        # s, give up after 4 h
    dt_control: float = 10.0               # s, controller sampling time (zero-order hold)

    # ------------- derived -------------
    @property
    def n_W0(self) -> float:
        return self.V_water0_mL * self.rho_W / self.M_W

    @property
    def mol_per_mL_A(self) -> float:
        return self.rho_A / self.M_A

    @property
    def V_A_total_mL(self) -> float:
        return self.n_A_total / self.mol_per_mL_A

    def with_mismatch(self, k_factor=1.0, dH_factor=1.0, UA_factor=1.0, Ea_factor=1.0):
        """Return a 'true plant' whose parameters differ from the nominal model."""
        return replace(self, k_ref=self.k_ref * k_factor, dH_r=self.dH_r * dH_factor,
                       UA=self.UA * UA_factor, Ea=self.Ea * Ea_factor)


# State vector indices
IA, IW, IP, IT, IFED = range(5)
STATE_NAMES = ["n_A", "n_W", "n_P", "T", "n_fed"]


def initial_state(p: ReactorParams) -> np.ndarray:
    return np.array([0.0, p.n_W0, 0.0, p.T0, 0.0])


def volume_L(x: np.ndarray, p: ReactorParams) -> float:
    return (x[IA] * p.M_A / p.rho_A + x[IW] * p.M_W / p.rho_W + x[IP] * p.M_P / p.rho_P) / 1000.0


def heat_capacity(x: np.ndarray, p: ReactorParams) -> float:
    return x[IA] * p.Cp_A + x[IW] * p.Cp_W + x[IP] * p.Cp_P


def rate_constant(T: float, p: ReactorParams) -> float:
    return p.k_ref * math.exp(-p.Ea / R_GAS * (1.0 / T - 1.0 / p.T_ref))


def reaction_rate(x: np.ndarray, p: ReactorParams) -> float:
    nA = max(x[IA], 0.0)
    r = rate_constant(x[IT], p) * nA
    if p.water_order != 0.0:
        V = volume_L(x, p)
        C_W = max(x[IW], 0.0) / V
        C_W_ref = p.rho_W * 1000.0 / p.M_W  # pure water, mol/L
        r *= (C_W / C_W_ref) ** p.water_order
    return r


def rhs(t: float, x: np.ndarray, F_mL_min: float, p: ReactorParams) -> np.ndarray:
    F_mol = F_mL_min / 60.0 * p.mol_per_mL_A
    r = reaction_rate(x, p)
    T = x[IT]
    Q_feed = F_mol * p.Cp_A * (p.T_feed - T)
    Q_rxn = -p.dH_r * r
    Q_mix = -F_mol * p.dH_mix
    Q_cool = p.UA * (T - p.T_jacket)
    dT = (Q_feed + Q_rxn + Q_mix - Q_cool) / heat_capacity(x, p)
    return np.array([F_mol - r, -r, 2.0 * r, dT, F_mol])


def _rk4_step(x, F, p, h):
    k1 = rhs(0, x, F, p)
    k2 = rhs(0, x + 0.5 * h * k1, F, p)
    k3 = rhs(0, x + 0.5 * h * k2, F, p)
    k4 = rhs(0, x + h * k3, F, p)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def advance(x: np.ndarray, F_mL_min: float, p: ReactorParams, dt: float,
            method: str = "rk4", substeps: int = 10) -> np.ndarray:
    """Integrate the plant over one control interval with constant feed F."""
    if method == "rk4":
        h = dt / substeps
        for _ in range(substeps):
            x = _rk4_step(x, F_mL_min, p, h)
        return x
    elif method == "scipy":
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, (0.0, dt), x, args=(F_mL_min, p), method="LSODA",
                        rtol=1e-9, atol=1e-11)
        return sol.y[:, -1]
    raise ValueError(method)


# ---------------- Safety indicators ----------------
def adiabatic_rise_of_accumulation(x: np.ndarray, p: ReactorParams) -> float:
    """Temperature rise (K) if all unreacted anhydride reacted with no cooling."""
    nA = max(x[IA], 0.0)
    C_end = heat_capacity(x, p) - nA * p.Cp_A + nA * (2 * p.Cp_P - p.Cp_W)
    return nA * (-p.dH_r) / C_end


def mtsr(x: np.ndarray, p: ReactorParams) -> float:
    """Maximum Temperature of the Synthesis Reaction (Stoessel): T if cooling failed now."""
    return x[IT] + adiabatic_rise_of_accumulation(x, p)


def conversion(x: np.ndarray, p: ReactorParams) -> float:
    """Fraction of the TOTAL planned charge that has reacted."""
    return x[IP] / (2.0 * p.n_A_total)


# ---------------- Closed-loop simulation ----------------
@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray            # (N, 5)
    F: np.ndarray            # feed applied over each interval, mL/min
    T_C: np.ndarray
    mtsr_C: np.ndarray
    batch_time_s: Optional[float]
    T_peak_C: float
    violated: bool
    n_violation_steps: int
    info: dict = field(default_factory=dict)


def simulate(controller, p: ReactorParams, method: str = "rk4",
             stop_at_completion: bool = True) -> SimResult:
    """Run one batch. `controller(t, x, p_ctrl) -> F_mL_min`. The controller is reset first.

    The controller sees the measured state x; the plant is `p` (which may be a
    mismatched 'true plant'). The feed is clipped to the pump range and stops once
    the full charge has been dosed.
    """
    if hasattr(controller, "reset"):
        controller.reset()
    x = initial_state(p)
    t = 0.0
    ts, xs, Fs = [t], [x.copy()], []
    batch_time = None
    n_viol = 0
    while t < p.t_final_max - 1e-9:
        F = float(controller(t, x))
        F = min(max(F, 0.0), p.F_max_mL_min)
        remaining_mL = max(p.n_A_total - x[IFED], 0.0) / p.mol_per_mL_A
        F = min(F, remaining_mL / (p.dt_control / 60.0))  # never overdose the charge
        x = advance(x, F, p, p.dt_control, method)
        t += p.dt_control
        ts.append(t); xs.append(x.copy()); Fs.append(F)
        if x[IT] > p.T_max + 1e-6:      # 1 micro-K numerical tolerance
            n_viol += 1
        if batch_time is None and conversion(x, p) >= p.conversion_target:
            batch_time = t
            if stop_at_completion:
                break
    xs = np.array(xs)
    T_C = xs[:, IT] - 273.15
    m = np.array([mtsr(xi, p) for xi in xs]) - 273.15
    return SimResult(t=np.array(ts), x=xs, F=np.array(Fs), T_C=T_C, mtsr_C=m,
                     batch_time_s=batch_time, T_peak_C=float(T_C.max()),
                     violated=n_viol > 0, n_violation_steps=n_viol)
