"""
Physics checks for the reactor model. Run with:  python test_reactor_model.py
(or `pytest test_reactor_model.py` if pytest is installed).

Each test compares the simulator against something independent: the source data,
an analytical solution, or a conservation law.
"""
import math
from dataclasses import replace

import numpy as np

from controllers import FixedTiming, PIDDosing
from reactor_model import (IA, IFED, IP, IT, IW, ReactorParams, advance, heat_capacity,
                           initial_state, mtsr, rate_constant, simulate, volume_L)

P = ReactorParams()


def test_arrhenius_reproduces_source_data():
    # Source: 0.129 1/min at 20 C and 0.522 1/min at 56.3 C
    assert math.isclose(rate_constant(293.15, P) * 60, 0.129, rel_tol=1e-9)
    assert math.isclose(rate_constant(329.45, P) * 60, 0.522, rel_tol=1e-9)
    # Magritek cross-check: Ea 28.4 kJ/mol; ours must be within 10%
    assert abs(P.Ea / 1000 - 28.41) / 28.41 < 0.10


def test_isothermal_first_order_decay_matches_analytic():
    p = replace(P, dH_r=0.0, UA=0.0)           # no heat -> T stays at T0
    x = initial_state(p); x[IA] = 1.0
    k = rate_constant(p.T0, p)
    for _ in range(60):                        # 10 minutes
        x = advance(x, 0.0, p, 10.0)
    exact = math.exp(-k * 600.0)
    assert abs(x[IA] - exact) < 1e-8, (x[IA], exact)
    assert abs(x[IT] - p.T0) < 1e-12


def test_adiabatic_temperature_rise_matches_analytic():
    # Batch, no cooling: C(xi) dT = -dH dxi with C(xi) = C0 + dCp*xi
    p = replace(P, UA=0.0)
    n0 = 1.0
    x = initial_state(p); x[IA] = n0
    C0 = heat_capacity(x, p)
    dCp = 2 * p.Cp_P - p.Cp_A - p.Cp_W
    dT_exact = (-p.dH_r) / dCp * math.log(1 + dCp * n0 / C0)
    for _ in range(720):                       # 2 h, reaction complete
        x = advance(x, 0.0, p, 10.0)
    assert x[IA] < 1e-9
    assert abs((x[IT] - p.T0) - dT_exact) < 1e-6, (x[IT] - p.T0, dT_exact)


def test_mass_balance_closes_during_dosing():
    r = simulate(FixedTiming(8.0), P, stop_at_completion=False)
    for x in r.x:
        xi = x[IFED] - x[IA]                    # moles reacted
        assert abs(x[IP] - 2 * xi) < 1e-9
        assert abs((P.n_W0 - x[IW]) - xi) < 1e-9
    assert abs(r.x[-1, IFED] - P.n_A_total) < 1e-9   # exactly the planned charge was dosed


def test_energy_balance_closes_with_cooling():
    # Heat released by reaction = heat removed by jacket + heat to warm the feed + stored heat
    p = replace(P, dt_control=1.0)
    r = simulate(FixedTiming(8.0), p, stop_at_completion=False)
    t, X = r.t, r.x
    T = X[:, IT]
    Q_cool = p.UA * (T - p.T_jacket)
    E_cool = np.trapezoid(Q_cool, t) if hasattr(np, "trapezoid") else np.trapz(Q_cool, t)
    F_mol = np.append(r.F, 0.0) / 60 * p.mol_per_mL_A
    feed_term = F_mol * p.Cp_A * (T - p.T_feed)
    E_feed = np.trapezoid(feed_term, t) if hasattr(np, "trapezoid") else np.trapz(feed_term, t)
    reacted = X[-1, IP] / 2
    released = -p.dH_r * reacted
    # Identity for this model:  released = E_cool + E_feed + integral(C dT)
    CdT = np.sum([0.5 * (heat_capacity(X[i], p) + heat_capacity(X[i + 1], p)) * (T[i + 1] - T[i])
                  for i in range(len(T) - 1)])
    rel_err = abs(released - (E_cool + E_feed + CdT)) / released
    assert rel_err < 2e-3, rel_err


def test_rk4_matches_scipy_lsoda_closed_loop():
    c = PIDDosing(48.0, 1.0, 1200.0, F_max=P.F_max_mL_min, dt=P.dt_control)
    a = simulate(c, P, method="rk4")
    b = simulate(c, P, method="scipy")
    assert abs(a.T_peak_C - b.T_peak_C) < 1e-3, (a.T_peak_C, b.T_peak_C)
    assert a.batch_time_s == b.batch_time_s


def test_mtsr_and_volume_sanity():
    x = initial_state(P)
    assert abs(volume_L(x, P) - 0.4) < 1e-9
    assert abs(mtsr(x, P) - x[IT]) < 1e-12            # nothing accumulated -> MTSR = T
    x2 = x.copy(); x2[IA] = 3.0
    assert mtsr(x2, P) - x2[IT] > 60                  # full charge accumulated -> big rise


def test_full_pump_speed_is_unsafe():
    # The control problem must be non-trivial: dumping at max pump speed overheats.
    r = simulate(FixedTiming(P.F_max_mL_min), P)
    assert r.violated and r.T_peak_C > 60


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in tests:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\nAll {len(tests)} physics tests passed.")
