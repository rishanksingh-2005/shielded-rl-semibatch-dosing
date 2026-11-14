"""
Step 7 physics checks for the second reactor (r2_model.py).   python r2_test_model.py

1. Mass balance: A fed = A left + C made; B left + C made = B charged; volume = V0 + fed/c_in.
2. Energy balance closes: change in stored heat = feed heat + reaction heat - jacket heat (and the
   jacket's own balance), integrated with LSODA at tight tolerance.
3. Our RK4 (2 s step) agrees with SciPy LSODA on a full batch (peak T within 0.05 K).
4. Adiabatic test: with no cooling and no feed, T rises by exactly the analytical amount.
5. The MDC formula equals the original criterion of Kummer & Varga (Fricz Eq. 7), divided by k*beta.
6. Reproduces Fricz 2026 Fig. 4 when driven by the valve profile read from that figure:
   start-up dip to about 325 K and peak MDC about 0.075 (Fricz Table 4, NMPC: 0.0772).
7. Full valve from the start overheats past MAT = 373 K, so the control problem is real.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from r2_model import (IA, IB, IC, IFED, IT, ITJ, IV, R2Params, UA, adiabatic_rise, advance,
                      feeding_done, initial_state, max_OP_for_remaining, mdc, rate, rhs)

P = R2Params()


def op_fig4(k: int) -> float:
    """Feed valve profile read from Fricz 2026 Fig. 4 (TD3, lambda = 0), one value per 50 s."""
    t = k * 50.0
    if t < 950: return min(100.0, 5.0 + 5.0 * k)
    if t < 1150: return 100.0
    if t < 2150: return max(0.0, 100.0 - 5.0 * (k - 23))
    if t < 2250: return 0.0
    if t < 2900: return min(65.0, 5.0 * (k - 45))
    if t < 6500: return 65.0 + 35.0 * (t - 2900) / 3600
    return 100.0


def run(profile, p=P, method="rk4", n_max=400):
    s = initial_state(p)
    traj = [s]
    for k in range(n_max):
        if feeding_done(s, p):
            break
        op = min(profile(k), max_OP_for_remaining(s, p))
        if method == "rk4":
            s = advance(s, op, p)
        else:
            F, FJ = p.F_max * op / 100, p.FJ_max * p.OPJ / 100
            sol = solve_ivp(lambda t, y: rhs(tuple(y), F, FJ, p), (0, p.dt_control), s,
                            method="LSODA", rtol=1e-10, atol=1e-12)
            s = tuple(sol.y[:, -1])
        traj.append(s)
    return traj


def test_mass_balance():
    traj = run(op_fig4)
    worst = 0.0
    for s in traj:
        worst = max(worst, abs(s[IA] + s[IC] - s[IFED]), abs(s[IB] + s[IC] - P.nB0),
                    abs(s[IV] - P.V0 - s[IFED] / P.c_in))
    assert worst < 1e-9, worst
    return f"worst error {worst:.1e} kmol / m3"


def test_energy_balance():
    # augmented state: [model state..., integral of feed heat, reaction heat, jacket heat, coolant heat]
    def f(t, y, F, FJ):
        s = tuple(y[:7])
        d = rhs(s, F, FJ, P)
        q_feed = F * P.rho_cp * P.T_in                       # enthalpy carried in by the feed (ref 0 K)
        q_rxn = -P.dH * s[IV] * rate(s, P)
        q_cool = UA(s, P) * (s[IT] - s[ITJ])
        q_jin = FJ * P.rho_cpJ * (P.TJ_in - s[ITJ])
        return list(d) + [q_feed, q_rxn, q_cool, q_jin]
    y = list(initial_state(P)) + [0, 0, 0, 0]
    y0 = list(y)
    for k in range(200):
        op = min(op_fig4(k), max_OP_for_remaining(tuple(y[:7]), P))
        F, FJ = P.F_max * op / 100, P.FJ_max * P.OPJ / 100
        y = solve_ivp(f, (0, P.dt_control), y, args=(F, FJ), method="LSODA", rtol=1e-10, atol=1e-10).y[:, -1]
    H = lambda v: v[IV] * P.rho_cp * v[IT]
    reactor = (H(y) - H(y0)) - (y[7] + y[8] - y[9])
    HJ = lambda v: P.VJ * P.rho_cpJ * v[ITJ]
    jacket = (HJ(y) - HJ(y0)) - (y[9] + y[10])
    rel = max(abs(reactor) / y[8], abs(jacket) / y[8])
    assert rel < 1e-6, rel
    return f"reactor and jacket balances close to {rel:.1e} of the heat released ({y[8]/1e3:.0f} MJ)"


def test_rk4_vs_lsoda():
    a = run(op_fig4, n_max=120)
    b = run(op_fig4, method="lsoda", n_max=120)
    dT = max(abs(x[IT] - y[IT]) for x, y in zip(a, b))
    dpk = abs(max(x[IT] for x in a) - max(y[IT] for y in b))
    assert dT < 0.05 and dpk < 0.05, (dT, dpk)
    return f"max |T difference| {dT:.4f} K, peak difference {dpk:.4f} K over 6,000 s"


def test_adiabatic():
    p = R2Params(UA0=0.0)
    s = (1.0, 1.0, 3.0, 2.0, 330.0, 298.0, 0.0)      # 1 kmol A accumulated, 3 kmol B
    rise = adiabatic_rise(s, p)
    for _ in range(400):
        s = advance(s, 0.0, p, dt=50.0)
    err = abs(s[IT] - (330.0 + rise))
    assert err < 1e-6 and s[IA] < 1e-6, (err, s[IA])
    return f"T rose {rise:.2f} K as predicted (error {err:.1e} K)"


def test_mdc_definition():
    """Fricz Eq. 7: dq_gen/dT + dm/dc <= (q_gen/q_rem) dq_rem/dT, checked by finite differences."""
    worst = 0.0
    for s in run(op_fig4, n_max=120)[5::7]:
        V, T, TJ = s[IV], s[IT], s[ITJ]
        beta = -P.dH / P.rho_cp
        cA, cB = s[IA] / V, s[IB] / V
        k = lambda T_: P.k0 * math.exp(-P.E_R / T_)
        qgen = lambda T_, a, b: beta * k(T_) * a * b
        qrem = lambda T_: UA(s, P) * (T_ - TJ) / (V * P.rho_cp)
        e = 1e-5
        dqgen = (qgen(T + e, cA, cB) - qgen(T - e, cA, cB)) / (2 * e)
        dm = -(k(T) * (cA + e) * cB - k(T) * (cA - e) * cB + k(T) * cA * (cB + e) - k(T) * cA * (cB - e)) / (2 * e)
        dqrem = (qrem(T + e) - qrem(T - e)) / (2 * e)
        lhs_minus_rhs = dqgen + dm - qgen(T, cA, cB) / qrem(T) * dqrem
        worst = max(worst, abs(lhs_minus_rhs / (k(T) * beta) - mdc(s, P)))
    assert worst < 1e-6, worst
    return f"matches the criterion to {worst:.1e}"


def test_fig4_reproduction():
    traj = run(op_fig4)
    T = np.array([s[IT] for s in traj]); M = np.array([mdc(s, P) for s in traj])
    t_min = 50 * int(T.argmin())
    assert 322 < T.min() < 330 and 700 <= t_min <= 1200, (T.min(), t_min)
    assert 0.065 < M.max() < 0.09, M.max()
    return (f"dip to {T.min():.1f} K at {t_min} s (Fig. 4: ~325 K at ~1,050 s); "
            f"peak MDC {M.max():.4f} (Fricz: ~0.075, NMPC 0.0772)")


def test_full_valve_overheats():
    traj = run(lambda k: 100.0)
    pk = max(s[IT] for s in traj)
    assert pk > P.MAT + 5, pk
    return f"valve at 100 % from the start: peak {pk:.1f} K > MAT {P.MAT:.0f} K"


if __name__ == "__main__":
    tests = [test_mass_balance, test_energy_balance, test_rk4_vs_lsoda, test_adiabatic,
             test_mdc_definition, test_fig4_reproduction, test_full_valve_overheats]
    for i, t in enumerate(tests, 1):
        print(f"{i}. {t.__name__[5:]:24s} PASS  {t()}")
    print(f"All {len(tests)} reactor-2 physics tests passed.")
