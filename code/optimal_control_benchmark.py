"""
Independent optimal-control benchmark (nominal model, perfect knowledge).

Direct single shooting: for a fixed batch time tf, choose a piecewise-constant feed profile
that maximizes conversion subject to T <= T_max (checked every 30 s) and the planned
charge. The smallest tf that reaches 99 % conversion is the minimum batch time.

Result obtained (2026-10-24): tf = 31 min -> 98.68 %, tf = 32 min -> 99.06 %,
so the minimum batch time is about 31.8 min.

    python optimal_control_benchmark.py      # ~5 min
"""
import time

import numpy as np
from scipy.optimize import minimize

from reactor_model import IT, ReactorParams, advance, conversion, initial_state

p = ReactorParams()


def rollout(u, tf):
    N = len(u)
    seg = tf / N
    steps = int(round(seg / p.dt_control))
    assert abs(steps * p.dt_control - seg) < 1e-9, "segment length must be a multiple of dt"
    x = initial_state(p)
    Ts = []
    for ui in u:
        for _ in range(steps):
            x = advance(x, ui, p, p.dt_control, substeps=4)
            Ts.append(x[IT])
    return x, np.array(Ts)


def solve(tf, N):
    Vtot = p.V_A_total_mL
    seg_min = tf / N / 60
    u0 = np.clip(np.full(N, Vtot / (tf / 60 * 0.8)), 0, p.F_max_mL_min)
    cons = [{"type": "ineq", "fun": lambda u: (p.T_max - rollout(u, tf)[1])[::3]},
            {"type": "ineq", "fun": lambda u: Vtot - np.sum(u) * seg_min}]
    res = minimize(lambda u: -conversion(rollout(u, tf)[0], p), u0, method="SLSQP",
                   bounds=[(0, p.F_max_mL_min)] * N, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-7})
    x, T = rollout(res.x, tf)
    return conversion(x, p), T.max() - 273.15, res


if __name__ == "__main__":
    t0 = time.time()
    for tf_min, N in [(28, 24), (30, 18), (31, 31), (32, 24)]:
        c, Tp, res = solve(tf_min * 60.0, N)
        print(f"tf={tf_min} min: max conversion {c:.4f}, peak {Tp:.2f} C, "
              f"success={res.success} ({time.time()-t0:.0f}s)", flush=True)
