"""
Step 7 baselines for the second reactor (no AI).        python r2_baselines.py   (~2-4 min)

    PI (nominal tuned)   velocity-form PI on the feed valve, +/-5 % per 50 s (same limits as the AI);
                         gains chosen for the lowest IAE on the nominal reactor without passing MAT
    PI (robust tuned)    gains chosen for the lowest mean IAE on the 27 grid reactors with NONE passing MAT
                         (first run, 12 Nov 2026: NO gain pair qualified for either, see below)
    PI (best tracking)   lowest nominal IAE among gains that finish feeding, safe or not
    PI (best tracking) + robust brake   the same PI behind the careful brake
    Max feed + robust brake   always ask for +5 % (up to 100 %), the careful brake cuts it (no AI)
    Oracle brake         the same, but the brake knows the true reactor: fastest safe feeding

Overheating counts the hour after feeding too (see r2_env.py). Output: results/r2/summary.json
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from r2_env import R2DosingEnv
from r2_model import R2Params
from r2_shield import R2Shield

OUT = os.path.join("results", "r2")
P = R2Params()
GRID = [dict(k_factor=a, dH_factor=b, UA_factor=c) for a, b, c in itertools.product((0.8, 1.0, 1.2), repeat=3)]
NOMINAL = dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0)


class PI:
    """Velocity-form PI: dOP = Kp (e - e_prev) + Kp dt/Ti e, sent as an action in [-1, 1] (= +/-5 %)."""

    def __init__(self, Kp: float, Ti: float):
        self.Kp, self.Ti = Kp, Ti

    def __call__(self, obs, env):
        e, e_prev = obs[6] * 50.0, obs[7] * 50.0
        d = self.Kp * (e - e_prev) + self.Kp * env.p_nom.dt_control / self.Ti * e
        return np.array([np.clip(d / env.max_step_OP, -1.0, 1.0)], dtype=np.float32)


def max_feed(obs, env):
    return np.array([1.0], dtype=np.float32)


def run(policy, factors: dict, safety="none", shield_type="robust", oracle=False) -> dict:
    env = R2DosingEnv(safety=safety, shield_type=shield_type)
    obs, _ = env.reset(options={"plant_factors": factors})
    if oracle:
        env.shield = R2Shield([env.plant])
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        ret += r
        done = term or trunc
    return summarize(info, ret)


def summarize(info: dict, ret: float) -> dict:
    return dict(return_=ret, feed_min=info["batch_time_min"], overheated=bool(info["violated"]),
                T_peak_K=info["T_peak_K"], tail_peak_K=info["tail_peak_K"],
                time_to_sp_s=info["time_to_sp_s"], iae=info["iae"], mdc_max=info["mdc_max"],
                mdc_pos_s=info["mdc_pos_s"], mtsr_peak_K=info["mtsr_peak_C"] + 273.15,
                feed_tv=info["feed_tv"], interventions=info["n_interventions"])


def tune_pi():
    Kps = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    Tis = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
    rows = []
    for Kp, Ti in itertools.product(Kps, Tis):
        nom = run(PI(Kp, Ti), NOMINAL)
        grid = [run(PI(Kp, Ti), f) for f in GRID]
        rows.append(dict(Kp=Kp, Ti=Ti, nom=nom, grid_over=sum(g["overheated"] for g in grid),
                         grid_unfinished=sum(g["feed_min"] is None for g in grid),
                         grid_iae=float(np.mean([g["iae"] for g in grid]))))
        print(f"  PI Kp={Kp:<5} Ti={Ti:<6.0f} nominal: peak {nom['T_peak_K']:6.1f} K, IAE {nom['iae']:8.0f} | "
              f"grid: overheated {rows[-1]['grid_over']:2d}/27, unfinished {rows[-1]['grid_unfinished']:2d}, "
              f"mean IAE {rows[-1]['grid_iae']:8.0f}", flush=True)
    ok_nom = [r for r in rows if not r["nom"]["overheated"] and r["nom"]["feed_min"] is not None]
    ok_rob = [r for r in rows if r["grid_over"] == 0 and r["grid_unfinished"] == 0]
    best_nom = min(ok_nom, key=lambda r: r["nom"]["iae"]) if ok_nom else None
    best_rob = min(ok_rob, key=lambda r: r["grid_iae"]) if ok_rob else None
    return rows, best_nom, best_rob


def table(name: str, res: list):
    done = [r for r in res if r["feed_min"] is not None]
    reach = [r["time_to_sp_s"] for r in res if r["time_to_sp_s"] is not None]
    print(f"  {name:28s} overheated {sum(r['overheated'] for r in res):2d}/{len(res)} | "
          f"worst peak {max(r['T_peak_K'] for r in res):6.1f} K | unfinished {len(res) - len(done):2d} | "
          f"mean feed {np.mean([r['feed_min'] for r in done]) if done else float('nan'):6.1f} min | "
          f"mean time to SP {np.mean(reach) if reach else float('nan'):6.0f} s ({len(reach)} reached) | "
          f"mean IAE {np.mean([r['iae'] for r in res]):8.0f} | worst MTSR {max(r['mtsr_peak_K'] for r in res):6.1f} K")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("Tuning PI on the feed valve (30 gain pairs x 28 reactors) ...")
    rows, best_nom, best_rob = tune_pi()
    methods = {}
    if best_nom:
        methods["PI (nominal tuned)"] = dict(policy=PI(best_nom["Kp"], best_nom["Ti"]))
    if best_rob:
        methods["PI (robust tuned)"] = dict(policy=PI(best_rob["Kp"], best_rob["Ti"]))
    # Best tracking PI regardless of safety (lowest nominal IAE among gains that finish feeding),
    # alone and behind the careful brake: the classic controller + our brake, no AI.
    finished = [r for r in rows if r["nom"]["feed_min"] is not None]
    best_trk = min(finished, key=lambda r: r["nom"]["iae"])
    pi_trk = PI(best_trk["Kp"], best_trk["Ti"])
    methods["PI (best tracking)"] = dict(policy=pi_trk)
    methods["PI (best tracking) + robust brake"] = dict(policy=pi_trk, safety="shield", shield_type="robust")
    methods["Max feed + nominal brake"] = dict(policy=max_feed, safety="shield", shield_type="nominal")
    methods["Max feed + robust brake"] = dict(policy=max_feed, safety="shield", shield_type="robust")
    methods["Oracle brake"] = dict(policy=max_feed, safety="shield", oracle=True)

    results = {}
    for group, plants in (("nominal", [NOMINAL]), ("grid", GRID)):
        print(f"\n{group} ({len(plants)} reactor{'s' if len(plants) > 1 else ''}):")
        for name, m in methods.items():
            kw = {k: v for k, v in m.items() if k != "policy"}
            res = [run(m["policy"], f, **kw) for f in plants]
            results.setdefault(name, {})[group] = res
            table(name, res)
    summary = dict(
        reactor="Fricz et al. 2026 benchmark (A + B -> C)", setpoint_K=P.T_sp, MAT_K=P.MAT,
        tuned=dict(pi_nominal=None if not best_nom else dict(Kp=best_nom["Kp"], Ti=best_nom["Ti"]),
                   pi_robust=None if not best_rob else dict(Kp=best_rob["Kp"], Ti=best_rob["Ti"]),
                   pi_best_tracking=dict(Kp=best_trk["Kp"], Ti=best_trk["Ti"])),
        pi_tuning=[dict(Kp=r["Kp"], Ti=r["Ti"], nominal_peak_K=r["nom"]["T_peak_K"],
                        nominal_overheated=r["nom"]["overheated"], nominal_iae=r["nom"]["iae"],
                        grid_overheated=r["grid_over"], grid_unfinished=r["grid_unfinished"],
                        grid_mean_iae=r["grid_iae"]) for r in rows],
        results=results)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    if not best_rob:
        print("\nNOTE: no PI gain pair in the grid was safe on all 27 reactors.")
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
