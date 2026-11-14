"""
Step 3 experiment: tune baselines on the nominal model, then test them on 'true plants'
with model mismatch. Produces results/ (CSV, JSON, figures).

    python run_baselines.py            # full run (27 mismatched plants)
"""
from __future__ import annotations

import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers import PIDDosing, FixedTiming, tune_fixed_timing, tune_pid
from reactor_model import IA, ReactorParams, simulate
from shield import Shield, max_feed_with_shield, robust_models

OUT = "results"
LEVELS = (0.8, 1.0, 1.2)        # +/-20 % mismatch, as in the project plan


def summarize(r):
    return dict(batch_min=None if r.batch_time_s is None else r.batch_time_s / 60,
                T_peak_C=r.T_peak_C, violated=bool(r.violated),
                max_mtsr_C=float(r.mtsr_C.max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    p = ReactorParams()
    t0 = time.time()

    # ---------- tune on nominal model (zero back-off: most favourable to baselines) ----------
    fixed, t_dose = tune_fixed_timing(p, backoff=0.0)
    pid, _ = tune_pid(p, backoff=0.0, verbose=True)
    # robust versions: limit must hold on the worst-case corners as well (same set as robust shield)
    corners = robust_models(p)
    fixed_r, t_dose_r = tune_fixed_timing(p, backoff=0.0, check_plants=corners)
    pid_r, _ = tune_pid(p, backoff=0.0, verbose=True, check_plants=corners)
    print(f"robust fixed dose time {t_dose_r:.2f} min")
    print(f"tuned in {time.time()-t0:.0f}s; fixed dose time {t_dose:.2f} min (F={fixed.F_mL_min:.2f} mL/min)")

    controllers = {
        "Fixed timing": lambda plant: fixed,
        "PID": lambda plant: pid,
        "Fixed timing (robust tuned)": lambda plant: fixed_r,
        "PID (robust tuned)": lambda plant: pid_r,
        "Max feed + nominal shield": lambda plant: max_feed_with_shield(Shield([p])),
        "Max feed + robust shield": lambda plant: max_feed_with_shield(Shield(robust_models(p))),
        "Oracle (shield knows true plant)": lambda plant: max_feed_with_shield(Shield([plant])),
    }

    # ---------- nominal runs + figure ----------
    nominal = {}
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for name, make in controllers.items():
        if name.startswith("Oracle"):
            continue
        r = simulate(make(p), p)
        nominal[name] = summarize(r)
        tm = r.t / 60
        ax[0].plot(tm, r.T_C, label=name)
        ax[1].step(tm[:-1], r.F, where="post", label=name)
        ax[2].plot(tm, r.x[:, IA], label=name)
    ax[0].axhline(p.T_max - 273.15, color="k", ls="--", lw=1, label="T limit 50 C")
    ax[0].set_ylabel("Reactor T (C)"); ax[1].set_ylabel("Feed (mL/min)")
    ax[2].set_ylabel("Unreacted anhydride (mol)"); ax[2].set_xlabel("Time (min)")
    ax[0].legend(fontsize=8); ax[0].set_title("Nominal model: baseline controllers")
    fig.tight_layout(); fig.savefig(f"{OUT}/nominal_trajectories.png", dpi=150); plt.close(fig)
    print("nominal:", json.dumps(nominal, indent=1))

    # ---------- mismatch sweep ----------
    rows = []
    for kf, hf, uf in itertools.product(LEVELS, LEVELS, LEVELS):
        plant = p.with_mismatch(k_factor=kf, dH_factor=hf, UA_factor=uf)
        for name, make in controllers.items():
            r = simulate(make(plant), plant)
            rows.append(dict(controller=name, k_factor=kf, dH_factor=hf, UA_factor=uf, **summarize(r)))
        print(f"plant k={kf} dH={hf} UA={uf} done ({time.time()-t0:.0f}s)", flush=True)

    import csv
    with open(f"{OUT}/mismatch_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # time relative to oracle on the same plant
    oracle = {(r["k_factor"], r["dH_factor"], r["UA_factor"]): r["batch_min"]
              for r in rows if r["controller"].startswith("Oracle")}
    summary = {}
    for name in controllers:
        rs = [r for r in rows if r["controller"] == name]
        slow = [r["batch_min"] / oracle[(r["k_factor"], r["dH_factor"], r["UA_factor"])] - 1
                for r in rs if r["batch_min"] is not None]
        summary[name] = dict(
            plants=len(rs),
            plants_overheated=sum(r["violated"] for r in rs),
            worst_peak_C=max(r["T_peak_C"] for r in rs),
            mean_batch_min=float(np.mean([r["batch_min"] for r in rs if r["batch_min"]])),
            mean_slower_than_oracle_pct=100 * float(np.mean(slow)),
            unfinished=sum(r["batch_min"] is None for r in rs))
    out = dict(nominal=nominal, mismatch_summary=summary,
               tuned=dict(fixed_F_mL_min=fixed.F_mL_min, fixed_dose_min=t_dose,
                          pid=dict(T_sp_C=pid.T_sp_C, Kp=pid.Kp, Ti=pid.Ti, Td=pid.Td),
                          fixed_robust_F_mL_min=fixed_r.F_mL_min, fixed_robust_dose_min=t_dose_r,
                          pid_robust=dict(T_sp_C=pid_r.T_sp_C, Kp=pid_r.Kp, Ti=pid_r.Ti, Td=pid_r.Td)))
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=1))

    # ---------- figure: peak temperature across plants ----------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = list(controllers)
    for i, name in enumerate(names):
        peaks = [r["T_peak_C"] for r in rows if r["controller"] == name]
        ax.scatter(np.full(len(peaks), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(peaks)),
                   peaks, s=14, alpha=0.7)
    ax.axhline(p.T_max - 273.15, color="r", ls="--", lw=1, label="T limit 50 C")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("Peak reactor T (C)"); ax.legend()
    ax.set_title("Peak temperature on 27 mismatched plants (k, dH, UA each x0.8 / x1.0 / x1.2)")
    fig.tight_layout(); fig.savefig(f"{OUT}/mismatch_peak_temperatures.png", dpi=150); plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
