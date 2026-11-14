"""
Step 7o: the delay-aware brake (r2_shield.py, cf_delay_steps) on reactor 2.

The normal brake assumes the feed stops the instant the cooling fails. The cooling-failure test lets the
controller run 100 s more before the feed is shut, and controllers that run close to the limit then
passed MAT by up to 1.2 K (Steps 7d/7m). The delay-aware brake also checks: "if the cooling failed
at the start of this step and the feed stayed on for 100 s, would T stay <= MAT?" (worst-case corners).

This script re-tests the main methods with the delay-aware brake swapped in at test time. The agents
are NOT retrained (the brake is a runtime layer); the coolant PIs keep their Step 7d tuning.

    python r2_cf_aware.py                     all methods (5 seeds per agent), 127 test reactors + 12 cooling failures
    python r2_cf_aware.py --quick             no agents; nominal + 27 grid reactors only
    --variant nominal (default) | robust      models used for the extra cooling-failure check
Output: results/r2/cf_aware_<variant>[_quick]/ (agent_eval.csv, cooling_failure.csv, summary.json)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from r2_baselines import max_feed
from r2_coolant import CoolantPI, episode
from r2_env import R2DosingEnv
from r2_evaluate import test_plants
from r2_train import ALL_CONFIGS, LAMBDA, OUT as AGENTS

OUT = os.path.join("results", "r2", "cf_aware")
AGENT_CONFIGS = ("B_robust_pen1", "B_coolF_B", "B_coolF_A")


def classic(name, chosen):
    """name -> (policy, env kwargs)"""
    if name == "Max feed + brake (one valve)":
        return max_feed, dict(safety="shield", shield_type="robust")
    rule = name[-1]
    c = chosen[rule]
    return CoolantPI(c["Kp"], c["Ti"], 100.0), dict(safety="shield", shield_type="robust", coolant_control=True,
                                                    backup_OPJ=100.0 if rule == "A" else "hold")


CLASSIC = ("Max feed + brake (one valve)", "Coolant PI + max feed + brake A", "Coolant PI + max feed + brake B")


def evaluate(spec, chosen, plants, cf_times, variant):
    import torch
    torch.set_num_threads(1)
    if spec[0] == "classic":
        name, seed = spec[1], ""
        pol, kw = classic(name, chosen)
    else:
        from stable_baselines3 import TD3
        _, name, seed, path = spec
        model = TD3.load(path, device="cpu")
        pol = lambda obs, env: model.predict(obs, deterministic=True)[0]
        kw = {k: v for k, v in ALL_CONFIGS[name].items() if k not in ("randomize_plant", "finish_reward")}
    env = R2DosingEnv(lam=LAMBDA, cf_aware=variant, **kw)
    rows = [dict(method=name, seed=seed, reactor=i, group=g, **f, **episode(pol, env, f))
            for i, (g, f) in enumerate(plants)]
    cf = []
    for t in cf_times:
        r = episode(pol, env, plants[0][1], cooling_failure_s=t)
        cf.append(dict(method=name, seed=seed, failure_at_s=t, T_peak_K=r["T_peak_K"], overheated=r["overheated"]))
    return rows, cf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parallel", type=int, default=0, help="runs at the same time (default: half the CPUs, max 6)")
    ap.add_argument("--variant", choices=("nominal", "robust"), default="nominal")
    a = ap.parse_args()
    out = f"{OUT}_{a.variant}" + ("_quick" if a.quick else "")
    os.makedirs(out, exist_ok=True)
    n_par = a.parallel or max(1, min(6, (os.cpu_count() or 2) // 2))
    chosen = json.load(open(os.path.join("results", "r2", "coolant", "tuning.json")))["chosen"]
    plants = test_plants(0) if a.quick else test_plants(100)
    cf_times = [600 * k for k in range(1, 13)]
    specs = [("classic", n) for n in CLASSIC]
    if not a.quick:
        for path in sorted(glob.glob(os.path.join(AGENTS, "*.zip"))):
            m = re.match(r"(.+)_seed(\d+)\.zip$", os.path.basename(path))
            if m and m.group(1) in AGENT_CONFIGS:
                specs.append(("agent", m.group(1), int(m.group(2)), path))
    t0 = time.time()
    print(f"Testing {len(specs)} methods with the delay-aware brake on {len(plants)} reactors + "
          f"{len(cf_times)} cooling failures, {n_par} at a time ...", flush=True)
    rows, cf_rows = [], []
    with ProcessPoolExecutor(max_workers=n_par) as pool:
        futs = {pool.submit(evaluate, s, chosen, plants, cf_times, a.variant): s for s in specs}
        for f in as_completed(futs):
            r, c = f.result(); rows += r; cf_rows += c
            s = futs[f]
            print(f"  {s[1] + ('' if s[0] == 'classic' else f' seed {s[2]}'):42s} done ({time.time() - t0:.0f} s)",
                  flush=True)
    order = list(CLASSIC) + list(AGENT_CONFIGS)
    rows.sort(key=lambda r: (order.index(r["method"]), str(r["seed"]), r["reactor"]))
    cf_rows.sort(key=lambda r: (order.index(r["method"]), str(r["seed"]), r["failure_at_s"]))
    for name, data in (("agent_eval.csv", rows), ("cooling_failure.csv", cf_rows)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    print(f"\n{'Method (delay-aware brake)':34s} {'seeds':>5s} {'overheated':>12s} {'unfinished':>12s} {'mean IAE':>9s} "
          f"{'median feed':>11s} {'cooling failure':>16s} {'worst CF peak':>13s}")
    summary = {}
    for m in [x for x in order if any(r["method"] == x for r in rows)]:
        t = [r for r in rows if r["method"] == m and r["group"] != "nominal"]
        c = [r for r in cf_rows if r["method"] == m]
        fin = [r["feed_min"] for r in t if r["feed_min"] is not None]
        summary[m] = s = dict(seeds=len({r["seed"] for r in t}), test=len(t), overheated=sum(r["overheated"] for r in t),
                              unfinished=sum(r["feed_min"] is None for r in t), mean_iae=float(np.mean([r["iae"] for r in t])),
                              median_feed_min=float(np.median(fin)) if fin else None,
                              worst_peak_K=max(r["T_peak_K"] for r in t), cf_tests=len(c),
                              cf_overheated=sum(r["overheated"] for r in c), cf_worst_peak_K=max(r["T_peak_K"] for r in c))
        print(f"{m:34s} {s['seeds']:>5d} {s['overheated']:>5d}/{s['test']:<6d} {s['unfinished']:>5d}/{s['test']:<6d} "
              f"{s['mean_iae']:>9.0f} {str(round(s['median_feed_min'])) if fin else '-':>11s} "
              f"{s['cf_overheated']:>7d}/{s['cf_tests']:<8d} {s['cf_worst_peak_K']:>13.2f}")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Files in {out}")


if __name__ == "__main__":
    main()
