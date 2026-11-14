"""
Step 7 evaluation: every trained AI on the second reactor against the no-AI methods.

Test reactors (seed 777 for the random ones, never used in training):
    nominal (1), grid: k, dH, UA each x0.8 / x1.0 / x1.2 (27), random: uniform +/-20 % (100)
Cooling-failure test (Fricz Sec. 3.2), nominal reactor: the coolant fails at t = 10, 20, ... 120 min;
the driver keeps going for 100 s, then the feed is shut; peak temperature with no cooling.

Measures: overheated (T > MAT = 373 K, including 1 h after feeding), worst peak, feeding time,
time to reach the setpoint, IAE (integral |T - 363 K| dt, K s), Fricz's MDC measures
(max MDC, seconds with MDC > 0), worst MTSR on the true reactor, valve movement.

    python r2_evaluate.py              # all trained AIs (~30-60 min; methods run in parallel)
    python r2_evaluate.py --quick      # include *_quick models, 20 random reactors, 4 failure times

Output: results/r2/agent_eval.csv, agent_eval_summary.json, cooling_failure.csv, agent_eval_peaks.png,
        agent_training.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r2_baselines import PI, max_feed
from r2_env import R2DosingEnv
from r2_shield import R2Shield
from r2_train import CONFIGS, LAMBDA, OUT as AGENTS

OUT = os.path.join("results", "r2")
TEST_SEED = 777


def test_plants(n_random: int):
    plants = [("nominal", dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0))]
    for kf, hf, uf in itertools.product((0.8, 1.0, 1.2), repeat=3):
        plants.append(("grid", dict(k_factor=kf, dH_factor=hf, UA_factor=uf)))
    rng = np.random.default_rng(TEST_SEED)
    for _ in range(n_random):
        f = rng.uniform(0.8, 1.2, 3)
        plants.append(("random", dict(k_factor=f[0], dH_factor=f[1], UA_factor=f[2])))
    return plants


def episode(policy, env: R2DosingEnv, factors: dict, oracle=False, cooling_failure_s=None) -> dict:
    opts = {"plant_factors": factors}
    if cooling_failure_s is not None:
        opts["cooling_failure_s"] = cooling_failure_s
    obs, _ = env.reset(options=opts)
    if oracle:
        env.shield = R2Shield([env.plant])
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
    return dict(feed_min=info["batch_time_min"], overheated=bool(info["violated"]), T_peak_K=info["T_peak_K"],
                time_to_sp_s=info["time_to_sp_s"], iae=info["iae"], mdc_max=info["mdc_max"],
                mdc_pos_s=info["mdc_pos_s"], mtsr_peak_K=info["mtsr_peak_C"] + 273.15,
                feed_tv=info["feed_tv"], interventions=info["n_interventions"])


def classic_methods():
    tuned = json.load(open(os.path.join(OUT, "summary.json")))["tuned"]["pi_best_tracking"]
    pi = PI(tuned["Kp"], tuned["Ti"])
    return {  # name: (policy, env kwargs, oracle)
        "PI (best tracking)": (pi, dict(safety="none"), False),
        "PI + robust brake": (pi, dict(safety="shield", shield_type="robust"), False),
        "Max feed + robust brake (no AI)": (max_feed, dict(safety="shield", shield_type="robust"), False),
        "Oracle brake": (max_feed, dict(safety="shield"), True),
    }


def find_agents(include_quick: bool):
    out = []
    for path in sorted(glob.glob(os.path.join(AGENTS, "*.zip"))):
        m = re.match(r"(.+)_seed(\d+)(_quick)?\.zip$", os.path.basename(path))
        if m and m.group(1) in CONFIGS and (include_quick or not m.group(3)):
            out.append((m.group(1), int(m.group(2)), bool(m.group(3)), path))
    return out


def agent_policy(model):
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def evaluate_method(spec, plants, cf_times):
    """Run one method (or one trained AI) on every test reactor. Runs in its own process."""
    import torch
    torch.set_num_threads(1)
    kind, name = spec[0], spec[1]
    if kind == "classic":
        pol, kw, orc = classic_methods()[name]
        config, seed = name, ""
    else:
        from stable_baselines3 import TD3
        _, label, config_key, seed, path = spec
        pol, orc, config = agent_policy(TD3.load(path, device="cpu")), False, label
        kw = {k: v for k, v in CONFIGS[config_key].items() if k != "randomize_plant"}
    env = R2DosingEnv(lam=LAMBDA, **kw)
    rows, cf_rows = [], []
    for i, (group, f) in enumerate(plants):
        rows.append(dict(method=config, seed=seed, reactor=i, group=group, **f, **episode(pol, env, f, orc)))
    for t_cf in cf_times:
        r = episode(pol, env, plants[0][1], orc, cooling_failure_s=t_cf)
        cf_rows.append(dict(method=config, seed=seed, failure_at_s=t_cf, T_peak_K=r["T_peak_K"],
                            overheated=r["overheated"]))
    return rows, cf_rows


def fmt(vals, f="{:.1f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f.format(np.mean(vals)) if len(vals) == 1 or np.ptp(vals) == 0 else \
        f"{f.format(np.mean(vals))} [{f.format(min(vals))}-{f.format(max(vals))}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=0,
                    help="methods tested at the same time (1 CPU core each). Default: half the CPUs, at most 6")
    a = ap.parse_args()

    plants = test_plants(20 if a.quick else a.n_random)
    cf_times = [600 * k for k in ((2, 5, 8, 11) if a.quick else range(1, 13))]
    specs = [("classic", name) for name in classic_methods()]
    for config, seed, quick, path in find_agents(a.quick):
        specs.append(("agent", config + ("_quick" if quick else ""), config, seed, path))
    if not any(s[0] == "agent" for s in specs):
        print("No trained AIs found in results/r2/agents/. Run r2_train.py first (baselines are still evaluated).")
    # slowest first (methods with the careful brake), so the parallel pool finishes sooner
    specs.sort(key=lambda s: 0 if ("robust" in s[1] or "B_" in s[1]) else 1)

    t0, rows, cf_rows = time.time(), [], []
    n_par = a.parallel or max(1, min(6, (os.cpu_count() or 2) // 2))
    print(f"Testing {len(specs)} methods on {len(plants)} reactors + {len(cf_times)} cooling failures, "
          f"{n_par} at a time ...", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_par) as pool:
        futs = {pool.submit(evaluate_method, s, plants, cf_times): s for s in specs}
        for f in as_completed(futs):
            r, c = f.result()
            rows += r; cf_rows += c
            s = futs[f]
            print(f"  {s[1] + ('' if s[0] == 'classic' else f' seed {s[3]}'):40s} done ({time.time() - t0:.0f} s)",
                  flush=True)
    # fixed table order: no-AI methods first, then each AI set-up, seeds in order
    rank = {m: i for i, m in enumerate(list(classic_methods()) + sorted({s[1] for s in specs if s[0] == "agent"}))}
    rows.sort(key=lambda r: (rank[r["method"]], str(r["seed"]), r["reactor"]))
    cf_rows.sort(key=lambda r: (rank[r["method"]], str(r["seed"]), r["failure_at_s"]))

    for path, data in (("agent_eval.csv", rows), ("cooling_failure.csv", cf_rows)):
        with open(os.path.join(OUT, path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    summary, names = {}, list(dict.fromkeys(r["method"] for r in rows))
    for m in names:
        summary[m] = {}
        seeds = sorted(set(r["seed"] for r in rows if r["method"] == m), key=str)
        for group in ("nominal", "grid", "random"):
            per_seed = []
            for seed in seeds:
                rs = [r for r in rows if r["method"] == m and r["seed"] == seed and r["group"] == group]
                done = [r for r in rs if r["feed_min"] is not None]
                reached = [r["time_to_sp_s"] for r in rs if r["time_to_sp_s"] is not None]
                per_seed.append(dict(
                    reactors=len(rs), overheated=sum(r["overheated"] for r in rs),
                    worst_peak_K=max(r["T_peak_K"] for r in rs), unfinished=len(rs) - len(done),
                    mean_feed_min=float(np.mean([r["feed_min"] for r in done])) if done else None,
                    reached_sp=len(reached), mean_time_to_sp_s=float(np.mean(reached)) if reached else None,
                    mean_iae=float(np.mean([r["iae"] for r in rs])),
                    mean_mdc_max=float(np.mean([r["mdc_max"] for r in rs])),
                    mean_mdc_pos_s=float(np.mean([r["mdc_pos_s"] for r in rs])),
                    worst_mtsr_K=max(r["mtsr_peak_K"] for r in rs),
                    mean_valve_moves=float(np.mean([r["feed_tv"] for r in rs]))))
            summary[m][group] = dict(seeds=len(per_seed), per_seed=per_seed)
        cf = [r for r in cf_rows if r["method"] == m]
        summary[m]["cooling_failure"] = dict(tests=len(cf), overheated=sum(r["overheated"] for r in cf),
                                             worst_peak_K=max(r["T_peak_K"] for r in cf))
    json.dump(summary, open(os.path.join(OUT, "agent_eval_summary.json"), "w"), indent=2)

    for group in ("nominal", "grid", "random"):
        print(f"\n{group}  (AI: mean [min-max] over seeds; Fricz measures: IAE, max MDC, s with MDC > 0)")
        print(f"{'Method':34s} {'Overheated':>12s} {'Worst peak K':>18s} {'Unfin.':>7s} {'Feed min':>18s} "
              f"{'Reach SP s':>18s} {'IAE':>22s} {'MDC>0 s':>14s} {'Worst MTSR K':>18s}")
        for m in names:
            ps = summary[m][group]["per_seed"]
            print(f"{m:34s} {fmt([p['overheated'] for p in ps]) + '/' + str(ps[0]['reactors']):>12s} "
                  f"{fmt([p['worst_peak_K'] for p in ps]):>18s} {fmt([p['unfinished'] for p in ps], '{:.0f}'):>7s} "
                  f"{fmt([p['mean_feed_min'] for p in ps]):>18s} {fmt([p['mean_time_to_sp_s'] for p in ps], '{:.0f}'):>18s} "
                  f"{fmt([p['mean_iae'] for p in ps], '{:.0f}'):>22s} {fmt([p['mean_mdc_pos_s'] for p in ps], '{:.0f}'):>14s} "
                  f"{fmt([p['worst_mtsr_K'] for p in ps]):>18s}")
    print(f"\nCooling failure (nominal reactor, {len(cf_times)} failure times per method/seed):")
    for m in names:
        c = summary[m]["cooling_failure"]
        print(f"  {m:34s} overheated {c['overheated']:3d}/{c['tests']:<3d} worst peak {c['worst_peak_K']:.1f} K")

    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(names):
        pk = [r["T_peak_K"] for r in rows if r["method"] == m and r["group"] != "nominal"]
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(pk)), pk, s=8, alpha=0.5)
    ax.axhline(373, color="r", ls="--", lw=1, label="MAT 373 K")
    ax.axhline(363, color="k", ls=":", lw=1, label="setpoint 363 K")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ") for m in names], fontsize=8)
    ax.set_ylabel("Peak reactor temperature (K)"); ax.legend()
    ax.set_title("Reactor 2 (Fricz benchmark): peak temperature on all test reactors")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "agent_eval_peaks.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for ci, config in enumerate(CONFIGS):
        for k, path in enumerate(sorted(glob.glob(os.path.join(AGENTS, f"{config}_seed*_train.csv")))):
            if "_quick" in path and not a.quick:
                continue
            d = list(csv.DictReader(open(path)))
            if not d:
                continue
            ep = [int(r["episode"]) for r in d]
            ax[0].plot(ep, [float(r["return"]) for r in d], f"C{ci}", alpha=0.5, lw=0.8, label=config if k == 0 else None)
            ax[1].plot(ep, np.cumsum([int(r["overheated"]) for r in d]), f"C{ci}", label=config if k == 0 else None)
    ax[0].set_xlabel("Practice batch"); ax[0].set_ylabel("Return (Fricz reward)")
    ax[1].set_xlabel("Practice batch"); ax[1].set_ylabel("Overheated practice batches (cumulative)")
    for x in ax:
        x.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "agent_training.png"), dpi=150); plt.close(fig)
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}/agent_eval.csv, agent_eval_summary.json, cooling_failure.csv, *.png")


if __name__ == "__main__":
    main()
