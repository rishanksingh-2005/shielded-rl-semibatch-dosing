"""
Step 5/6 evaluation: test every trained AI driver against the normal controllers.

Test reactors (the AI never chooses them, and random ones are NOT the practice reactors):
    nominal      the model is exactly right                                   (1 reactor)
    grid         k, dH, UA each x0.8 / x1.0 / x1.2, as in run_baselines.py     (27 reactors)
    random       k, dH, UA each uniform in x0.8 - x1.2, fixed test seed        (100 reactors)

Measures: batch time; peak temperature and 'overheated' (above 50 C); feed chatter (sum of
|feed changes|, mL/min); and worst MTSR = the highest Stoessel MTSR on the TRUE reactor during
the batch, i.e. how hot it would get if the cooling failed at the worst moment.
Note: none of the controllers here (the brake included) limits MTSR to 50 C. The brake checks
that STOPPING THE FEED (cooling on) keeps T <= 50 C. MTSR is a different safety question and
is judged against the maximum technical temperature (e.g. the boiling point of the mixture),
not against the 50 C operating limit. It is reported so readers can see this risk.

Methods:
    every trained AI in results/agents/  (A_nominal, A_random, B_robust, B_nominal; all seeds)
    PID (robust tuned), Fixed timing (robust tuned)   - settings read from results/summary.json
    Max feed + robust brake                           - the brake alone, no AI
    Oracle                                            - brake that knows the true reactor (best possible)

    python evaluate_agents.py            # all trained AIs (~5-15 min)
    python evaluate_agents.py --quick    # also include *_quick test models, 20 random reactors
    python evaluate_agents.py --checkpoints   # learning curve: every checkpoint on the 27 grid reactors

Output (results/):
    agent_eval.csv            one line per method x reactor
    agent_eval_summary.json   the summary table
    agent_eval_peaks.png      peak temperature of every method on every test reactor
    agent_training.png        practice curves: batch time and overheating while learning
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

from controllers import FixedTiming, PIDDosing
from reactor_env import ReactorDosingEnv
from reactor_model import ReactorParams, simulate
from shield import Shield, max_feed_with_shield, robust_models

OUT = "results"
AGENTS = os.path.join(OUT, "agents")
TEST_SEED = 777            # different from every training seed
P = ReactorParams()


def test_plants(n_random: int):
    plants = [("nominal", dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0))]
    for kf, hf, uf in itertools.product((0.8, 1.0, 1.2), repeat=3):
        plants.append(("grid", dict(k_factor=kf, dH_factor=hf, UA_factor=uf)))
    rng = np.random.default_rng(TEST_SEED)
    for _ in range(n_random):
        f = rng.uniform(0.8, 1.2, 3)
        plants.append(("random", dict(k_factor=f[0], dH_factor=f[1], UA_factor=f[2])))
    return plants


def run_agent(model, env: ReactorDosingEnv, factors: dict) -> dict:
    obs, _ = env.reset(options={"plant_factors": factors})
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return dict(batch_min=info["batch_time_min"], T_peak_C=info["T_peak_C"], overheated=info["violated"],
                mtsr_peak_C=info["mtsr_peak_C"], feed_tv=env.feed_tv, interventions=env.n_interventions)


def run_classic(controller, plant: ReactorParams) -> dict:
    r = simulate(controller, plant)
    F = np.concatenate([[0.0], r.F])
    return dict(batch_min=None if r.batch_time_s is None else r.batch_time_s / 60, T_peak_C=r.T_peak_C,
                overheated=r.violated, mtsr_peak_C=float(r.mtsr_C.max()), feed_tv=float(np.abs(np.diff(F)).sum()), interventions=None)


def classic_methods():
    tuned = json.load(open(os.path.join(OUT, "summary.json")))["tuned"]
    pr = tuned["pid_robust"]
    return {
        "PID (robust tuned)": lambda plant: PIDDosing(pr["T_sp_C"], pr["Kp"], pr["Ti"], pr["Td"],
                                                      F_max=P.F_max_mL_min, dt=P.dt_control),
        "Fixed timing (robust tuned)": lambda plant: FixedTiming(tuned["fixed_robust_F_mL_min"]),
        "Max feed + robust brake (no AI)": lambda plant: max_feed_with_shield(Shield(robust_models(P))),
        "Oracle": lambda plant: max_feed_with_shield(Shield([plant])),
    }


def find_agents(include_quick: bool):
    from train_agents import CONFIGS
    out = []
    for path in sorted(glob.glob(os.path.join(AGENTS, "*.zip"))):
        m = re.match(r"(.+)_seed(\d+)(_quick)?\.zip$", os.path.basename(path))
        if not m or m.group(1) not in CONFIGS or (m.group(3) and not include_quick):
            continue
        out.append((m.group(1), int(m.group(2)), bool(m.group(3)), path))
    return out


def evaluate_checkpoints():
    """Test every saved checkpoint on the 27 grid reactors: test score vs practice steps."""
    from stable_baselines3 import TD3
    from train_agents import CONFIGS
    plants = [f for g, f in test_plants(0) if g == "grid"]
    oracle = [run_classic(max_feed_with_shield(Shield([P.with_mismatch(**f)])), P.with_mismatch(**f))["batch_min"]
              for f in plants]
    rows = []
    for path in sorted(glob.glob(os.path.join(AGENTS, "checkpoints", "*_steps.zip"))):
        m = re.match(r"(.+)_seed(\d+)_(\d+)_steps\.zip$", os.path.basename(path))
        if not m or m.group(1) not in CONFIGS:
            continue
        config, seed, steps = m.group(1), int(m.group(2)), int(m.group(3))
        model = TD3.load(path, device="cpu")
        env_kw = dict(CONFIGS[config]); env_kw["randomize_plant"] = False
        env = ReactorDosingEnv(**env_kw)
        res = [run_agent(model, env, f) for f in plants]
        slow = [100 * (r["batch_min"] / o - 1) for r, o in zip(res, oracle) if r["batch_min"] is not None]
        rows.append(dict(config=config, seed=seed, steps=steps,
                         overheated=sum(bool(r["overheated"]) for r in res),
                         worst_peak_C=max(r["T_peak_C"] for r in res),
                         unfinished=sum(r["batch_min"] is None for r in res),
                         slower_than_oracle_pct=float(np.mean(slow)) if slow else None))
        print(f"  {config} seed {seed} at {steps:7,d} steps: overheated {rows[-1]['overheated']:2d}/27, "
              f"slower than oracle {rows[-1]['slower_than_oracle_pct'] or float('nan'):5.1f} %", flush=True)
    if not rows:
        print("No checkpoints found in results/agents/checkpoints/.")
        return
    with open(os.path.join(OUT, "agent_checkpoints.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for config in CONFIGS:
        c = f"C{list(CONFIGS).index(config)}"
        for k, seed in enumerate(sorted(set(r["seed"] for r in rows if r["config"] == config))):
            rs = sorted((r for r in rows if r["config"] == config and r["seed"] == seed), key=lambda r: r["steps"])
            lab = config if k == 0 else None
            ax[0].plot([r["steps"] for r in rs], [r["slower_than_oracle_pct"] for r in rs], c + "-o", ms=3, label=lab)
            ax[1].plot([r["steps"] for r in rs], [r["overheated"] for r in rs], c + "-o", ms=3, label=lab)
    ax[0].set_ylabel("Slower than oracle on 27 test reactors (%)"); ax[1].set_ylabel("Test reactors overheated (of 27)")
    for x in ax:
        x.set_xlabel("Practice steps"); x.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "agent_checkpoints.png"), dpi=150); plt.close(fig)
    print("-> results/agent_checkpoints.csv, agent_checkpoints.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--checkpoints", action="store_true",
                    help="instead: test every 20,000-step checkpoint on the 27 grid reactors")
    a = ap.parse_args()
    if a.checkpoints:
        evaluate_checkpoints()
        return
    from stable_baselines3 import TD3
    from train_agents import CONFIGS

    plants = test_plants(20 if a.quick else a.n_random)
    agents = find_agents(a.quick)
    if not agents:
        print("No trained AIs found in results/agents/. Run train_agents.py first.")
        return
    t0, rows = time.time(), []

    for name, make in classic_methods().items():
        for i, (group, f) in enumerate(plants):
            plant = P.with_mismatch(**f)
            rows.append(dict(method=name, config=name, seed="", reactor=i, group=group, **f,
                             **run_classic(make(plant), plant)))
        print(f"  {name:34s} done ({time.time()-t0:.0f} s)", flush=True)

    for config, seed, quick, path in agents:
        model = TD3.load(path, device="cpu")
        env_kw = dict(CONFIGS[config]); env_kw["randomize_plant"] = False   # test reactors set explicitly
        env = ReactorDosingEnv(**env_kw)
        label = f"{config}{'_quick' if quick else ''}"
        for i, (group, f) in enumerate(plants):
            rows.append(dict(method=f"AI {label}", config=label, seed=seed, reactor=i, group=group, **f,
                             **run_agent(model, env, f)))
        print(f"  AI {label} seed {seed:<2d}{'':18s} done ({time.time()-t0:.0f} s)", flush=True)

    # time relative to the oracle on the same reactor
    oracle = {r["reactor"]: r["batch_min"] for r in rows if r["method"] == "Oracle"}
    for r in rows:
        r["slower_than_oracle_pct"] = (None if r["batch_min"] is None
                                       else 100 * (r["batch_min"] / oracle[r["reactor"]] - 1))
    with open(os.path.join(OUT, "agent_eval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # summary: per method, per test group; AI results as mean and range over seeds
    summary = {}
    methods = list(dict.fromkeys(r["config"] for r in rows))
    for m in methods:
        summary[m] = {}
        for group in ("nominal", "grid", "random"):
            per_seed = []
            for seed in sorted(set(r["seed"] for r in rows if r["config"] == m), key=str):
                rs = [r for r in rows if r["config"] == m and r["seed"] == seed and r["group"] == group]
                done = [r for r in rs if r["batch_min"] is not None]
                per_seed.append(dict(
                    overheated=sum(bool(r["overheated"]) for r in rs), reactors=len(rs),
                    worst_peak_C=float(max(r["T_peak_C"] for r in rs)),
                    worst_mtsr_C=float(max(r["mtsr_peak_C"] for r in rs)),
                    mean_batch_min=float(np.mean([r["batch_min"] for r in done])) if done else None,
                    slower_than_oracle_pct=float(np.mean([r["slower_than_oracle_pct"] for r in done])) if done else None,
                    unfinished=len(rs) - len(done),
                    mean_feed_tv=float(np.mean([r["feed_tv"] for r in rs]))))
            summary[m][group] = dict(seeds=len(per_seed), per_seed=per_seed)
    json.dump(summary, open(os.path.join(OUT, "agent_eval_summary.json"), "w"), indent=2)

    def fmt(vals, f="{:.1f}"):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        return f.format(np.mean(vals)) if len(vals) == 1 or np.ptp(vals) == 0 else \
            f"{f.format(np.mean(vals))} [{f.format(min(vals))}-{f.format(max(vals))}]"

    for group, title in (("nominal", "Nominal model (1 reactor)"), ("grid", "27 grid reactors"),
                         ("random", f"{len([p for p in plants if p[0]=='random'])} new random reactors")):
        print(f"\n{title}  (AI: mean [min-max] over seeds)")
        print(f"{'Method':36s} {'Overheated':>14s} {'Worst peak C':>16s} {'Mean batch min':>18s} "
              f"{'Slower than oracle %':>22s} {'Feed chatter':>14s} {'Worst MTSR C':>16s}")
        for m in methods:
            ps = summary[m][group]["per_seed"]
            n = ps[0]["reactors"]
            print(f"{m:36s} {fmt([p['overheated'] for p in ps], '{:.1f}') + f'/{n}':>14s} "
                  f"{fmt([p['worst_peak_C'] for p in ps]):>16s} {fmt([p['mean_batch_min'] for p in ps]):>18s} "
                  f"{fmt([p['slower_than_oracle_pct'] for p in ps]):>22s} {fmt([p['mean_feed_tv'] for p in ps], '{:.0f}'):>14s} "
                  f"{fmt([p['worst_mtsr_C'] for p in ps]):>16s}")

    # ---------- figure 1: peak temperatures ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        peaks = [r["T_peak_C"] for r in rows if r["config"] == m and r["group"] != "nominal"]
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(peaks)), peaks, s=8, alpha=0.5)
    ax.axhline(50, color="r", ls="--", lw=1, label="limit 50 C")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ") for m in methods], fontsize=8)
    ax.set_ylabel("Peak reactor temperature (C)"); ax.legend()
    ax.set_title("Peak temperature on all test reactors (grid + random); AI: all seeds")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "agent_eval_peaks.png"), dpi=150); plt.close(fig)

    # ---------- figure 2: practice (training) curves ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for config in CONFIGS:
        for k, path in enumerate(sorted(glob.glob(os.path.join(AGENTS, f"{config}_seed*_train.csv")))):
            if "_quick" in path and not a.quick:
                continue
            d = list(csv.DictReader(open(path)))
            if not d:
                continue
            ep = np.array([int(r["episode"]) for r in d])
            bt = np.array([float(r["batch_min"]) if r["batch_min"] else np.nan for r in d])
            ov = np.cumsum([int(r["overheated"]) for r in d])
            c = f"C{list(CONFIGS).index(config)}"
            lab = config if k == 0 else None
            ax[0].plot(ep, bt, c, alpha=0.5, lw=0.8, label=lab)
            ax[1].plot(ep, ov, c, alpha=0.8, label=lab)
    ax[0].set_xlabel("Practice batch"); ax[0].set_ylabel("Batch time (min)"); ax[0].set_ylim(25, 80)
    ax[1].set_xlabel("Practice batch"); ax[1].set_ylabel("Overheated practice batches (cumulative)")
    ax[0].legend(); ax[1].legend()
    ax[0].set_title("Learning speed"); ax[1].set_title("Safety while learning")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "agent_training.png"), dpi=150); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f} s -> results/agent_eval.csv, agent_eval_summary.json, "
          f"agent_eval_peaks.png, agent_training.png")


if __name__ == "__main__":
    main()
