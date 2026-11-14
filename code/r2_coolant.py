"""
Step 7d: the coolant valve as a second control (extension of reactor 2; not the exact Fricz benchmark).

Question: if the controller may also move the coolant valve, do fewer batches stay unfinished, and
what does it cost in safety margin when the cooling itself fails? Two emergency rules for the brake:
    A  backup = feed shut + coolant opened fully (100 %)   -> lets the controller feed more
    B  backup = feed shut + coolant kept where it is        -> keeps more margin

Methods (all on the 127 test reactors + the 12 cooling-failure tests of r2_evaluate.py):
    no AI   Const feed + coolant PI (no brake)   the classic plant set-up: fixed feed, PI on the coolant
            Max feed + brake A / B               feed request always +5 %, coolant request stays at 80 %
            Coolant PI + max feed + brake A / B  the classic coolant controller behind our brake
    AI      B_cool_A, B_cool_B                   trained with r2_train.py (5 seeds each)
The PI gains are tuned here on the 27 grid reactors (lowest mean IAE; for the no-brake PI, only gain
sets with no overheating on any grid reactor qualify, else the best-tracking set is used and flagged).

    python r2_coolant.py                 # tune the PIs, then test everything (about 2-3 h)
    python r2_coolant.py --quick         # 20 random reactors, 4 failure times, small tuning grid

Output: results/r2/coolant/  (agent_eval.csv, cooling_failure.csv, tuning.json, summary.json)
Step 7 results in results/r2/ are not touched.
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from r2_env import R2DosingEnv
from r2_evaluate import test_plants
from r2_train import COOLANT_CONFIGS, LAMBDA, OUT as AGENTS

OUT = os.path.join("results", "r2", "coolant")
GRID = [p for p in test_plants(0) if p[0] == "grid"]
BRAKES = {"A": 100.0, "B": "hold"}


# ------------------------------------------------------------------ no-AI policies (2 actions)
class CoolantPI:
    """Velocity-form PI on the coolant valve: T below the setpoint -> close the coolant.
    dOPJ = -Kp [(e - e_prev) + dt/Ti e],  e = T_sp - T,  limited to +/-5 % per 50 s like the AI.
    The feed valve moves towards feed_target (% opening) at up to 5 % per step."""

    def __init__(self, Kp: float, Ti: float, feed_target: float = 100.0):
        self.Kp, self.Ti, self.feed_target = Kp, Ti, feed_target

    def __call__(self, obs, env):
        e, e_prev = obs[6] * 50.0, obs[7] * 50.0
        d = -self.Kp * ((e - e_prev) + env.p_nom.dt_control / self.Ti * e)
        a_feed = np.clip((self.feed_target - env.OP) / env.max_step_OP, -1.0, 1.0)
        return np.array([a_feed, np.clip(d / env.max_step_OP, -1.0, 1.0)], dtype=np.float32)

    def __repr__(self):
        return f"CoolantPI(Kp={self.Kp}, Ti={self.Ti}, feed={self.feed_target})"


def max_feed_fixed_coolant(obs, env):
    return np.array([1.0, 0.0], dtype=np.float32)


def make_env(brake: str | None) -> R2DosingEnv:
    if brake is None:
        return R2DosingEnv(lam=LAMBDA, safety="none", coolant_control=True)
    return R2DosingEnv(lam=LAMBDA, safety="shield", shield_type="robust", coolant_control=True,
                       backup_OPJ=BRAKES[brake])


def episode(policy, env: R2DosingEnv, factors: dict, cooling_failure_s=None) -> dict:
    opts = {"plant_factors": factors}
    if cooling_failure_s is not None:
        opts["cooling_failure_s"] = cooling_failure_s
    obs, _ = env.reset(options=opts)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
    return dict(feed_min=info["batch_time_min"], overheated=bool(info["violated"]), T_peak_K=info["T_peak_K"],
                time_to_sp_s=info["time_to_sp_s"], iae=info["iae"], mdc_max=info["mdc_max"],
                mdc_pos_s=info["mdc_pos_s"], mtsr_peak_K=info["mtsr_peak_C"] + 273.15,
                feed_tv=info["feed_tv"], coolant_tv=info["coolant_tv"], interventions=info["n_interventions"])


# ------------------------------------------------------------------ tuning
def _tune_one(args):
    kind, brake, Kp, Ti, feed = args
    env = make_env(brake)
    pol = CoolantPI(Kp, Ti, feed)
    res = [episode(pol, env, f) for _, f in GRID]
    return dict(kind=kind, brake=brake, Kp=Kp, Ti=Ti, feed_target=feed,
                overheated=sum(r["overheated"] for r in res),
                unfinished=sum(r["feed_min"] is None for r in res),
                mean_iae=float(np.mean([r["iae"] for r in res])))


def tune(quick: bool, n_par: int) -> dict:
    Kps = (2.0, 5.0) if quick else (1.0, 2.0, 5.0, 10.0)
    Tis = (1000.0,) if quick else (250.0, 1000.0, 4000.0)
    # below about 55 % the 5 kmol cannot be fed within the 4 h limit (50 % takes 3.97 h + ramp-up)
    feeds = (60.0, 100.0) if quick else (60.0, 70.0, 85.0, 100.0)
    jobs = [("no brake", None, kp, ti, fd) for kp, ti, fd in itertools.product(Kps, Tis, feeds)]
    jobs += [("brake", b, kp, ti, 100.0) for b in BRAKES for kp, ti in itertools.product(Kps, Tis)]
    print(f"Tuning the coolant PI: {len(jobs)} gain sets x 27 grid reactors, {n_par} at a time ...", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=n_par) as pool:
        for f in as_completed([pool.submit(_tune_one, j) for j in jobs]):
            r = f.result(); rows.append(r)
            print(f"  {r['kind']:8s} {str(r['brake']):4s} Kp={r['Kp']:<4} Ti={r['Ti']:<6.0f} feed={r['feed_target']:<5.0f}"
                  f" overheated {r['overheated']:2d}/27 unfinished {r['unfinished']:2d} mean IAE {r['mean_iae']:8.0f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    chosen = {}
    nb = [r for r in rows if r["kind"] == "no brake"]
    safe = [r for r in nb if r["overheated"] == 0 and r["unfinished"] == 0]
    best = min(safe or nb, key=lambda r: r["mean_iae"])
    chosen["no brake"] = dict(best, safe_on_grid=bool(safe))
    for b in BRAKES:
        br = [r for r in rows if r["kind"] == "brake" and r["brake"] == b]
        chosen[b] = min(br, key=lambda r: (r["unfinished"], r["mean_iae"]))
    return dict(rows=rows, chosen=chosen)


def classic_methods(chosen: dict) -> dict:
    """name: (policy, brake or None)"""
    c = chosen["no brake"]
    m = {"Const feed + coolant PI (no brake)": (CoolantPI(c["Kp"], c["Ti"], c["feed_target"]), None)}
    for b in BRAKES:
        m[f"Max feed + brake {b} (no AI)"] = (max_feed_fixed_coolant, b)
        m[f"Coolant PI + max feed + brake {b}"] = (CoolantPI(chosen[b]["Kp"], chosen[b]["Ti"], 100.0), b)
    return m


# ------------------------------------------------------------------ evaluation
def find_agents():
    out = []
    for path in sorted(glob.glob(os.path.join(AGENTS, "*.zip"))):
        m = re.match(r"(.+)_seed(\d+)\.zip$", os.path.basename(path))
        if m and m.group(1) in COOLANT_CONFIGS:
            out.append((m.group(1), int(m.group(2)), path))
    return out


def evaluate(spec, chosen, plants, cf_times):
    import torch
    torch.set_num_threads(1)
    if spec[0] == "classic":
        name = spec[1]
        pol, brake = classic_methods(chosen)[name]
        env, seed = make_env(brake), ""
    else:
        from stable_baselines3 import TD3
        _, name, seed, path = spec
        model = TD3.load(path, device="cpu")
        pol = lambda obs, env: model.predict(obs, deterministic=True)[0]
        kw = {k: v for k, v in COOLANT_CONFIGS[name].items() if k != "randomize_plant"}
        env = R2DosingEnv(lam=LAMBDA, **kw)
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
    ap.add_argument("--retune", action="store_true", help="tune the PIs again even if tuning.json exists")
    a = ap.parse_args()
    out = OUT + ("_quick" if a.quick else "")
    os.makedirs(out, exist_ok=True)
    n_par = a.parallel or max(1, min(6, (os.cpu_count() or 2) // 2))
    t0 = time.time()

    tpath = os.path.join(out, "tuning.json")
    if os.path.exists(tpath) and not a.retune:
        tuning = json.load(open(tpath))
        print(f"Using the PI tuning in {tpath} (--retune to redo it).")
    else:
        tuning = tune(a.quick, n_par)
        json.dump(tuning, open(tpath, "w"), indent=1)
    for k, c in tuning["chosen"].items():
        print(f"  chosen PI ({k}): Kp {c['Kp']}, Ti {c['Ti']:.0f} s, feed {c['feed_target']:.0f} % -> grid overheated "
              f"{c['overheated']}/27, unfinished {c['unfinished']}, mean IAE {c['mean_iae']:.0f}"
              + ("" if k != "no brake" or c["safe_on_grid"] else "   (NO safe gain set found: best tracking used)"))

    plants = test_plants(20 if a.quick else 100)
    cf_times = [600 * k for k in ((2, 5, 8, 11) if a.quick else range(1, 13))]
    specs = [("classic", n) for n in classic_methods(tuning["chosen"])]
    specs += [("agent", c, s, p) for c, s, p in find_agents()]
    if not any(s[0] == "agent" for s in specs):
        print("No B_cool_* agents found yet (train them with r2_train.py); testing the no-AI methods only.")
    print(f"Testing {len(specs)} methods on {len(plants)} reactors + {len(cf_times)} cooling failures, "
          f"{n_par} at a time ...", flush=True)
    rows, cf_rows = [], []
    with ProcessPoolExecutor(max_workers=n_par) as pool:
        futs = {pool.submit(evaluate, s, tuning["chosen"], plants, cf_times): s for s in specs}
        for f in as_completed(futs):
            r, c = f.result(); rows += r; cf_rows += c
            s = futs[f]
            print(f"  {s[1] + ('' if s[0] == 'classic' else f' seed {s[2]}'):42s} done ({time.time() - t0:.0f} s)",
                  flush=True)
    order = list(classic_methods(tuning["chosen"])) + list(COOLANT_CONFIGS)
    rows.sort(key=lambda r: (order.index(r["method"]), str(r["seed"]), r["reactor"]))
    cf_rows.sort(key=lambda r: (order.index(r["method"]), str(r["seed"]), r["failure_at_s"]))
    for name, data in (("agent_eval.csv", rows), ("cooling_failure.csv", cf_rows)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    # summary (test = grid + random, like Step 8)
    print(f"\n{'Method':42s} {'seeds':>5s} {'overheated':>12s} {'unfinished':>11s} {'mean IAE':>10s} "
          f"{'worst peak K':>12s} {'cooling failure':>16s} {'worst CF peak':>13s}")
    summary = {}
    for m in [x for x in order if any(r["method"] == x for r in rows)]:
        t = [r for r in rows if r["method"] == m and r["group"] != "nominal"]
        c = [r for r in cf_rows if r["method"] == m]
        summary[m] = dict(seeds=len({r["seed"] for r in t}), test=len(t), overheated=sum(r["overheated"] for r in t),
                          unfinished=sum(r["feed_min"] is None for r in t), mean_iae=float(np.mean([r["iae"] for r in t])),
                          worst_peak_K=max(r["T_peak_K"] for r in t), cf_tests=len(c),
                          cf_overheated=sum(r["overheated"] for r in c), cf_worst_peak_K=max(r["T_peak_K"] for r in c))
        s = summary[m]
        print(f"{m:42s} {s['seeds']:>5d} {s['overheated']:>5d}/{s['test']:<6d} {s['unfinished']:>5d}/{s['test']:<5d} "
              f"{s['mean_iae']:>10.0f} {s['worst_peak_K']:>12.1f} {s['cf_overheated']:>7d}/{s['cf_tests']:<8d} "
              f"{s['cf_worst_peak_K']:>13.1f}")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min -> {out}/")


if __name__ == "__main__":
    main()
