"""
Step 7d checks for the coolant-valve extension.            python r2_test_coolant.py   (~10-20 min)

1. Nothing changed for Step 7: the one-valve brake run reproduces the stored Step 7 result exactly.
2. The two-valve training ground passes Gymnasium's own checks.
3. With the coolant held at 80 %, brake B (backup keeps the coolant) gives the Step 7 brake's answer.
4. Safety: a random two-valve driver behind brake A and brake B never passes MAT on the 27 grid
   reactors + 20 random reactors (inside the +/-20 % design range).
5. Step 7m finish-the-batch reward: identical rewards except the last step (bonus / unfinished penalty),
   and the 4 h limit becomes terminal only when the option is on.
6. Step 7o delay-aware brake: maximum feed survives all 12 cooling failures (feed on for 100 s after).
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

from r2_baselines import max_feed
from r2_coolant import BRAKES, make_env, max_feed_fixed_coolant
from r2_env import R2DosingEnv
from r2_evaluate import test_plants

NOMINAL = dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0)
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def run(env, policy, factors, rng=None):
    obs, _ = env.reset(options={"plant_factors": factors})
    done, ops = False, []
    while not done:
        act = policy(obs, env) if rng is None else rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, _, term, trunc, info = env.step(act)
        ops.append((info["OP_applied"], info["OPJ_applied"]))
        done = term or trunc
    return info, ops


def main():
    t0 = time.time()
    print("1. Step 7 unchanged (one valve, careful brake, max feed, nominal reactor)")
    info, _ = run(R2DosingEnv(safety="shield", shield_type="robust"), max_feed, NOMINAL)
    ref = None
    path = os.path.join("results", "r2", "agent_eval.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["method"] == "Max feed + robust brake (no AI)" and r["group"] == "nominal":
                ref = r
    if ref:
        check("IAE and peak identical to the stored Step 7 result",
              abs(info["iae"] - float(ref["iae"])) < 1e-6 and abs(info["T_peak_K"] - float(ref["T_peak_K"])) < 1e-9,
              f"(IAE {info['iae']:.4f} vs {float(ref['iae']):.4f}; peak {info['T_peak_K']:.4f} K)")
    else:
        print("  (no stored Step 7 result found; skipped)")

    print("2. Gymnasium checks on the two-valve training ground")
    from gymnasium.utils.env_checker import check_env
    try:
        for b in BRAKES:
            check_env(make_env(b), skip_render_check=True)
        check_env(make_env(None), skip_render_check=True)
        check("check_env (brake A, brake B, no brake)", True)
    except Exception as e:                          # noqa: BLE001
        check("check_env", False, repr(e))

    print("3. Brake B with the coolant held at 80 % behaves like the Step 7 brake")
    i7, ops7 = run(R2DosingEnv(safety="shield", shield_type="robust"), max_feed, NOMINAL)
    iB, opsB = run(make_env("B"), max_feed_fixed_coolant, NOMINAL)
    same = len(ops7) == len(opsB) and all(abs(a[0] - b[0]) < 1e-9 for a, b in zip(ops7, opsB))
    raised = sum(b[1] > 80.0 + 1e-9 for b in opsB)
    check("same feed openings at every step", same,
          f"(steps {len(ops7)} vs {len(opsB)}; coolant raised by the brake in {raised} steps; "
          f"IAE {i7['iae']:.0f} vs {iB['iae']:.0f})")

    print("4. Safety: random two-valve driver behind each brake, 27 grid + 20 random reactors")
    plants = [p for p in test_plants(20) if p[0] != "nominal"]
    for b in BRAKES:
        env, rng = make_env(b), np.random.default_rng(123)
        t1, over, worst, unfinished = time.time(), 0, 0.0, 0
        for _, f in plants:
            info, _ = run(env, None, f, rng)
            over += info["violated"]; worst = max(worst, info["T_peak_K"]); unfinished += info["batch_time_min"] is None
        check(f"brake {b}: overheated 0/{len(plants)}", over == 0,
              f"(overheated {over}, worst peak {worst:.2f} K, unfinished {unfinished}, "
              f"{(time.time() - t1) / len(plants):.1f} s per batch)")

    print("5. Finish-the-batch reward (Step 7m): only the last step's reward changes")
    def rewards(env, act):
        env.reset(options={"plant_factors": NOMINAL})
        rs, done = [], False
        while not done:
            _, r, term, trunc, info = env.step(np.array(act, dtype=np.float32))
            rs.append(r); done = term or trunc
        return np.array(rs), term, trunc, info
    kw = dict(safety="shield", shield_type="robust", override_penalty=1.0, coolant_control=True, backup_OPJ="hold")
    r0, *_ = rewards(R2DosingEnv(**kw), [1.0, 0.0])
    r1, term, trunc, info = rewards(R2DosingEnv(finish_reward=True, **kw), [1.0, 0.0])
    n, M = len(r1), R2DosingEnv().max_steps
    check("finished batch: bonus +10 x remaining steps on the last step only",
          len(r0) == n and np.allclose(r0[:-1], r1[:-1]) and abs(r1[-1] - r0[-1] - 10.0 * (M - n)) < 1e-9 and term,
          f"(finished at step {n} of {M}; bonus {r1[-1] - r0[-1]:.0f})")
    r2, term, trunc, info = rewards(R2DosingEnv(finish_reward=True, **kw), [-1.0, 0.0])
    check("unfinished batch: terminal at 4 h with -10 x max_steps x unfed fraction",
          len(r2) == M and term and not trunc and abs(info["finish_term"] + 10.0 * M) < 1e-9,
          f"(steps {len(r2)}, terminated={term}, truncated={trunc}, penalty {info['finish_term']:.0f})")
    r3, term, trunc, _ = rewards(R2DosingEnv(**kw), [-1.0, 0.0])
    check("default (finish_reward off) still truncates at 4 h", trunc and not term and np.allclose(r3, r2[:-1].tolist() + [r3[-1]]))

    print("6. Delay-aware brake (Step 7o): max feed, nominal reactor, cooling failure every 10 min")
    env = R2DosingEnv(safety="shield", shield_type="robust", cf_aware="nominal")
    peaks = []
    for k in range(1, 13):
        env.reset(options={"plant_factors": NOMINAL, "cooling_failure_s": 600 * k})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(max_feed(None, env))
            done = term or trunc
        peaks.append(info["T_peak_K"])
    check("0/12 cooling failures pass MAT", max(peaks) <= 373.0 + 1e-6, f"(worst peak {max(peaks):.2f} K)")

    print(f"\n{'All checks passed' if not failures else 'FAILED: ' + ', '.join(failures)} "
          f"({time.time() - t0:.0f} s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
