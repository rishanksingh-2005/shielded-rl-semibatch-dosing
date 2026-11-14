"""
Step 4 check: run this ONCE on your laptop (VS Code terminal, venv active, inside code/):

    python check_env_setup.py

It checks that (1) the AI libraries work, (2) the training environment follows the
Gymnasium rules, (3) the environment gives exactly the same reactor results as the
simulator, (4) the emergency brake keeps every test batch safe, and (5) a tiny AI
training run works. It takes about 2-4 minutes. Keep the whole output: it is the record that the set-up works.
"""
import platform
import sys
import time

import numpy as np

print("=" * 70)
print("STEP 4 CHECK: AI training environment")
print("=" * 70)
print(f"Python {platform.python_version()} on {platform.system()}")

# ---------------------------------------------------------------- 1. libraries
import scipy
import gymnasium
import stable_baselines3
import torch
print(f"numpy {np.__version__} | scipy {scipy.__version__} | gymnasium {gymnasium.__version__} | "
      f"stable-baselines3 {stable_baselines3.__version__} | torch {torch.__version__}")
print("[1/5] PASS libraries import")

from gymnasium.utils.env_checker import check_env

from controllers import FixedTiming
from reactor_env import ReactorDosingEnv, run_episode
from reactor_model import ReactorParams, simulate

p = ReactorParams()
FULL_SPEED = np.array([1.0], dtype=np.float32)

# ---------------------------------------------------------------- 2. Gymnasium rules
for safety in ("penalty", "shield", "none"):
    check_env(ReactorDosingEnv(safety=safety), skip_render_check=True)
print("[2/5] PASS environment follows the Gymnasium rules (penalty, shield, none)")

# ---------------------------------------------------------------- 3. same as simulator
F = 8.0
sim = simulate(FixedTiming(F), p)
env = ReactorDosingEnv(safety="none")
res = run_episode(env, lambda obs, e: e.feed_to_action(F, p.F_max_mL_min))
assert res["batch_time_min"] is not None and abs(res["batch_time_min"] - sim.batch_time_s / 60) < 1e-9, (res, sim.batch_time_s)
assert abs(res["T_peak_C"] - sim.T_peak_C) < 1e-6, (res["T_peak_C"], sim.T_peak_C)
print(f"[3/5] PASS environment = simulator (constant {F} mL/min: {res['batch_time_min']:.2f} min, "
      f"peak {res['T_peak_C']:.3f} C in both)")

# ---------------------------------------------------------------- 4. brake keeps it safe
rows = []
r = run_episode(ReactorDosingEnv(safety="penalty"), lambda o, e: FULL_SPEED)
rows.append(("No brake, full speed (correct chemistry)", r))
assert r["violated"], "full speed without brake should overheat"
r = run_episode(ReactorDosingEnv(safety="shield", shield_type="nominal"), lambda o, e: FULL_SPEED)
rows.append(("Normal brake, full speed (correct chemistry)", r))
assert not r["violated"] and abs(r["batch_time_min"] - 31.67) < 0.2, r
r = run_episode(ReactorDosingEnv(safety="shield", shield_type="robust"), lambda o, e: FULL_SPEED)
rows.append(("Careful brake, full speed (correct chemistry)", r))
assert not r["violated"] and abs(r["batch_time_min"] - 33.83) < 0.2, r
env = ReactorDosingEnv(safety="shield", shield_type="robust", randomize_plant=True)
worst = None
for seed in range(5):
    r = run_episode(env, lambda o, e: FULL_SPEED, seed=seed)
    assert not r["violated"], ("careful brake overheated on a wrong-chemistry reactor", r)
    worst = r if worst is None or r["T_peak_C"] > worst["T_peak_C"] else worst
rows.append(("Careful brake, 5 wrong-chemistry reactors (worst)", worst))
for name, r in rows:
    bt = "not finished" if r["batch_time_min"] is None else f"{r['batch_time_min']:.1f} min"
    print(f"      {name:52s} {bt:>13s}  peak {r['T_peak_C']:5.1f} C  overheated={r['violated']}")
print("[4/5] PASS brake keeps every test batch at or below 50 C; no brake overheats")

# ---------------------------------------------------------------- 5. tiny AI training run
from stable_baselines3 import PPO
for safety in ("penalty", "shield"):
    env = ReactorDosingEnv(safety=safety, shield_type="robust")
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, verbose=0, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=2048)
    rate = 2048 / (time.time() - t0)
    print(f"      Driver {'A (penalty only)' if safety == 'penalty' else 'B (with brake) '}: "
          f"{rate:6.0f} steps/s  -> 200,000 training steps would take about {200000 / rate / 60:.0f} min")
print("[5/5] PASS a short AI training run works for both drivers")
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
