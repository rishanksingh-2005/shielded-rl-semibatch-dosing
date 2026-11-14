"""
Step 5: train the two AI drivers.

    Driver A = AI that is only TOLD to be careful (reward penalty above 50 C).
               This copies the closest paper (Fricz et al. 2026, TD3 + runaway penalty).
    Driver B = the SAME AI, with every move passed through the emergency brake (shield.py).

Both use the same algorithm (TD3, as in Fricz et al.), the same network, the same reward
and the same training budget. The only difference is the brake.

Four training set-ups (each is trained with several random seeds):

    A_nominal   Driver A, practises only on the nominal model (like the tuned baselines)
    A_random    Driver A, practises on random 'wrong-chemistry' reactors (+/-20 %)
                -> the strongest fair version of the penalty-only idea
    B_robust    Driver B with the careful (robust) brake, random reactors (+/-20 %)
                -> our method
    B_nominal   Driver B with the normal (nominal-model) brake, nominal model only
                -> shows that a brake built on the wrong model is not enough, even with AI

Usage (inside code/, venv active):

    python train_agents.py --list                       # show set-ups and run plan
    python train_agents.py --config A_random --seed 0   # one run
    python train_agents.py --all                        # every set-up x 3 seeds (the full Step 5)
    python train_agents.py --all --seeds 0-4            # 5 seeds (finished runs are skipped)
    python train_agents.py --all --parallel 2           # force 2 runs at a time (same results)
    python train_agents.py --all --quick                # 5,000 steps each: checks that it all works

Every run writes, into results/agents/:
    <config>_seed<k>.zip         the trained AI
    <config>_seed<k>_train.csv   one line per practice batch: time, peak T, overheated?, ...
    checkpoints/<config>_seed<k>_<n>_steps.zip  the AI every 20,000 steps (learning curves)
A run that is already finished is skipped, so --all can be stopped and restarted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from reactor_env import ReactorDosingEnv

OUT = os.path.join("results", "agents")
SEEDS = (0, 1, 2)
TOTAL_STEPS = 100_000  # about 450 practice batches. Trial runs: Driver B settles by 20k steps,
                       # Driver A swings between overheating and over-caution until ~100k.

CONFIGS = {
    "A_nominal": dict(safety="penalty", randomize_plant=False),
    "A_random":  dict(safety="penalty", randomize_plant=True),
    "B_robust":  dict(safety="shield", shield_type="robust", randomize_plant=True),
    "B_nominal": dict(safety="shield", shield_type="nominal", randomize_plant=False),
}

# One set of settings for every driver (no per-driver tuning, so the comparison is fair).
TD3_SETTINGS = dict(
    learning_rate=1e-3,
    buffer_size=200_000,
    learning_starts=2_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)
EXPLORATION_NOISE = 0.2     # std of the Gaussian noise added to the AI's moves while practising


def make_env(config: str) -> ReactorDosingEnv:
    return ReactorDosingEnv(**CONFIGS[config])


class BatchLogger(BaseCallback):
    """Writes one CSV line per finished practice batch (episode)."""

    FIELDS = ["episode", "timestep", "return", "batch_min", "T_peak_C", "overheated", "runaway",
              "interventions", "feed_tv", "k_factor", "dH_factor", "UA_factor", "wall_s"]

    def __init__(self, path: str, print_every: int = 25):
        super().__init__()
        self.path, self.print_every = path, print_every
        self.ep, self.ret, self.t0 = 0, 0.0, time.time()
        self.recent = []

    def _on_training_start(self):
        self.fh = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=self.FIELDS)
        self.w.writeheader()

    def _on_step(self) -> bool:
        self.ret += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.ep += 1
            row = dict(episode=self.ep, timestep=self.num_timesteps, **{"return": round(self.ret, 3)},
                       batch_min=info["batch_time_min"], T_peak_C=round(info["T_peak_C"], 3),
                       overheated=int(info["violated"]), runaway=int(info["runaway"]),
                       interventions=info["n_interventions"], feed_tv=round(info["feed_tv"], 1),
                       wall_s=round(time.time() - self.t0, 1),
                       **{k: round(v, 4) for k, v in info["plant_factors"].items()})
            self.w.writerow(row)
            self.fh.flush()
            self.recent.append(row)
            if self.ep % self.print_every == 0:
                rs = self.recent[-self.print_every:]
                done = [r["batch_min"] for r in rs if r["batch_min"] is not None]
                print(f"   step {self.num_timesteps:7d} | batch {self.ep:4d} | "
                      f"last {len(rs)}: mean time {np.mean(done) if done else float('nan'):5.1f} min, "
                      f"overheated {sum(r['overheated'] for r in rs):2d}, "
                      f"max peak {max(r['T_peak_C'] for r in rs):5.1f} C | "
                      f"{self.num_timesteps / (time.time() - self.t0):4.0f} steps/s", flush=True)
            self.ret = 0.0
        return True

    def _on_training_end(self):
        self.fh.close()


def train(config: str, seed: int, total_steps: int, tag: str = "", quiet: bool = False) -> dict:
    os.makedirs(OUT, exist_ok=True)
    name = f"{config}_seed{seed}{tag}"
    model_path = os.path.join(OUT, name + ".zip")
    if os.path.exists(model_path):
        print(f"[skip] {name} already trained ({model_path})")
        return {}
    torch.set_num_threads(1)          # small network: 1 thread is fastest and reproducible
    env = make_env(config)
    noise = NormalActionNoise(mean=np.zeros(1), sigma=EXPLORATION_NOISE * np.ones(1))
    model = TD3("MlpPolicy", env, action_noise=noise, seed=seed, device="cpu", verbose=0,
                **TD3_SETTINGS)
    print(f"[train] {name}: {total_steps:,} steps ({CONFIGS[config]})", flush=True)
    t0 = time.time()
    ckpt = CheckpointCallback(save_freq=20_000, save_path=os.path.join(OUT, "checkpoints"),
                              name_prefix=name, save_replay_buffer=False)
    ckpt_logger = BatchLogger(os.path.join(OUT, name + "_train.csv"),
                              print_every=1_000_000 if quiet else 25)
    model.learn(total_timesteps=total_steps, callback=CallbackList([ckpt_logger, ckpt]))
    model.save(model_path)
    minutes = (time.time() - t0) / 60
    rows = ckpt_logger.recent
    print(f"[done]  {name} in {minutes:.1f} min: {len(rows)} practice batches, "
          f"{sum(r['overheated'] for r in rows)} overheated while practising -> {model_path}", flush=True)
    meta = dict(config=config, env=CONFIGS[config], seed=seed, total_steps=total_steps,
                algorithm="TD3", settings={k: str(v) for k, v in TD3_SETTINGS.items()},
                exploration_noise=EXPLORATION_NOISE, minutes=round(minutes, 2),
                practice_batches=len(rows), practice_overheated=sum(r["overheated"] for r in rows),
                torch=torch.__version__)
    with open(os.path.join(OUT, name + "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--all", action="store_true", help="every set-up x seeds 0,1,2")
    ap.add_argument("--seeds", default=None, help="seed range, e.g. 0-4 (default for --all: 0-2)")
    ap.add_argument("--quick", action="store_true", help="5,000 steps, saved with a _quick tag")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parallel", type=int, default=0,
                    help="runs at the same time (each uses 1 CPU core; results do not change). "
                         "Default with --all: half the logical CPUs, at most 6")
    a = ap.parse_args()

    steps, tag = (5_000, "_quick") if a.quick else (a.steps, "")
    if a.list or not (a.all or a.config):
        print("Set-ups:")
        for k, v in CONFIGS.items():
            print(f"  {k:10s} {v}")
        print(f"Full Step 5 = {len(CONFIGS)} set-ups x {len(SEEDS)} seeds x {TOTAL_STEPS:,} steps")
        return
    if a.seeds:
        lo, _, hi = a.seeds.partition("-")
        seeds = list(range(int(lo), int(hi or lo) + 1))
    else:
        seeds = list(SEEDS) if a.all else [a.seed]
    jobs = [(c, s) for c in (list(CONFIGS) if a.all else [a.config]) for s in seeds]
    t0 = time.time()
    n_par = a.parallel or (max(1, min(6, (os.cpu_count() or 2) // 2)) if len(jobs) > 1 else 1)
    if n_par > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Running {len(jobs)} runs, {n_par} at a time ({os.cpu_count()} logical CPUs found). "
              f"A line appears as each run starts and finishes.", flush=True)
        with ProcessPoolExecutor(max_workers=n_par) as pool:
            futs = [pool.submit(train, c, s, steps, tag, True) for c, s in jobs]
            for f in futs:
                f.result()
    else:
        for i, (c, s) in enumerate(jobs, 1):
            print(f"=== run {i}/{len(jobs)} ===")
            train(c, s, steps, tag)
    print(f"All {len(jobs)} runs finished in {(time.time() - t0) / 60:.0f} min.")


if __name__ == "__main__":
    main()
