"""
Step 7: train the AI drivers on the second reactor (Fricz et al. 2026 benchmark).

Same four set-ups as Step 5 (reactor 1), with Fricz's task, reward and TD3 settings:
    A_nominal   Driver A (Fricz: MDC penalty, lambda = 1000), nominal reactor only
    A_random    Driver A, random 'wrong-chemistry' reactors (+/-20 %)
    B_robust    Driver B = same AI + careful brake, random reactors (+/-20 %)   <- our method
    B_nominal   Driver B with the normal brake, nominal reactor only
    B_robust_pen  B_robust + override penalty (Step 7b): each brake override costs
                  5 x (requested - allowed valve %) / 5 %, so the AI learns to stay inside
                  what the brake allows instead of leaning on it
    B_robust_pen1 same with a 5x smaller penalty (1 x ... / 5 %): sensitivity check

TD3 settings are Fricz 2026 Table 1 (learning rate 1e-4, tau 1e-3, gamma 0.99, memory 1e5,
batch 512, 2 x 256 neurons). Fricz trained 10,000 episodes; we use 150,000 steps
(about 600 batches) per run so the full set fits on a laptop. Like Fricz, the first 7.5 % of
practice batches start with the valve at 50 %.

    python r2_train.py --list
    python r2_train.py --all                 # 4 set-ups x 3 seeds (about 1-2 h)
    python r2_train.py --all --quick         # 3,000 steps each: checks that everything works
    python r2_train.py --config B_robust --seed 0
    python r2_train.py --config B_robust_pen --seeds 0-4 --parallel 5
    python r2_train.py --all --seeds 0-4     # 5 seeds (finished runs are skipped)
    python r2_train.py --config B_cool_A B_cool_B --seeds 0-4 --parallel 5   # Step 7d coolant extension

Output: results/r2/agents/  (same file layout as Step 5)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from r2_env import R2DosingEnv
from train_agents import BatchLogger

OUT = os.path.join("results", "r2", "agents")
SEEDS = (0, 1, 2)
TOTAL_STEPS = 150_000
PRETRAIN_EPISODES = 45      # 7.5 % of ~600 batches (Fricz: 750 of 10,000)
LAMBDA = 1000.0             # Fricz: 750-1000 prevented overshoot in their cooling-failure test

CONFIGS = {
    "A_nominal": dict(safety="penalty", randomize_plant=False),
    "A_random":  dict(safety="penalty", randomize_plant=True),
    "B_robust":  dict(safety="shield", shield_type="robust", randomize_plant=True),
    "B_nominal": dict(safety="shield", shield_type="nominal", randomize_plant=False),
    "B_robust_pen": dict(safety="shield", shield_type="robust", randomize_plant=True, override_penalty=5.0),
    # Step 7c: gentler override penalty (sensitivity check; mu = 5 made 2 of 5 seeds timid)
    "B_robust_pen1": dict(safety="shield", shield_type="robust", randomize_plant=True, override_penalty=1.0),
}

# Step 7d (extension, not part of --all): the AI also moves the coolant valve. Same as the final
# method B_robust_pen1, plus coolant control, with the brake's two emergency rules:
#   A: the backup opens the coolant fully (100 %);  B: the backup keeps the coolant where it is.
COOLANT_CONFIGS = {
    "B_cool_A": dict(safety="shield", shield_type="robust", randomize_plant=True, override_penalty=1.0,
                     coolant_control=True, backup_OPJ=100.0),
    "B_cool_B": dict(safety="shield", shield_type="robust", randomize_plant=True, override_penalty=1.0,
                     coolant_control=True, backup_OPJ="hold"),
}
#   Step 7m: same two set-ups plus the finish-the-batch reward (r2_env docstring), after the first
#   coolant agents learned to stall
COOLANT_CONFIGS["B_coolF_A"] = dict(COOLANT_CONFIGS["B_cool_A"], finish_reward=True)
COOLANT_CONFIGS["B_coolF_B"] = dict(COOLANT_CONFIGS["B_cool_B"], finish_reward=True)
ALL_CONFIGS = {**CONFIGS, **COOLANT_CONFIGS}

TD3_SETTINGS = dict(            # Fricz et al. 2026, Table 1
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=2_000,
    batch_size=512,
    gamma=0.99,
    tau=1e-3,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)
EXPLORATION_NOISE = 0.2


def make_env(config: str, pretrain: bool = True) -> R2DosingEnv:
    return R2DosingEnv(lam=LAMBDA, pretrain_episodes=PRETRAIN_EPISODES if pretrain else 0,
                       **ALL_CONFIGS[config])


def train(config: str, seed: int, total_steps: int, tag: str = "", quiet: bool = False,
          out: str = OUT) -> dict:
    os.makedirs(out, exist_ok=True)
    name = f"{config}_seed{seed}{tag}"
    model_path = os.path.join(out, name + ".zip")
    if os.path.exists(model_path):
        print(f"[skip] {name} already trained ({model_path})")
        return {}
    torch.set_num_threads(1)
    env = make_env(config)
    n_act = env.action_space.shape[0]
    noise = NormalActionNoise(mean=np.zeros(n_act), sigma=EXPLORATION_NOISE * np.ones(n_act))
    model = TD3("MlpPolicy", env, action_noise=noise, seed=seed, device="cpu", verbose=0, **TD3_SETTINGS)
    print(f"[train] reactor 2 {name}: {total_steps:,} steps ({ALL_CONFIGS[config]})", flush=True)
    t0 = time.time()
    ckpt = CheckpointCallback(save_freq=25_000, save_path=os.path.join(out, "checkpoints"),
                              name_prefix=name, save_replay_buffer=False)
    logger = BatchLogger(os.path.join(out, name + "_train.csv"), print_every=1_000_000 if quiet else 10)
    model.learn(total_timesteps=total_steps, callback=CallbackList([logger, ckpt]))
    model.save(model_path)
    minutes = (time.time() - t0) / 60
    rows = logger.recent
    n_over = sum(r["overheated"] for r in rows)
    print(f"[done]  reactor 2 {name} in {minutes:.1f} min: {len(rows)} practice batches, "
          f"{n_over} overheated while practising -> {model_path}", flush=True)
    meta = dict(reactor=2, config=config, env=ALL_CONFIGS[config], lam=LAMBDA, seed=seed,
                total_steps=total_steps, algorithm="TD3",
                settings={k: str(v) for k, v in TD3_SETTINGS.items()},
                exploration_noise=EXPLORATION_NOISE, pretrain_episodes=PRETRAIN_EPISODES,
                minutes=round(minutes, 2), practice_batches=len(rows), practice_overheated=n_over,
                torch=torch.__version__)
    with open(os.path.join(out, name + "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", choices=list(ALL_CONFIGS),
                    help="one or more set-ups, e.g. --config B_cool_A B_cool_B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="seed range, e.g. 0-4 (default for --all: 0-2)")
    ap.add_argument("--out", default=OUT, help="output folder (default results/r2/agents)")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3,000 steps, saved with a _quick tag")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parallel", type=int, default=0,
                    help="runs at the same time (1 CPU core each). Default with --all: half the CPUs, at most 6")
    a = ap.parse_args()
    steps, tag = (3_000, "_quick") if a.quick else (a.steps, "")
    if a.list or not (a.all or a.config):
        print("Reactor 2 set-ups:")
        for k, v in ALL_CONFIGS.items():
            print(f"  {k:10s} {v}")
        print(f"Full Step 7 training = {len(CONFIGS)} set-ups x {len(SEEDS)} seeds x {TOTAL_STEPS:,} steps")
        return
    if a.seeds:
        lo, _, hi = a.seeds.partition("-")
        seeds = list(range(int(lo), int(hi or lo) + 1))
    else:
        seeds = list(SEEDS) if a.all else [a.seed]
    configs = list(CONFIGS) if a.all else list(a.config)
    jobs = [(c, s) for c in configs for s in seeds]
    t0 = time.time()
    n_par = a.parallel or (max(1, min(6, (os.cpu_count() or 2) // 2)) if len(jobs) > 1 else 1)
    if n_par > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Running {len(jobs)} runs, {n_par} at a time. A line appears as each run starts and finishes.",
              flush=True)
        with ProcessPoolExecutor(max_workers=n_par) as pool:
            for f in [pool.submit(train, c, s, steps, tag, True, a.out) for c, s in jobs]:
                f.result()
    else:
        for i, (c, s) in enumerate(jobs, 1):
            print(f"=== run {i}/{len(jobs)} ===")
            train(c, s, steps, tag, out=a.out)
    print(f"All {len(jobs)} runs finished in {(time.time() - t0) / 60:.0f} min.")


if __name__ == "__main__":
    main()
