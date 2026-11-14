"""
Step 4: training ground for the AI driver (a Gymnasium environment).

The AI chooses the feed rate every 10 s. The reactor model from reactor_model.py moves
forward 10 s. The AI gets a reward and sees the new state.

Two kinds of driver use this same environment:
    safety="penalty"  -> Driver A: only TOLD to be careful (reward penalty when the
                         reactor goes above 50 C). Same idea as Fricz et al. 2026.
    safety="shield"   -> Driver B: the same AI, but every move first passes the
                         emergency brake (shield.py). Unsafe moves are reduced.
    safety="none"     -> no penalty, no brake (only for tests).

"Wrong chemistry" training/testing:
    randomize_plant=True gives every episode a different 'true reactor' whose reaction
    speed, reaction heat and cooling are each between x0.8 and x1.2 of the model.
    The AI and the brake still only know the nominal model.

Observation (what the AI sees, all scaled to roughly 0-1):
    0 reactor temperature, 0 at 20 C and 1 at the 50 C limit
    1 temperature above the coolant, divided by 30 K
    2 unreacted anhydride in the reactor / total charge   (assumed measured)
    3 anhydride fed so far / total charge
    4 conversion
    5 safety margin: (50 C - MTSR) / 30 K, MTSR from the NOMINAL model
    6 previous feed rate / pump maximum
    7 elapsed time / 4 h

Action: one number in [-1, 1], mapped to a feed rate between 0 and 20 mL/min.

Reward per 10 s step:
    + 100 x (increase in conversion)        progress; adds up to about +99 per batch
    - 0.1                                   time cost, so faster batches score higher
    - smooth_weight x (change in feed / F_max)^2     discourages on/off pump chatter
    - penalty_weight x (K above 50 C)       overheating penalty (all modes)
    - 100 once, the first time T > 65 C     runaway (recorded; the batch is NOT stopped)

Why a runaway does not end the batch: in trial training, ending the batch at 65 C let the
AI escape the overheating penalty, so a full-speed runaway (-1,484) scored better than a
moderate overheat at 12 mL/min (-4,624). With the batch continuing, more overheating is
always worse. (The charge is finite, so the temperature stays bounded, peak about 74 C.)

Smooth pump (Step 6): ramp_mL_min_per_step limits how fast the feed may INCREASE between
two 10 s steps (decreases are never limited, so the brake can always cut the feed at once).
The limit is applied before the brake, so the brake still checks the final feed.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reactor_model import (IA, IFED, IT, ReactorParams, advance, conversion, initial_state,
                           mtsr)
from shield import Shield, robust_models

RUNAWAY_MARGIN_K = 15.0


class ReactorDosingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, safety: str = "penalty", shield_type: str = "robust",
                 randomize_plant: bool = False, mismatch: float = 0.2,
                 penalty_weight: float = 10.0, smooth_weight: float = 0.5,
                 time_cost: float = 0.1, ramp_mL_min_per_step: float | None = None,
                 params: ReactorParams | None = None):
        super().__init__()
        assert safety in ("penalty", "shield", "none")
        assert shield_type in ("nominal", "robust")
        self.p_nom = params or ReactorParams()
        self.safety, self.shield_type = safety, shield_type
        self.randomize_plant, self.mismatch = randomize_plant, mismatch
        self.penalty_weight = penalty_weight if safety != "none" else 0.0
        self.smooth_weight, self.time_cost = smooth_weight, time_cost
        self.ramp = ramp_mL_min_per_step
        self.shield = None
        if safety == "shield":
            models = [self.p_nom] if shield_type == "nominal" else robust_models(self.p_nom, mismatch)
            self.shield = Shield(models)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(8,), dtype=np.float32)
        self.max_steps = int(round(self.p_nom.t_final_max / self.p_nom.dt_control))
        self.plant = self.p_nom
        self.plant_factors = dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0)

    # ------------------------------------------------------------------ helpers
    def action_to_feed(self, a) -> float:
        a = float(np.clip(np.asarray(a, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        return (a + 1.0) / 2.0 * self.p_nom.F_max_mL_min

    @staticmethod
    def feed_to_action(F: float, F_max: float) -> np.ndarray:
        return np.array([2.0 * F / F_max - 1.0], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        p, x = self.p_nom, self.x
        T0, Tmax = p.T0, p.T_max
        o = np.array([
            (x[IT] - T0) / (Tmax - T0),
            (x[IT] - p.T_jacket) / 30.0,
            x[IA] / p.n_A_total,
            x[IFED] / p.n_A_total,
            conversion(x, p),
            (Tmax - mtsr(x, p)) / 30.0,
            self.F_prev / p.F_max_mL_min,
            self.t / p.t_final_max,
        ], dtype=np.float32)
        return np.clip(o, -5.0, 5.0)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "plant_factors" in options:
            self.plant_factors = dict(options["plant_factors"])
        elif self.randomize_plant:
            lo, hi = 1 - self.mismatch, 1 + self.mismatch
            self.plant_factors = dict(k_factor=float(self.np_random.uniform(lo, hi)),
                                      dH_factor=float(self.np_random.uniform(lo, hi)),
                                      UA_factor=float(self.np_random.uniform(lo, hi)))
        else:
            self.plant_factors = dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0)
        self.plant = self.p_nom.with_mismatch(**self.plant_factors)
        self.x = initial_state(self.plant)
        self.t, self.steps, self.F_prev = 0.0, 0, 0.0
        self.T_peak = self.x[IT]
        self.mtsr_peak = mtsr(self.x, self.plant)   # cooling-failure temperature, TRUE reactor
        self.n_violation_steps, self.n_interventions = 0, 0
        self.ran_away = False
        self.feed_tv = 0.0          # total variation of the feed, sum |dF| (mL/min): chatter
        return self._obs(), {"plant_factors": dict(self.plant_factors)}

    def step(self, action):
        p, plant = self.p_nom, self.plant
        F_prop = self.action_to_feed(action)
        F = F_prop
        if self.ramp is not None:
            F = min(F, self.F_prev + self.ramp)
        F_req = F
        intervened = False
        if self.shield is not None:
            F = self.shield.filter(self.x, F_req)
            intervened = F < F_req - 1e-9
            self.n_interventions += int(intervened)
        remaining_mL = max(p.n_A_total - self.x[IFED], 0.0) / p.mol_per_mL_A
        F = min(F, remaining_mL / (p.dt_control / 60.0))   # never overdose (same as simulate())

        conv_before = conversion(self.x, p)
        self.x = advance(self.x, F, plant, p.dt_control)
        self.t += p.dt_control
        self.steps += 1
        conv = conversion(self.x, p)
        T = self.x[IT]
        self.T_peak = max(self.T_peak, T)
        self.mtsr_peak = max(self.mtsr_peak, mtsr(self.x, plant))
        over = max(0.0, T - p.T_max)
        if over > 1e-6:
            self.n_violation_steps += 1

        reward = 100.0 * (conv - conv_before) - self.time_cost
        reward -= self.smooth_weight * ((F - self.F_prev) / p.F_max_mL_min) ** 2
        reward -= self.penalty_weight * over
        self.feed_tv += abs(F - self.F_prev)
        self.F_prev = F

        terminated = conv >= p.conversion_target
        if T > p.T_max + RUNAWAY_MARGIN_K and not self.ran_away:
            self.ran_away = True
            reward -= 100.0
        runaway = self.ran_away
        truncated = (not terminated) and self.steps >= self.max_steps

        info = dict(F_proposed=F_prop, F_applied=F, shield_intervened=intervened,
                    T_C=T - 273.15, T_peak_C=self.T_peak - 273.15, conversion=conv,
                    mtsr_peak_C=self.mtsr_peak - 273.15,
                    violated=self.n_violation_steps > 0, runaway=runaway,
                    batch_time_min=(self.t / 60.0) if (conv >= p.conversion_target) else None,
                    plant_factors=dict(self.plant_factors),
                    n_interventions=self.n_interventions, feed_tv=self.feed_tv)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


def run_episode(env: ReactorDosingEnv, policy, options=None, seed=None):
    """Run one batch. policy(obs, env) -> action. Returns a summary dict."""
    obs, info = env.reset(seed=seed, options=options)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        total += r
        done = term or trunc
    return dict(return_=total, batch_time_min=info["batch_time_min"], T_peak_C=info["T_peak_C"],
                mtsr_peak_C=info["mtsr_peak_C"],
                violated=info["violated"], runaway=info["runaway"],
                interventions=env.n_interventions, feed_tv=env.feed_tv,
                plant_factors=info["plant_factors"])
