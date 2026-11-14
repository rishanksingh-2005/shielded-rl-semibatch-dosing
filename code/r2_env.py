"""
Step 7: training ground for the second reactor, set up like Fricz et al. 2026 so the results
can be compared with theirs.

The AI moves the feed valve every 50 s by -5 % to +5 % (Fricz: same action and limits).
Task (Fricz): reach and hold the temperature setpoint 363 K while feeding 5 kmol of A.
The batch (feeding phase) ends when all A has been fed, or after 4 h.

Reward per 50 s step (Fricz Eqs. 16-17):
    +10                       if |T - 363 K| <= delta
    -|T - 363 K|              otherwise
    - lambda * MDC            only when MDC > 0 (runaway potential), lambda = 1000
MDC is computed from the measured state with the NOMINAL model constants (what the designer
knows). delta = 0.5 K is our reading of Fricz Fig. 4 (ASSUMPTION; the text gives no value).

Drivers:
    safety="penalty"  Driver A = Fricz's AI: the MDC penalty only.
    safety="shield"   Driver B = the same AI and reward; every valve move passes the brake
                      (r2_shield.py). The brake may shut the valve faster than 5 %/step.

Observation (Fricz Eq. 6, min-max scaled): T, TJ, valve opening, cA, cB, cC, setpoint error
now and one step ago, T - TJ.

Override penalty (Step 7b, our addition; default off): with override_penalty = mu > 0, every
brake override costs  mu * (OP_requested - OP_applied) / 5 %  in that step's reward. Why: when the
brake cuts a request, every request above the cut gives the same next state, so TD3's critic
sees a flat region there and nothing pulls the AI back; one Step 7 seed drifted to 'always
+5 %' and let the brake drive. The penalty slopes that flat region towards what the brake allows.

Coolant valve as a second control (Step 7d, our extension; default off): with coolant_control=True
the action has two numbers, feed-valve change and coolant-valve change (each -5 % to +5 % per 50 s),
the observation gets the coolant opening as a 10th entry, and the coolant starts at 80 %. The brake
checks both valves (r2_shield.R2Shield.filter2) with backup_OPJ = 100.0 (option A, emergency full
cooling) or "hold" (option B). A coolant opening forced up by the brake counts as an override too.
After feeding ends the coolant is opened fully (operators cool the batch down); with coolant_control
off the coolant stays at 80 % as in Step 7. This set-up is no longer the exact Fricz benchmark
(Fricz keep the coolant fixed), so its results are reported as an extension.

Finish-the-batch reward (Step 7m, our addition; default off): with finish_reward=True,
    finishing the feed at step n earns  +10 * (max_steps - n)  (as if on target for the rest of the 4 h), and
    reaching 4 h unfinished ends the episode as TERMINAL (no bootstrapping) with  -10 * max_steps * (unfed fraction).
Why: the first two-valve agents learned to stall (feed a trickle, shut the coolant, sit on the setpoint) because the
benchmark reward pays only for tracking and the 4 h limit was a truncation, so a never-ending in-band batch was
worth more than finishing. With the bonus, stalling on target can at best tie with finishing (and loses to it
after discounting); the penalty makes any unfinished batch strictly worse.

Safety bookkeeping (reported, not rewarded): 'overheated' = T > MAT = 373 K at any time,
INCLUDING one hour after feeding ends with the valve shut and the coolant on (Fricz stops the
simulation when feeding ends; unreacted A left at that moment can still overheat the reactor).

Cooling-failure test (Fricz Sec. 3.2), reset(options={"cooling_failure_s": t}): from time t the
coolant valve is at 0 %. The driver keeps operating for 2 more steps (100 s; Fricz: 90 s), then
the feed is shut (operators react) and the batch ends; the reactor is then followed for 2 h with
no feed and no cooling, and the peak temperature is recorded.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from r2_model import (IA, IB, IC, IFED, IT, ITJ, IV, R2Params, advance, feeding_done,
                      initial_state, max_OP_for_remaining, mdc, mtsr)
from r2_shield import R2Shield, robust_models

TAIL_STEPS = 72          # 1 h after feeding ends (valve shut, coolant on)
CF_REACT_STEPS = 2       # cooling-failure test: steps before the feed is shut (100 s)
CF_TAIL_STEPS = 144      # then 2 h with no feed and no cooling
RUNAWAY_MARGIN_K = 15.0
R_BAND = 10.0            # reward per step on target (Fricz)


class R2DosingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, safety: str = "penalty", shield_type: str = "robust",
                 randomize_plant: bool = False, mismatch: float = 0.2,
                 lam: float = 1000.0, delta: float = 0.5, max_step_OP: float = 5.0,
                 pretrain_episodes: int = 0, params: R2Params | None = None,
                 override_penalty: float = 0.0, coolant_control: bool = False,
                 backup_OPJ: float | str | None = None, finish_reward: bool = False,
                 cf_aware: bool | str = False):
        super().__init__()
        assert safety in ("penalty", "shield", "none")
        self.override_penalty = override_penalty
        self.finish_reward = finish_reward
        self.coolant_control = coolant_control
        self.p_nom = params or R2Params()
        self.safety, self.shield_type = safety, shield_type
        self.randomize_plant, self.mismatch = randomize_plant, mismatch
        self.lam = lam if safety != "none" else 0.0
        self.delta, self.max_step_OP = delta, max_step_OP
        # Fricz: the first 750 of 10,000 training episodes start with the valve at 50 %, so the AI
        # also sees the effect of closing the valve. pretrain_episodes does the same.
        self.pretrain_episodes, self.episodes = pretrain_episodes, 0
        self.shield = None
        if safety == "shield":
            models = [self.p_nom] if shield_type == "nominal" else robust_models(self.p_nom, mismatch)
            # cf_aware (Step 7o delay-aware brake): False, "robust" (corner models) or "nominal" (design model)
            assert cf_aware in (False, "robust", "nominal")
            self.shield = R2Shield(models, backup_OPJ=backup_OPJ,
                                   cf_delay_steps=CF_REACT_STEPS if cf_aware else 0, cf_tail_steps=CF_TAIL_STEPS,
                                   cf_models=[self.p_nom] if cf_aware == "nominal" else None)
        n_act, n_obs = (2, 10) if coolant_control else (1, 9)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(n_obs,), dtype=np.float32)
        self.max_steps = int(round(self.p_nom.t_final_max / self.p_nom.dt_control))
        self.plant = self.p_nom
        self.plant_factors = dict(k_factor=1.0, dH_factor=1.0, UA_factor=1.0)

    def _obs(self) -> np.ndarray:
        s, p = self.s, self.p_nom
        V = s[IV]
        o = np.array([(s[IT] - 300.0) / 100.0, (s[ITJ] - 300.0) / 100.0, self.OP / 100.0,
                      s[IA] / V / 5.0, s[IB] / V / 10.0, s[IC] / V / 5.0,
                      self.e / 50.0, self.e_prev / 50.0, (s[IT] - s[ITJ]) / 100.0], dtype=np.float32)
        if self.coolant_control:
            o = np.append(o, np.float32(self.OPJ / 100.0))
        return np.clip(o, -5.0, 5.0)

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
        self.cf_s = options.get("cooling_failure_s")
        self.s = initial_state(self.plant)
        self.OP = 50.0 if self.episodes < self.pretrain_episodes else 0.0
        self.OPJ = self.p_nom.OPJ                # 80 % (fixed unless coolant_control)
        self.coolant_tv = 0.0
        self.episodes += 1
        self.t, self.steps = 0.0, 0
        self.e = self.e_prev = self.p_nom.T_sp - self.s[IT]
        self.T_peak, self.mtsr_peak = self.s[IT], mtsr(self.s, self.plant)
        self.mdc_max, self.mdc_pos_s, self.iae = -1e9, 0.0, 0.0
        self.time_to_sp = None
        self.n_violation_steps = self.n_interventions = 0
        self.ran_away, self.feed_tv, self.tail_peak = False, 0.0, None
        return self._obs(), {"plant_factors": dict(self.plant_factors)}

    def step(self, action):
        p, plant = self.p_nom, self.plant
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        a = float(act[0])
        OP_req = min(max(self.OP + self.max_step_OP * a, 0.0), 100.0)
        OP, OPJ_req = OP_req, self.OPJ
        if self.coolant_control:
            OPJ_req = min(max(self.OPJ + self.max_step_OP * float(act[1]), 0.0), 100.0)
        OPJ = OPJ_req
        intervened, override = False, 0.0
        if self.shield is not None:
            if self.coolant_control:
                OP, OPJ = self.shield.filter2(self.s, OP_req, OPJ_req)
                intervened = OP < OP_req - 1e-9 or OPJ > OPJ_req + 1e-9
                override = (max(OP_req - OP, 0.0) + max(OPJ - OPJ_req, 0.0)) if intervened else 0.0
            else:
                OP = self.shield.filter(self.s, OP_req)
                intervened = OP < OP_req - 1e-9
                override = OP_req - OP if intervened else 0.0   # brake cut only (not the end-of-feed cap)
            self.n_interventions += int(intervened)
        OP = min(OP, max_OP_for_remaining(self.s, p))
        self.feed_tv += abs(OP - self.OP)
        self.coolant_tv += abs(OPJ - self.OPJ)
        self.OP, self.OPJ = OP, OPJ

        cooling_failed = self.cf_s is not None and self.t >= self.cf_s - 1e-9
        self.s = advance(self.s, OP, plant,
                         OPJ=0.0 if cooling_failed else (OPJ if self.coolant_control else None))
        self.t += p.dt_control
        self.steps += 1
        T = self.s[IT]
        self.e_prev, self.e = self.e, p.T_sp - T
        self._track(self.s)
        m = mdc(self.s, p)                       # designer's model constants
        self.mdc_max = max(self.mdc_max, m)
        if m > 0:
            self.mdc_pos_s += p.dt_control
        self.iae += abs(self.e) * p.dt_control
        if self.time_to_sp is None and abs(self.e) <= self.delta:
            self.time_to_sp = self.t

        reward = R_BAND if abs(self.e) <= self.delta else -abs(self.e)
        if m > 0:
            reward -= self.lam * m
        if self.override_penalty > 0 and override > 0:
            reward -= self.override_penalty * override / self.max_step_OP

        terminated = feeding_done(self.s, p)
        truncated = (not terminated) and self.steps >= self.max_steps
        finish_term = 0.0
        if self.finish_reward and not cooling_failed:
            if terminated:
                finish_term = R_BAND * (self.max_steps - self.steps)
            elif truncated:                      # the 4 h limit is a real end of the task here
                terminated, truncated = True, False
                unfed = max(p.nA_feed - self.s[IFED], 0.0) / p.nA_feed
                finish_term = -R_BAND * self.max_steps * unfed
            reward += finish_term
        if cooling_failed and self.t >= self.cf_s + CF_REACT_STEPS * p.dt_control - 1e-9:
            y, pk = self.s, T                    # feed shut, still no cooling
            for _ in range(CF_TAIL_STEPS):
                y = advance(y, 0.0, plant, OPJ=0.0)
                pk = max(pk, y[IT])
                self._track(y)
            self.tail_peak = pk
            terminated, truncated = True, False
        elif terminated or truncated:
            y, pk = self.s, T
            tail_OPJ = 100.0 if self.coolant_control else None   # Step 7: coolant stays at 80 %
            for _ in range(TAIL_STEPS):          # valve shut, coolant on, true reactor
                y = advance(y, 0.0, plant, OPJ=tail_OPJ)
                pk = max(pk, y[IT])
                self._track(y)
            self.tail_peak = pk

        info = dict(OP_requested=OP_req, OP_applied=OP, shield_intervened=intervened, override=override,
                    T_K=T, T_peak_C=self.T_peak - 273.15, T_peak_K=self.T_peak,
                    mtsr_peak_C=self.mtsr_peak - 273.15, mdc=m, mdc_max=self.mdc_max,
                    mdc_pos_s=self.mdc_pos_s, iae=self.iae, time_to_sp_s=self.time_to_sp,
                    violated=self.n_violation_steps > 0, runaway=self.ran_away,
                    batch_time_min=(self.t / 60.0) if feeding_done(self.s, p) else None,
                    tail_peak_K=self.tail_peak, plant_factors=dict(self.plant_factors),
                    n_interventions=self.n_interventions, feed_tv=self.feed_tv,
                    OPJ_requested=OPJ_req, OPJ_applied=OPJ, coolant_tv=self.coolant_tv,
                    finish_term=finish_term)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _track(self, s):
        T = s[IT]
        self.T_peak = max(self.T_peak, T)
        self.mtsr_peak = max(self.mtsr_peak, mtsr(s, self.plant))
        if T > self.p_nom.MAT + 1e-6:
            self.n_violation_steps += 1
        if T > self.p_nom.MAT + RUNAWAY_MARGIN_K:
            self.ran_away = True
