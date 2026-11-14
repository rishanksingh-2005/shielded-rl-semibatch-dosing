# Safe, self-learning reagent dosing: code (Steps 1-8)

Simulation-only study. Not a design for a real plant.

## Easy steps: what to do so far

### A. One-time setup (already done on your laptop)
1. Python 3.12 is installed (PyTorch does not support 3.15 yet).
2. The project folder has a `venv` with all libraries installed.
   If you ever need to rebuild it: `py -3.12 -m venv venv`, then `venv\Scripts\activate`,
   then `pip install -r code\requirements.txt`.

### B. Every time you open the project
1. Open `D:\PROJECT CHEMICAL ENGINEERING` in VS Code.
2. Open the terminal: press `` Ctrl+` ``.
3. Turn on the environment: `venv\Scripts\activate` (you should see `(venv)` at the start of the line).
4. Go into the code folder: `cd code`

### C. Run the steps in order
| Step | Command | Time | What you should see |
|---|---|---|---|
| 2. Check the reactor physics | `python test_reactor_model.py` | ~5 s | `All 8 physics tests passed.` |
| 3. Normal controllers on 27 test reactors | `python run_baselines.py` | ~5 min | Tables printed; new files in `results/` |
| 3b. Fastest possible batch (optional) | `python optimal_control_benchmark.py` | ~5 min | 99 % conversion first reached between 31 and 32 min |
| 4. Check the AI training ground | `python check_env_setup.py` | ~1-2 min | `ALL CHECKS PASSED` (done 10 Nov) |
| **5a. Quick test of the brake speed-up** | `python fast_model.py` | ~5 s | `fast_model: identical to reactor_model to 1e-9` |
| **5b. Train both AI drivers** | `python train_agents.py --all` | about 1-3 h | 12 lines starting `[done]`, then `All 12 runs finished` |
| **5c. Test them against the normal controllers** | `python evaluate_agents.py` | ~10-20 min | Three tables; new files in `results/` |
| **5d. Learning curves (optional)** | `python evaluate_agents.py --checkpoints` | ~10 min | one line per checkpoint |
| **6. Brake stress test, 1,000+ random reactors** | `python validate_shield.py` | ~20-40 min | 8 lines; the `robust brake ... +/-20%` lines must say `overheated 0/500` |
| **7a. Second reactor: physics checks** | `python r2_test_model.py` | ~20 s | `All 7 reactor-2 physics tests passed.` |
| **7b. Second reactor: normal controllers** | `python r2_baselines.py` | ~7 min | Tables; `NOTE: no PI gain pair ... was safe` is expected |
| **7c. Second reactor: train both AI drivers** | `python r2_train.py --all` | about 3-5 h (Driver A ~70 min per run) | 12 lines starting `[done]  reactor 2`, then `All 12 runs finished` |
| **7d. Second reactor: test them** | `python r2_evaluate.py` | about 1-2 h | Three tables + a cooling-failure list; new files in `results/r2/` |
| **7e. Driver B with override penalty, 5 seeds** | `python r2_train.py --config B_robust_pen --seeds 0-4 --parallel 5` | about 2 h | 5 lines starting `[done]  reactor 2 B_robust_pen` |
| **7f. Seeds 3 and 4 for the other drivers** | `python r2_train.py --all --seeds 0-4` | about 3 h | 8 new `[done]` lines (finished runs are skipped) |
| **7g. Test everything again** | `python r2_evaluate.py` | about 2-3 h | same as 7d, now 5 set-ups x 5 seeds (done 13 Nov) |
| **5e. Reactor 1: seeds 3 and 4** | `python train_agents.py --all --seeds 0-4` | about 1-2 h | 8 new `[done]` lines |
| **5f. Reactor 1: test again** | `python evaluate_agents.py` | ~20-30 min | same as 5c, now 5 seeds |
| **7h. Reactor 2: gentler penalty, 5 seeds** | `python r2_train.py --config B_robust_pen1 --seeds 0-4 --parallel 5` | about 1.5 h | 5 lines `[done]  reactor 2 B_robust_pen1` |
| **7i. Reactor 2: test again** | `python r2_evaluate.py` | about 3 h | same as 7g, plus `B_robust_pen1` (5e-7i done 13 Nov) |
| **8. Combined comparison, both reactors** | `python step8_compare.py` | ~10 s | tables + 3 figures in `results/step8/`. **Not run by you.** Run once on this laptop during development (13 Nov); the files in `results/step8/` are the tables and figures in the paper. It only reads saved results, so rerunning gives the same output |
| **7j. Coolant extension: checks** | `python r2_test_coolant.py` | ~3 min | `All checks passed`. **Not run by you.** Run on this laptop during development (13 Nov; test 6 added and all 6 checks rerun for Step 7o). Rerun only if you want to see the checks pass yourself |
| **7k. Coolant extension: train A and B, 5 seeds** | `python r2_train.py --config B_cool_A B_cool_B --seeds 0-4 --parallel 5` | about 3-4 h | 10 lines starting `[done]  reactor 2 B_cool_` |
| **7l. Coolant extension: tune classic coolant PI + test everything** | `python r2_coolant.py` | about 1 h | a summary table; files in `results/r2/coolant/` (7k-7l done 13 Nov; see "Step 7d results") |
| **7m. Coolant extension with the finish-the-batch reward, 5 seeds** | `python r2_train.py --config B_coolF_A B_coolF_B --seeds 0-4 --parallel 5` | about 3-4 h | 10 lines starting `[done]  reactor 2 B_coolF_` |
| **7n. Test again (re-tests everything, PI tuning is reused)** | `python r2_coolant.py` | about 1-1.5 h | the summary table now also has `B_coolF_A` and `B_coolF_B` |
| **7o. Delay-aware brake, quick test (classic methods, grid)** | `python r2_cf_aware.py --quick --variant nominal` (and `--variant robust`) | ~15-20 min each | a summary table. **Not run by you.** Both variants were run once on this laptop during development (13 Nov); results are in `results/r2/cf_aware_nominal_quick/` and `results/r2/cf_aware_robust_quick/` and are the numbers in the paper. Rerun only if you want to reproduce them yourself |
| *7p. Optional: delay-aware brake with all agents, 127 reactors* | `python r2_cf_aware.py --variant nominal` | about 3-4 h (~20 s per batch) | **Not run** (by anyone). Optional; see "Step 7o" for why. The paper says the agents were not re-tested with this brake |

Steps 5b and 6 are long. You can leave the laptop running (plugged in, sleep turned off),
and you can run Step 6 at a different time from Step 5b. If 5b is stopped (laptop closed,
Ctrl+C), just run the same command again: finished runs are skipped.

**After Steps 5c and 6, keep both printed outputs.** The same results are saved in
`results/agent_eval_summary.json` and `results/shield_validation.json`.

**Step 7 (second reactor): done (12 Nov).** Results are in the section "Step 7 final results" below.

7e-7i and 5e-5f: done (13 Nov). Step 8: run during development, not by you. Results: section "Step 8: final combined comparison" below.
Coolant extension: 7k-7n done (13 Nov). Steps 7j and 7o (quick test) were run during development, not by you; Step 7p was not run.
Neither is required: nothing is left to run on the laptop. Step 9 (paper): draft v1.2 in `../paper/Paper_draft.md`
and `Paper_draft.docx` (14 Nov), with every reference checked against Crossref/arXiv and a source register in Appendix B.

### D. If something goes wrong
- `No module named ...` (for example `numpy`): VS Code is using the wrong Python (for example 3.15).
  Fix: `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose `venv` (Python 3.12), then open a
  new terminal and check that the line starts with `(venv)`. Run scripts from the terminal, not the
  Run button.
- `can't open file ...`: you are not inside the code folder. Run `cd code`.
- "running scripts is disabled" when activating: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer `Y`.
- Training seems stuck: it is not. Each run prints `[train]` when it starts and `[done]` when it ends,
  and in between there is no output when several runs train at the same time. The files in
  `results/agents/` (`*_train.csv`) grow while it works.
- Laptop gets hot or slow during 5b: use fewer runs at once, for example `python train_agents.py --all --parallel 2`
  (same results, just slower).
- Any other error: copy the full message exactly (including the last lines of the traceback) and keep it with the run log.

## Files
| File | What it is |
|---|---|
| `reactor_model.py` | Part 1: semi-batch reactor (mass + energy balances, Arrhenius kinetics, jacket cooling, MTSR). Every parameter has a source or is marked ASSUMPTION. |
| `controllers.py` | Part 4 baselines: fixed-timing dosing and PID temperature-controlled dosing, with nominal and robust tuning. |
| `shield.py` | Part 3 prototype: look-ahead safety shield (built early to measure how much room the AI has). |
| `test_reactor_model.py` | Physics verification. |
| `run_baselines.py` | Runs all baselines on the nominal model and on 27 mismatched "true plants". |
| `reactor_env.py` | Step 4: training ground for the AI (Gymnasium environment). Driver A = penalty only; Driver B = with the brake; optional wrong-chemistry reactors. |
| `check_env_setup.py` | Step 4 check to run on the laptop (libraries, Gymnasium rules, same results as the simulator, brake safety, tiny training run). |
| `fast_model.py` | Step 5: the same reactor equations on plain floats, used only inside the brake. Gives bit-identical brake decisions, 5-10x faster. Self-test included. |
| `train_agents.py` | Step 5: trains Driver A (penalty only) and Driver B (with brake), 4 set-ups x 3 seeds, TD3. |
| `evaluate_agents.py` | Step 5: tests every trained AI and the normal controllers on 1 + 27 + 100 test reactors. |
| `validate_shield.py` | Step 6: stress test of the brake on 500 random reactors per case, inside (+/-20 %) and outside (+/-30 %) its design range. |
| `optimal_control_benchmark.py` | Independent optimal-control solution: the fastest batch any controller could achieve (~5 min to run). |
| `r2_model.py` | Step 7: second reactor, the Kummer/Fricz benchmark (A + B -> C, jacket dynamics, MDC and MTSR). |
| `r2_test_model.py` | Step 7: 7 physics checks, including reproducing Fricz 2026 Fig. 4. |
| `r2_shield.py` | Step 7: the same brake rule for reactor 2 (acts on the feed valve; jacket simulated). |
| `r2_env.py` | Step 7: training ground with Fricz's task and reward (+ cooling-failure test). |
| `r2_baselines.py` | Step 7: PI tuning, brake alone, oracle brake. |
| `r2_train.py` / `r2_evaluate.py` | Step 7: train and test both drivers on reactor 2 (Fricz TD3 settings). |
| `r2_cf_aware.py` | Step 7o: re-tests methods with the delay-aware brake (cooling failure + 100 s operator delay). |
| `r2_coolant.py` / `r2_test_coolant.py` | Step 7d extension: the AI also moves the coolant valve; brake emergency rule A (coolant opens fully) vs B (coolant stays); classic coolant-PI baselines; checks. |
| `step8_compare.py` | Step 8: one combined comparison of both reactors (tables, confidence intervals, significance tests, 3 figures) from the finished results. |
| `results/` | `summary.json`, `mismatch_sweep.csv`, figures; after Steps 5-6 also `agents/`, `agent_eval*`, `shield_validation*`. |

## The model
Acetic anhydride is dosed into water in a 1 L jacketed reactor: (CH3CO)2O + H2O -> 2 CH3COOH.

| Quantity | Value | Source |
|---|---|---|
| Rate law | pseudo-first-order in anhydride (water in excess) | as reported by the kinetic source |
| k at 20 C | 0.129 1/min | React. Chem. Eng. 2020, DOI 10.1039/D0RE00304B |
| Ea | 30.9 kJ/mol (derived from that paper's 20 C and 56.3 C rate constants) | cross-check: Magritek NMR note 28.4 kJ/mol |
| dH_r | -63 kJ/mol | same paper; cross-checks -58.2 (Hess), -56.6 (NIST), -60.9 (Syrris) |
| Cp (water / Ac2O / AcOH) | 75.3 / 168.2 / 120.5 J/mol/K | same paper, SI Table S1 |
| Heat of mixing | ignored (conservative: the measured value is endothermic) | same paper reports +8.8 kJ/mol for a 3:1 Ac2O:AcOH feed |
| Charge | 400 mL water, 3.0 mol Ac2O (283 mL) | ASSUMPTION |
| UA | 3 W/K | ASSUMPTION (glass 1 L reactor: 1.5-6 W/K) |
| Coolant, feed, start T | 20 C | ASSUMPTION |
| Pump max | 20 mL/min (full speed is unsafe: peak 74 C) | ASSUMPTION |
| T limit | 50 C, chosen to stay inside the measured kinetic range (20-56.3 C) | design choice |
| Batch end | 99 % of the total charge reacted | design choice |

Source links:
- Kinetics and enthalpy: https://dspace.mit.edu/handle/1721.1/128541 (SI: https://www.rsc.org/suppdata/d0/re/d0re00304b/d0re00304b1.pdf)
- Activation-energy cross-check: https://magritek.com/wp-content/uploads/2020/08/App-Note-Kinetics-1.pdf
- Enthalpy cross-check: https://www.syrris.com/wp-content/uploads/2017/08/Atlas-Tech-Note-PCC-Acetic-anhydride-hydrolysis.pdf
- Note: the two kinetic sources differ about 3x in k at 20 C; Ea agrees (28-31 kJ/mol). The model uses the RCE 2020 source.

## Verification (all pass)
1. Arrhenius reproduces both source rate constants exactly.
2. Isothermal decay matches exp(-kt) to 1e-8.
3. Adiabatic temperature rise matches the analytical solution to 1e-6 K.
4. Mass balance closes at every step (moles reacted, water used, acid made).
5. Energy balance closes to < 0.2 % (heat released = cooling + feed warming + stored).
6. Fixed-step RK4 and SciPy LSODA agree on peak T to < 0.001 K in closed loop.
7. MTSR and volume sanity checks.
8. Full pump speed overheats, so the control problem is non-trivial.

An independent optimal-control solution (SciPy SLSQP, single shooting) puts the
**minimum possible batch time at about 31.8 min** on the nominal model.

## Results
Nominal model (the model is exactly right):

| Controller | Batch time | Peak T |
|---|---|---|
| Fixed timing (tuned) | 37.3 min | 49.98 C |
| PID (tuned) | 33.0 min | 49.57 C |
| Max feed + shield | 31.7 min | 50.00 C |
| Optimal control | ~31.8 min | 50.00 C |

27 mismatched plants (k, dH, UA each x0.8 / x1.0 / x1.2). Controllers are tuned on the nominal model only:

| Controller | Plants overheated | Worst peak | Mean batch | Slower than oracle |
|---|---|---|---|---|
| Fixed timing | 13 / 27 | 63.4 C | 38.0 min | +20 % |
| PID | 4 / 27 | 50.4 C | 34.2 min | +5 % |
| Fixed timing, robust tuned | 0 / 27 | 50.0 C | 56.8 min | +80 % |
| **PID, robust tuned** | **0 / 27** | 49.8 C | **34.4 min** | **+5 %** |
| Max feed + nominal shield | 14 / 27 | 56.6 C | 32.9 min | +1 % |
| Max feed + robust shield | 0 / 27 | 50.0 C | 35.5 min | +9 % |
| Oracle (shield knows true plant) | 0 / 27 | 50.0 C | 32.8 min | 0 |

## What the baseline runs showed (these changed the project design)
1. **With a correct model, the shield alone is already optimal.** "Max feed + shield"
   (31.7 min) matches the optimal-control solution (~31.8 min). An AI driver under that
   shield cannot be faster. Reviewers would ask "why RL?".
2. **A shield built on the wrong model is not safe.** With +/-20 % mismatch, the nominal
   shield overheated 14 of 27 plants (up to 56.6 C).
3. **A robust shield is safe but costs about 9 % batch time.** A well-tuned robust PID
   is safe on all 27 plants and only about 5 % slower than the oracle.
4. **The largest speed-up any controller could claim:** about 5 % against a well-tuned PID,
   but about 40 % against a safe fixed-timing recipe (the usual industrial practice).
5. The nominal shield's feed chatters between 0 and 17 mL/min every 10 s. A real pump
   would need a rate limit or a smoothness penalty.

## Steps 5-6: design decisions (10 Nov 2026)

**What the brake actually checks** (use this wording in the paper). Before every move, for
every model in its set, the brake simulates: "apply this feed for 10 s, then STOP the feed
(cooling still on). Does T stay <= 50 C until the unreacted anhydride is too little to push
T past 50 C even with no cooling (MTSR <= 50 C)?" If not, it lowers the feed (bisection).
So it is a backup-policy shield (Bastani et al. 2021) whose end point is the Stoessel MTSR
criterion. It does NOT keep MTSR itself below 50 C at all times, so it is not a
"cooling-failure" check like Guinand et al. 2014. The worst MTSR on the true reactor is now
reported for every method (`evaluate_agents.py`). MTSR should be judged against the maximum
technical temperature (e.g. boiling point of the mixture), not the 50 C operating limit.

| Decision | Why |
|---|---|
| TD3 for both drivers, identical settings (`train_agents.py`) | Fricz et al. 2026 used TD3/DDPG; same algorithm keeps the A vs B comparison clean |
| 4 set-ups x 3 seeds x 100k steps | Trial runs: B settles by 20k steps; A swings between overheating and over-caution until about 100k (see checkpoints) |
| Runaway (T > 65 C) no longer ends the batch | Bug found in trial runs: ending the batch let Driver A escape the penalty, so a full-speed runaway (return -1,484) scored better than a moderate overheat at 12 mL/min (-4,624). Now more overheating is always worse (full speed: -7,742) |
| Fast brake (`fast_model.py`) | Same arithmetic on plain floats. Brake decisions are bit-identical to the NumPy version (checked on 6 runs); careful brake 5-10x faster |
| Random test reactors use seed 777, never used in training | Test set is separate from the practice reactors |

## Step 6 results (first run 10 Nov; rerun on the laptop with `validate_shield.py`)

Smooth pump. Full-speed request + brake, nominal model. "Feed chatter" = sum of |feed changes| in mL/min:

| Brake | Ramp limit (mL/min per 10 s) | Batch time | Feed chatter |
|---|---|---|---|
| Normal | none | 31.67 min | 1,674 |
| Normal | 2 | 32.50 min | 316 |
| Normal | 1 | 33.83 min | 192 |
| Careful (robust) | none | 33.83 min | 50 |
| Careful (robust) | 2 | 34.50 min | 49 |

The careful brake is already smooth, so it needs no ramp. The ramp (`ramp_mL_min_per_step` in
`reactor_env.py`) only limits increases, so the brake can still cut the feed at once.

Stress test, 500 random reactors per line (k, dH, UA uniform in the range):

| Brake | Chemistry range | Driver | Overheated | Worst peak |
|---|---|---|---|---|
| **Careful** | +/-20 % (design range) | full speed | **0 / 500** | 49.94 C |
| **Careful** | +/-20 % (design range) | random | **0 / 500** | 49.94 C |
| Careful | +/-30 % (outside design) | full speed | 11 / 500 | 52.59 C |
| Careful | +/-30 % (outside design) | random | 7 / 500 | 51.19 C |
| Normal | +/-20 % | full speed | 269 / 500 | 56.52 C |
| Normal | +/-20 % | random | 237 / 500 | 52.54 C |
| Normal | +/-30 % | full speed | 268 / 500 | 59.10 C |
| Normal | +/-30 % | random | 235 / 500 | 53.99 C |

## Step 5 trial results (trial runs: 1 seed each; the full laptop runs replace these)

27 grid reactors, AI tested without exploration noise:

| Method | Overheated | Worst peak | Slower than oracle | Feed chatter |
|---|---|---|---|---|
| PID (robust tuned) | 0 / 27 | 49.8 C | +5.3 % | 56 |
| Max feed + careful brake, no AI | 0 / 27 | 50.0 C | +8.8 % | 51 |
| Driver A (penalty only), 60k steps | 7 / 27 | 52.3 C | +13.9 % | 789 |
| Driver B (AI + careful brake), 60k steps | 0 / 27 | 50.0 C | +9.6 % | 63 |

Checkpoints (seed 1, every 20k steps) on the same 27 reactors:

| Steps | 20k | 40k | 60k | 80k | 100k | 120k |
|---|---|---|---|---|---|---|
| Driver A overheated / slower | 2, +26 % | 1, +35 % | 10, +8 % | 0, +86 % | 0, +16 % | 0, +14 % |
| Driver B overheated / slower | 0, +10 % | 0, +10 % | 0, +9 % | 0, +10 % | 0, +10 % | 0, +10 % |

While practising, Driver A overheated 68 of 513 practice batches (seed 1); Driver B overheated 0 of 570.

What this means so far (to be confirmed with 3 seeds):
1. The brake gives zero overheating both while learning and after; the penalty-only AI does not.
2. The penalty-only AI is unstable: its safety depends on when training is stopped.
3. On top of the careful brake, the AI adds no speed in this reactor (+9.6 % vs +8.8 % for the
   brake alone). This fits finding 1 above and Oh et al. 2025. So the paper's claim has to be about
   safety and reliability, not speed.
4. A well-tuned robust PID is still the fastest safe controller here (+5 %). It is the baseline
   to beat; any claim must be made against it.

## Steps 5c and 6: final laptop results (12 Nov 2026, 3 seeds x 100k steps)

Step 6 on the laptop reproduced the table above exactly (careful brake: 0/500 and 0/500 inside +/-20 %).

Tested without exploration noise. "Slower" = mean batch time vs the oracle. Seeds pooled (3 seeds x reactors):

| Method | 27 grid reactors: overheated | worst peak | slower | 100 random reactors: overheated | slower | Worst MTSR (grid) |
|---|---|---|---|---|---|---|
| PID (robust tuned) | 0 / 27 | 49.8 C | +5.3 % | 0 / 100 | +5.3 % | 68.3 C |
| Max feed + careful brake, no AI | 0 / 27 | 50.0 C | +8.8 % | 0 / 100 | +8.8 % | 63.4 C |
| Driver A, nominal practice | 29 / 81 | 54.6 C | +9.3 % | 75 / 300 | +10.2 % | 71.1 C |
| Driver A, random-chemistry practice | 2 / 81 | 51.1 C | +18 to +34 % | 0 / 300 | +18 to +36 % | 61.2 C |
| Driver B, nominal brake | 36 / 81 | 56.2 C | +3.5 % | 147 / 300 | +3.2 % | 72.5 C |
| **Driver B, careful brake** | **0 / 81** | 50.0 C | **+10.4 %** | **0 / 300** | **+10.4 %** | 63.2 C |

Overheated practice batches during training: Driver A 239 of 2,496 (nominal 117, random 122); Driver B 0 of 2,979.
Checkpoints (every 20k steps, 27 grid reactors): Driver B careful overheated 0 times at all 15 checkpoints;
Driver A nominal overheated at 14 of 15; Driver A random at 6 of 15 (one checkpoint also left 26 of 27 batches unfinished).

What the final runs confirm:
1. Only the careful brake gives zero overheating in practice and in testing, on every seed and checkpoint.
2. Training Driver A on random chemistry makes it much safer in testing (0/300 random, 2/81 grid) but it still
   overheated 122 practice batches, is 2-4x slower than the brake, and its speed depends strongly on the seed.
   This is the strongest "penalty only" baseline and the one the paper should compare against.
3. The AI adds no speed on top of the careful brake (+10.4 % vs +8.8 % for the brake alone). Robust PID (+5.3 %)
   is still the fastest safe method on this reactor. A brake built on the nominal model is fast but unsafe (36/81).
4. The careful brake keeps less unreacted anhydride in the reactor than PID (worst MTSR 63 C vs 68 C).

## Step 7: second reactor (Fricz et al. 2026 benchmark), built 12 Nov 2026

Why: the closest paper (Fricz et al. 2026, Comput. Chem. Eng. 212, 109704) published its reactor.
Running our method on it, with their task, reward and AI settings, makes the results directly comparable.

| Item | Value | Source |
|---|---|---|
| Reaction | A + B -> C, r = k0 exp(-E/RT) cA cB | Fricz 2026, App. A (Eq. A.1-A.9) |
| k0, E/R, dH | 1.465e7 m3/(kmol s), 8,500 K, -3.5e5 kJ/kmol | Fricz Table A.5 |
| UA | 1.85 kW/K x (1 + dosed volume / 0.5 m3) | Fricz Table A.5, Eq. A.7 |
| Charge, feed | 5 kmol B in 0.5 m3; 5 kmol A at 5 kmol/m3, max 1.4e-4 m3/s | Fricz Table A.6 |
| Jacket | 0.41 m3, coolant max 1.1e-3 m3/s, inlet 298 K | Fricz Tables A.5-A.6 |
| Start | reactor 340 K, jacket 298 K, feed 298 K | Fricz Table A.6 |
| Limit | MAT = 373 K | Fricz Table A.5 |
| Setpoint | 363 K | read from Fricz Fig. 4 (and reward -23 at 340 K) |
| Coolant valve while feeding | 80 % | ASSUMPTION from Kummer, Varga, Nagy 2020 (CC BY), same group; it is also the value that reproduces Fricz's peak MDC (50 %: 0.054, 80 %: 0.077, 100 %: 0.088; Fricz NMPC: 0.0772) |
| Reward band delta | 0.5 K | ASSUMPTION read from Fricz Fig. 4 |
| Action | valve change -5 % to +5 % every 50 s | Fricz Sec. 2.2 |
| Reward | +10 if within delta, else -abs(error); minus 1000 x MDC when MDC > 0 | Fricz Eqs. 16-17, lambda in their recommended range |
| TD3 | lr 1e-4, tau 1e-3, gamma 0.99, memory 1e5, batch 512, 2 x 256 | Fricz Table 1 (we train 150k steps, not 10,000 episodes) |

Checks (all pass, `r2_test_model.py`): mass balance (1e-13), energy balance of reactor and jacket
(2e-10), RK4 vs LSODA (identical to 1e-4 K), adiabatic rise, MDC formula = the original criterion,
reproduces Fricz Fig. 4 (dip to 327 K vs ~325 K; peak MDC 0.0771 vs 0.0772), full valve overheats (396 K).

One inconsistency in Fricz 2026: feeding 5 kmol at the maximum flow takes at least 7,143 s, but
Fig. 4 ends at about 7,300 s although the valve is well below 100 % for long periods. Replaying
the Fig. 4 valve profile, feeding ends at about 9,300 s. We keep the published table values.

Our addition: 'overheated' also counts the hour AFTER feeding ends (valve shut, coolant on);
Fricz stops the simulation when feeding ends.

### Step 7 first results (no AI yet; `r2_baselines.py`, 12 Nov)

| Method | Nominal: peak | 27 grid reactors: overheated | worst peak | did not finish in 4 h |
|---|---|---|---|---|
| PI, best tracking (Kp 4, Ti 4000 s) | 380.5 K (over MAT) | 17 / 27 | 418 K | 2 |
| PI, best tracking + careful brake | 359.0 K | 0 / 27 | 369.7 K | 10 |
| Max feed + normal brake | 373.0 K | 15 / 27 | 386 K | 0 |
| **Max feed + careful brake (no AI)** | 370.9 K | **0 / 27** | 373.0 K | 2 |
| Oracle brake (knows the true reactor) | 373.0 K | 0 / 27 | 373.0 K | 0 |

1. **No PI setting is safe on this reactor.** 29 of the 30 gain pairs tried pass MAT even on the
   nominal model (375-420 K); the 30th never heats the reactor up at all. The start-up needs
   A to accumulate and then react (a controlled near-runaway), which a PI cannot judge. This is
   the opposite of reactor 1 (where a robust PID was the best safe method), so here the AI has a real job.
2. A brake built on the nominal model fails again with wrong chemistry (15/27), as on reactor 1.
3. The careful brake is safe on all 27, but on 2 slow, low-heat reactors (k x0.8, dH x0.8) it keeps
   the feed nearly shut: the reactor is too cold to consume A, and the accumulated A could
   push it past MAT if the cooling failed. This is a correct, safe stop, and it is reported as 'unfinished'.
4. None of the methods keeps MTSR below MAT (worst 396-424 K): at 363 K there is only 10 K of
   margin. As on reactor 1, the brake guards 'stop the feed, keep cooling', not total cooling failure.
   The cooling-failure test in `r2_evaluate.py` measures this directly.

### Step 7 final results (laptop, 12 Nov 2026, 3 seeds x 150k steps, Fricz TD3 settings, lambda 1000)

Tested without exploration noise; seeds pooled. "IAE" = tracking error (K s, lower is better), averaged over
the reactors that every method finished (15 grid, 61 random) so that methods are compared on the same batches.
Cooling failure = coolant to 0 % at 600, 1200, ..., 7200 s; the controller keeps going for 100 s, then the feed is shut.

| Method | 27 grid: overheated | worst peak | 100 random: overheated | unfinished (random) | reached SP (random) | IAE grid / random (k) | cooling failure: overheated |
|---|---|---|---|---|---|---|---|
| PI, best tracking | 17 / 27 | 418 K | 71 / 100 | 0 | 95 % | 124 / 113 | 10 / 12 |
| PI + careful brake | 0 / 27 | 370 K | 0 / 100 | 38 | 16 % | 197 / 172 | 0 / 12 |
| Max feed + careful brake, no AI | 0 / 27 | 373 K | 0 / 100 | 2 | 81 % | 119 / 109 | 0 / 12 |
| Oracle brake (knows the true reactor) | 0 / 27 | 373 K | 0 / 100 | 0 | 98 % | 98 / 94 | **3 / 12** |
| Driver A, nominal practice (= Fricz's setup) | 16 / 81 | 404 K | 16 / 300 | 1 | 73 % | 109 / 92 | 0 / 36 |
| Driver A, random-chemistry practice | 15 / 81 | 412 K | 33 / 300 | 10 | 60 % | 153 / 110 | 0 / 36 |
| Driver B, nominal brake | 28 / 81 | 378 K | 45 / 300 | 2 | 90 % | 110 / 95 | 0 / 36 |
| **Driver B, careful brake** | **0 / 81** | 373.0 K | **0 / 300** | 10 | 86 % | **111 / 98** | **0 / 36** (worst 372.3 K) |

Overheated practice batches during training: Driver A nominal 248 of 1,914 (72 of them runaways, > 388 K);
Driver A random 445 of 1,792 (188 runaways); Driver B nominal 0 of 1,875 (it practised on the nominal reactor,
where its brake is exact); **Driver B careful 0 of 1,861**. Training time: B careful 277 min for 3 runs vs 204-237 min for A.

Comparison with Fricz et al. 2026 (nominal reactor, lambda = 1000):

| | max MDC | IAE (K s) | time MDC > 0 (s) | time to SP (s) |
|---|---|---|---|---|
| Fricz TD3 (Table 2) | 0.049 | 48,796 | 88 | 1,735 |
| Fricz NMPC (Table 4) | 0.075 | 85,391 | 1,450 | 2,900 |
| Ours, Driver A nominal (their setup), 3 seeds | 0.011-0.027 | 84,600-91,900 | 900-1,050 | 2,600-3,300 |
| Ours, Driver B careful, 3 seeds | 0.044-0.046 | 89,400-106,600 | 950-1,000 | 2,450-9,450 |
| Ours, oracle brake | 0.092 | 95,500 | 850 | 2,700 |

What Step 7 shows:
1. **Only Driver B with the careful brake is safe everywhere:** 0 of 381 test batches, 0 of 36 cooling-failure
   tests, 0 of 1,861 practice batches. Every other learning method overheated in testing, and Driver A
   overheated in practice (248 and 445 batches). Same result as reactor 1, so it holds on two reactors.
2. **On this reactor the AI is useful, unlike reactor 1.** No PI is safe. With the careful brake, PI barely
   heats the reactor (38 of 100 unfinished, IAE 172k). Driver B careful tracks the setpoint 6-10 % better than
   the careful brake alone (IAE 111k vs 119k grid, 98k vs 109k random) and reaches it more often (86 % vs 81 %).
   It is still 5-13 % behind the oracle brake (which needs the true reactor model) and 2-6 % behind Driver A nominal,
   which is unsafe (16/300).
3. **Random-chemistry practice does not make Driver A safe here.** It was the safest penalty-only variant on reactor 1;
   on reactor 2 it was worse than nominal practice (33/300 vs 16/300). One seed kept overheating throughout training
   (279 practice batches, still rising at the end). Seed-to-seed spread is large (30, 3 and 0 of 100 random reactors).
4. **The brake's safety margin also covers cooling failure; a brake without margin does not.** The oracle brake
   runs right at MAT, so 100 s of feeding without cooling pushes it over (3 of 12). The careful brake stays below
   (worst 372.3 K, only 0.7 K of margin). The penalty-only agents also pass, because they run cooler (MDC 0.01-0.02).
5. **One of three Driver B careful seeds learned nothing useful:** seed 1 opens the valve fully and lets the brake
   do all the work (59,395 brake interventions, results identical to "max feed + careful brake"). It is still safe.
   Seeds 0 and 2 learned real tracking (nominal IAE 89k and 97k vs 106k for the brake alone).
6. **The same 3 grid reactors stay unfinished for every Driver B careful seed** (dH x0.8 with k x0.8 or UA x1.2):
   less heat is made than the 80 % coolant removes, so the reactor stays cold, A would accumulate, and the brake
   stops the feed. Letting the controller also turn the coolant down (Fricz keep it fixed) would fix this.
7. **Our numbers match Fricz's NMPC, not their TD3.** Their TD3 reaches the setpoint at 1,393-1,735 s. In the model
   built from their published tables, the fastest of 820 simple valve profiles (ramp at the allowed rate, hold,
   optional cut-off) reaches it at 2,000 s even if MAT is ignored; their own NMPC needs 2,900 s.
   Together with the feeding-time inconsistency above, this suggests their TD3 simulation differed from the published
   parameters in some way we cannot identify. We report their numbers but compare mainly against their NMPC and our own runs.

### Step 7b: 5 seeds and the override penalty (laptop, 13 Nov 2026)

`B_robust_pen` = Driver B careful + override penalty (5 x cut valve % / 5 % per override; see `r2_env.py`).
All AI set-ups now have 5 seeds (the table above used seeds 0-2). "Overrides" = brake cuts per batch (random reactors).
IAE is averaged over the reactors every method finished (14 grid, 60 random).

| Method (5 seeds) | Grid overheated | Random overheated | Unfinished (random) | IAE grid / random (k) | Overrides | Cooling failure | Practice overheated |
|---|---|---|---|---|---|---|---|
| Careful brake alone, no AI | 0 / 27 | 0 / 100 | 2 / 100 | 118 / 109 | 134 | 0 / 12 | - |
| Oracle brake | 0 / 27 | 0 / 100 | 0 / 100 | 97 / 94 | 18 | 3 / 12 | - |
| Driver A nominal | 25 / 135 | 23 / 500 | 2 / 500 | 111 / 98 | - | 0 / 60 | 635 / 3,161 (323 runaways) |
| Driver A random | 28 / 135 | 59 / 500 | 13 / 500 | 142 / 118 | - | 0 / 60 | 689 / 2,941 (292 runaways) |
| Driver B normal brake | 54 / 135 | 123 / 500 | 3 / 500 | 105 / 93 | 4 | **1 / 60** (374.9 K) | 0 / 3,145 |
| Driver B careful | **0 / 135** | **0 / 500** | 15 / 500 | 107 / 97 | 57 | 0 / 60 (worst 372.3 K) | **0 / 3,113** |
| Driver B careful + override penalty | **0 / 135** | **0 / 500** | 40 / 500 | 113 / 99 | **18** | 0 / 60 (worst **369.6 K**) | **0 / 2,973** |

Per seed (random reactors): overrides per batch and IAE (k).

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| B careful: overrides / IAE / unfinished | 29 / 94 / 3 | **116** / 109 / 4 | 32 / 91 / 3 | **83** / 98 / 2 | 25 / 94 / 3 |
| B careful + penalty: overrides / IAE / unfinished | 8 / 123 / 8 | 21 / 86 / 2 | 16 / **82** / 6 | 16 / 92 / 7 | 26 / 113 / **17** |

What 7b shows:
1. **The penalty fixed the lazy agent.** Without it, 2 of 5 seeds leaned on the brake (116 and 83 overrides per batch,
   close to "max feed + brake" at 134). With it, no seed did (8-26), and overrides fell 3x on average. Training was also
   faster (348 vs 458 min for 5 runs), because fewer overrides means fewer brake bisections.
2. **Safety is unchanged, with more margin:** still 0 overheated in 635 tests and 2,973 practice batches. Worst
   cooling-failure peak 369.6 K vs 372.3 K (3.4 K below MAT instead of 0.7 K): an agent that stays away from the brake's limit
   keeps more margin.
3. **But it made 2 of 5 seeds timid.** Seeds 0 and 4 track worse (IAE 123k and 113k) and leave more batches unfinished
   (8 and 17 of 100). They learned "do not ask for much, so the brake never cuts", and on weaker-than-nominal chemistry they
   under-feed. The other 3 seeds are the best braked controllers we have (IAE 82-92k; seed 2 beats even the oracle brake's 94k).
   On average, tracking is about the same as without the penalty (99k vs 97k).
4. **Seed spread is large for every AI**, which justifies 5 seeds. For example, Driver A random overheated 0-30 of 100 random reactors depending on the seed.
5. **New with 5 seeds:** Driver B with the normal brake failed one cooling-failure test (seed 4, which runs at a high MDC, 0.078),
   and Driver A's practice overheats grew to 635 and 689 batches, with about 300 runaways each.
6. **Penalty size matters (mu = 5 was a reasoned choice, not tuned).** Step 7c tests mu = 1 (`B_robust_pen1`, 5 seeds)
   to see whether a gentler penalty keeps the fix without the timidity.

Why the override penalty is NOT added to reactor 1: there, the task is "finish as fast as possible", and "full feed, let the
brake cut" is already the best possible strategy (31.7 vs 31.8 min optimal). On reactor 1, leaning on the brake is
*correct*, so penalising it would only slow the agent down. The penalty helps when the task (tracking a setpoint) differs
from what the brake would do by itself.

## Step 8: final combined comparison (13 Nov 2026; 5 seeds everywhere; `step8_compare.py`)

Full tables, p-values and per-seed numbers: `results/step8/main_table.md`, `stats.json`, `penalty_sensitivity.csv`.
Figures: `fig_safety.png` (overheated share, test and practice), `fig_tradeoff.png` (performance vs safety per seed),
`fig_penalty.png` (override penalty mu = 0 / 1 / 5). Test = 27 grid + 100 random reactors per seed (635 batches per AI set-up).
"Gap" = reactor 1 batch time vs oracle; reactor 2 tracking error (IAE) vs oracle brake, over the 74 test reactors every method
finished. 95 % interval = Clopper-Pearson (0 of 635 gives at most 0.6 %).

| | Reactor 1: overheated (test) | practice | gap [seed range] | Reactor 2: overheated (test) | practice | gap [seed range] | unfinished | cooling failure |
|---|---|---|---|---|---|---|---|---|
| Classic (R1 robust PID / R2 best PI) | 0 / 127 | - | +5.3 % | 88 / 127 | - | +22.8 % | 2 | 10 / 12 |
| Careful brake, no AI | 0 / 127 | - | +8.8 % | 0 / 127 | - | +18.0 % | 4 | 0 / 12 |
| Oracle brake | 0 / 127 | - | 0 | 0 / 127 | - | 0 | 0 | 3 / 12 |
| Driver A nominal (Fricz-style) | 126 / 635 | 305 / 1,971 | +13.1 % [+9.9, +22.5] | 48 / 635 | 635 / 3,161 | +6.2 % [-6.7, +20.8] | 8 | 0 / 60 |
| Driver A random | 11 / 635 | 190 / 1,983 | +32.6 % [+17.7, +72.0] | 87 / 635 | 689 / 2,941 | +28.8 % [+13.2, +61.8] | 30 | 0 / 60 |
| Driver B normal brake | 306 / 635 | 0 / 2,596 | +3.0 % | 177 / 635 | 0 / 3,145 | +1.1 % | 11 | 1 / 60 |
| Driver B careful brake | **0 / 635** | **0 / 2,376** | +10.1 % [+9.4, +10.8] | **0 / 635** | **0 / 3,113** | +5.2 % [-1.7, +18.5] | 30 | 0 / 60 |
| **Driver B careful + override penalty mu=1 (final method, R2)** | (not used: see 7b) | | | **0 / 635** | **0 / 3,054** | **-4.6 % [-11.0, +1.8]** | 39 | **0 / 60** (worst 371.2 K) |

Override penalty sensitivity (reactor 2): mu = 0 gives 58 brake cuts per batch (2 lazy seeds), gap +5.2 %, 30 unfinished;
**mu = 1 gives 30 cuts, gap -4.6 %, 39 unfinished**; mu = 5 gives 18 cuts, gap +7.5 %, 65 unfinished (2 timid seeds). mu = 1 is the final setting.

Final findings (these are the paper's claims):
1. **Safety.** The AI with the careful brake overheated 0 of 1,905 test batches and 0 of 8,543 practice batches over both reactors
   (5 seeds each; 3 set-ups: reactor 1, reactor 2 without and with the override penalty; mu = 5 was also 0). Every penalty-only set-up overheated in testing (1.7-19.8 % of batches on reactor 1,
   7.6-13.7 % on reactor 2) and while practising (9.6-23.4 %). All differences vs our method: Fisher p < 0.001.
   The normal brake fails under +/-20 % mismatch on both reactors (48 % and 28 % of test batches).
2. **Performance depends on the reactor.**
   - Reactor 1 (easy; a robust PID is safe): the AI adds nothing. Driver B careful (+10.1 %) is no faster than the
     brake alone (+8.8 %), and robust PID is fastest (+5.3 %). All 5 seeds learned the same "full feed, let the brake cut" strategy
     (spread 9.4-10.8 %), which is the right strategy for that task.
   - Reactor 2 (hard; no PI is safe): our final method tracks 4.6 % better than the oracle brake and much better than the
     careful brake alone (+18.0 %): better on 70 of 74 test reactors, Wilcoxon p = 7e-12. Every seed beat the brake alone.
3. **The override penalty is needed when the task differs from the brake's own behaviour.** Without it, 2 of 5 seeds let the brake
   drive. With it (mu = 1), there are no lazy seeds (22-49 cuts vs 26-115) and the seed spread narrows (12.8 vs 20.2 points).
   Too large (mu = 5) makes seeds timid.
4. **Costs of our method (state these):** 39 of 635 reactor-2 test batches unfinished in 4 h (6.1 %, vs 3.1 % for the brake alone),
   mostly on weaker-than-nominal chemistry where the brake refuses to let A accumulate. Training with the careful brake took up to
   about 25 % longer than penalty-only (brake simulations); the final reactor-2 method only about 3 % longer (fewer brake cuts).
5. **Random-chemistry practice is not a reliable fix for penalty-only AI.** Reactor 1: 11/635 overheated (2/381 with 3 seeds;
   seeds 3-4 added the rest) and up to +72 % slower. Reactor 2: worse than nominal practice (87 vs 48 of 635).
6. **Reactor 1, 5 seeds:** one Driver A nominal seed (seed 3) learned to almost stop feeding and never finished a test batch (127/127
   unfinished); the other four overheated 17-29 % of test batches. Seed-to-seed reliability is itself a weakness of penalty-only RL.

## Step 7d: coolant valve as a second control (extension, built 13 Nov 2026)

Why: 39 of 635 reactor-2 test batches stayed unfinished because weak reactors are over-cooled at the fixed
80 % coolant, so the brake refuses to let A accumulate. If the controller can close the coolant, they can warm up.
This is no longer the exact Fricz benchmark (they keep the coolant fixed), so it is reported as an extension.

| Item | Choice |
|---|---|
| AI action | feed-valve change and coolant-valve change, each -5 % to +5 % per 50 s; coolant starts at 80 % |
| Brake, step check | the requested feed AND coolant held for 50 s |
| Brake, backup | A: feed shut + coolant 100 % (emergency opening); B: feed shut + coolant kept where it is |
| Brake, if unsafe | first cut the feed; only if no feed is safe at the requested coolant, open the coolant (bisection) |
| Override penalty | mu = 1 on (feed cut + coolant forced open) / 5 %, same as the final method |
| After feeding | coolant opened fully for the 1 h tail (Step 7 set-ups keep 80 %) |
| New no-AI baselines | constant feed + coolant PI (no brake); max feed + brake A / B; coolant PI + max feed + brake A / B |

Checks (`r2_test_coolant.py`, all pass): the Step 7 brake result is reproduced bit for bit; Gymnasium checks;
brake B with the coolant held at 80 % gives the same feed as the Step 7 brake at every step; a random two-valve
driver behind brake A and brake B overheated 0 of 47 reactors (worst 372.5 K). Trial training (3,000 steps): 0 overheated.
Note: below about 55 % feed the charge cannot be fed within 4 h, so the no-brake coolant PI is tuned at 60-100 % feed.

### Step 7d results (laptop, 13 Nov 2026; 5 seeds x 150k steps; same 127 test reactors as Step 8)

Test set = 27 grid + 100 random reactors (+/-20 %), per seed. 95 % Clopper-Pearson intervals in brackets.

| Method | Overheated | Unfinished | Mean IAE | Median feed time | Cooling failure (overheated / worst) |
|---|---|---|---|---|---|
| Const feed + coolant PI, no brake (Kp 10, Ti 1000 s, feed 60 %) | 109/127 (78-91 %) | 0/127 | 91,179 | 203 min | 10/12, 380.1 K |
| Max feed + brake A (coolant fixed 80 %, no AI) | 0/127 | 4/127 | 135,701 | 166 min | 0/12, 372.5 K |
| Max feed + brake B (coolant fixed 80 %, no AI) | 0/127 | 4/127 | 136,405 | 166 min | 0/12, 372.3 K |
| **Coolant PI + max feed + brake A** (Kp 10, Ti 1000 s) | **0/127 (0-2.9 %)** | **0/127 (0-2.9 %)** | **89,557** | **157 min** | 1/12, 374.2 K |
| Coolant PI + max feed + brake B (Kp 10, Ti 250 s) | 0/127 | 0/127 | 91,260 | 168 min | 1/12, 373.2 K |
| AI, two valves, brake A (`B_cool_A`) | 0/635 | **469/635 (70-77 %)** | 106,803 | 214 min | 0/60 |
| AI, two valves, brake B (`B_cool_B`) | 0/635 | **327/635 (48-55 %)** | 99,236 | 196 min | 0/60 |
| *Step 7 final method, one valve (`B_robust_pen1`)* | *0/635* | *39/635 (4.4-8.3 %)* | *134,618* | *180 min* | *0/60* |

Findings:
1. **Brake + coolant valve is the best result so far, but the winner is the classic coolant PI behind our brake, not the AI.**
   Coolant PI + brake A: 0 overheated, 0 unfinished, IAE 33.5 % lower than `B_robust_pen1` (Wilcoxon on the 127 reactors,
   p = 3.4e-4), batches 23 min shorter. The same PI without the brake overheated 109/127 (worst 395.2 K): the brake is what makes it safe.
2. **The two-valve AI learned to stall.** Unfinished batches went from 6 % (one valve) to 51-74 %. Spread over seeds:
   A 89/19/127/107/127 unfinished of 127, B 57/127/0/127/16 of 127. Traces on the nominal reactor show two modes:
   - *Stall in band* (B seed 1, B seed 3): feed kept at 5-25 %, coolant nearly shut, T held at the setpoint; only 43 % of the
     charge is fed in 4 h. The reward pays +10 per step in the band and nothing for finishing, and the 4 h time limit is a
     truncation (TD3 bootstraps through it), so a never-ending in-band batch is worth more than finishing. With one valve,
     holding T needed feed, so tracking and finishing went together; the coolant valve separates them. **This is reward hacking. The reward causes it; the brake plays no part.**
   - *Freeze* (A seeds 2 and 4): feed stops at 11 %, coolant 0 %, T settles 3.9 K below the setpoint (-3.9 reward per step).
     A local optimum that training never left (unfinished in 98-100 % of practice batches all the way through).
3. Brake A vs B (classic controller): A gives lower IAE on 126/127 reactors (p = 2.7e-21) with fewer overrides (112 vs 136 per batch),
   because its stronger rescue lets more through. For the AI, A had more unfinished batches, but the seed spread is too large to blame the rule.
4. **Cooling failure: the coolant PI + brake overheated slightly in 1 of 12 failure times (at 2 h: 374.2 K with A, 373.2 K with B).**
   The test keeps feeding for 100 s after the failure before the operator shuts the feed; the brake assumes the feed stops at once.
   A controller that runs closer to the limit (more unreacted feed) is hurt by those 100 s. The Step 7 oracle brake showed the same (3/12).
   Fix if needed: make the brake's backup start after the 100 s operator delay. The AI's 0/60 is not a success: it barely fed.

What this means for the paper: report the extension as a negative result for the AI with a clear cause (reward design), and the
classic-PI + brake row as evidence that the safety comes from the brake and the better tracking from the coolant valve.

### Step 7m: finish-the-batch reward (built 13 Nov 2026; `finish_reward=True`, configs `B_coolF_A`, `B_coolF_B`)

| Event | Reward change |
|---|---|
| Feed finished at step n | + 10 x (288 - n): what staying on target for the rest of the 4 h would have earned |
| 4 h reached, feed unfinished | - 10 x 288 x (unfed fraction), and the episode is **terminal** (no bootstrapping through the time limit) |
| Every other step | unchanged (Fricz reward + MDC penalty + override penalty mu = 1) |

Effect: stalling on target can at best tie with finishing before discounting, and loses after it; any unfinished batch is strictly worse.
Everything else (brakes A/B, TD3 settings, 150k steps, seeds) is the same as `B_cool_A/B`, so the comparison isolates the reward.
Only the coolant set-ups use it; all Step 7/8 results are unchanged.

**Step 7m/7n results (laptop, 13 Nov 2026; 5 seeds x 150k steps):**

| Set-up | Overheated | Unfinished | Mean IAE | Median feed time | Cooling failure |
|---|---|---|---|---|---|
| `B_coolF_B` (finish reward, brake B) | 0/635 | **21/635 (2.1-5.0 %)** (was 327) | 101,454 | 177 min | 0/60 |
| `B_coolF_A` (finish reward, brake A) | 0/635 | 159/635 (22-29 %) (was 469) | 103,820 | 162 min | 2/60 (373.3, 373.1 K) |
| Coolant PI + brake A (no AI) | 0/127 | 0/127 | 89,557 | 157 min | 1/12 (374.2 K) |
| `B_robust_pen1` (Step 7, one valve) | 0/635 | 39/635 | 134,618 | 180 min | 0/60 |

Per seed, unfinished of 127: `B_coolF_B` 1/3/0/11/6; `B_coolF_A` 4/0/0/127/28 (seed 3 never finished in practice either).
- The fix works: unfinished 327 -> 21 (Fisher p = 4e-94) for B, 469 -> 159 (p = 3e-70) for A.
- `B_coolF_B` vs the one-valve final method: fewer unfinished (21 vs 39, p = 0.024) and IAE -24.6 % (Wilcoxon on 127 reactors, seed means, p = 1e-4).
- But the classic coolant PI + brake A still tracks better: the agent is +13.3 % IAE (better on only 58/127 reactors, p = 6e-4);
  +11.2 % vs PI + brake B (p = 0.011). The best seed (2) ties it: 89,521 vs 89,557, p = 0.33, 0 unfinished.
- Rule B is the better rule for the AI (fewer unfinished, no cooling-failure overshoot); rule A is slightly better for the classic PI.
- `B_coolF_B` unfinished reactors are almost all weak chemistry (dH x0.8 with k x0.8 or UA x1.2), the same type as Step 7.

### Step 7o: delay-aware brake (built 13 Nov 2026; `cf_aware="nominal"` or `"robust"`, default off)

Why: in the cooling-failure test the controller keeps running for 100 s before the feed is shut, but the brake assumed the
feed stops at once; coolant PI + brake passed MAT by up to 1.2 K. New extra check in `R2Shield.is_safe` (`r2_shield.py`):
"cooling fails at the start of this step, feed stays at the proposed opening for 100 s, then shut, no coolant flow:
does T stay <= MAT until MTSR <= MAT or for 2 h?" `"robust"` runs this on the brake's worst-case corners, `"nominal"` on the
design model only. Checks: `r2_test_coolant.py` test 6 (max feed survives all 12 failures; worst 372.28 K) and tests 1-5
unchanged (the default brake reproduces Step 7 bit for bit).

Quick test (`r2_cf_aware.py --quick`; nominal + 27 grid reactors; same PI tuning as Step 7d; the brake is about 10x slower):

| Method | Brake | Unfinished (grid) | Mean IAE (grid) | Median feed | Cooling failure (nominal) |
|---|---|---|---|---|---|
| Max feed (one valve) | normal (Step 7) | 2/27 | 163,932 | 175 min | 0/12, 372.3 K |
| | delay-aware, nominal | 3/27 | 179,895 | 191 min | 0/12, 372.3 K |
| | delay-aware, robust | 6/27 | 218,504 | 204 min | 0/12, 369.8 K |
| Coolant PI + brake A | normal | 0/27 | 93,299 | 164 min | **1/12, 374.2 K** |
| | delay-aware, nominal | **18/27** | 95,263 | 214 min | 0/12, 372.5 K |
| | delay-aware, robust | **27/27** | 86,870 | - | 0/12, 369.9 K |
| Coolant PI + brake B | normal | 0/27 | 94,724 | 174 min | **1/12, 373.2 K** |
| | delay-aware, nominal | 20/27 | 98,155 | 220 min | 0/12, 372.5 K |
| | delay-aware, robust | 27/27 | 84,363 | - | 0/12, 370.5 K |

Findings: the delay-aware brake closes the gap (0/12 for every method; the nominal variant leaves 0.5 K margin), but the
coolant PI can then no longer finish most batches, even on the nominal reactor. Reason: to survive a cooling failure with
100 s of extra feed, little unreacted A may be present, which needs a hot, fast reaction; the benchmark task holds 363 K,
10 K below MAT, where the reaction is slow and A accumulates. Holding 363 K and being safe against a delayed-reaction
cooling failure conflict on this reactor. (IAE of unfinished batches is low because they sit on the setpoint without finishing.)
Decision: keep the delay-aware brake as an option, report the trade-off in the paper; the main results keep the Step 7
brake, whose guarantee ("stop the feed, keep cooling") is stated as a limitation. The full agent run (7p) is optional:
agents trained to hold 363 K are expected to stall the same way.

**Conclusion for the paper (draft v1.2, Sections 3.7 and 4.6):** the brake makes any controller safe (the coolant PI goes from
109/127 overheated to 0); the AI beat the classic options only when the coolant was fixed. With the coolant valve, a classic PI behind
the brake is as good as the best AI seed and better than the AI on average. Checks (`r2_test_coolant.py` test 5, all pass): a finished batch
differs from the old reward only on its last step, by exactly 10 x remaining steps; a batch that never feeds ends terminal at step 288
with -2880; with the option off the 4 h limit is still a truncation.

## Limitations to state in the paper
- The state, including accumulated anhydride, is assumed measured (inline IR / calorimetry).
- The robust shield checks the worst-case corners only; safety is verified on a 27-plant grid and 1,000 random
  reactors within +/-20 %, not formally proven. Outside its design range (+/-30 %) it overheated 18 of 1,000 (max 52.6 C).
- The brake does not keep MTSR below 50 C (see 'What the brake actually checks').
- Uncertainty set covers k, dH, UA only (not Ea, heat of mixing, feed temperature).
- Constant coolant temperature (no jacket dynamics). dH_r is taken as temperature-independent (Kirchhoff correction < 0.2 %).
- Pseudo-first-order kinetics. Optional `water_order=1` is available as a sensitivity check.
