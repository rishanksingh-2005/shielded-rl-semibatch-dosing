# Shielded reinforcement learning for exothermic semi-batch dosing under model mismatch

Code, trained agents and result tables for the paper

> R. Singh, *Shielded reinforcement learning for exothermic semi-batch dosing under model mismatch: when does the agent help?*
> Department of Chemical Engineering, Delhi Technological University (DTU), Delhi, India. Manuscript in preparation.

Simulation study only. Nothing here is a design for a real plant.

## What this is

A TD3 reinforcement-learning agent doses reagent into an exothermic semi-batch reactor. Every move it proposes first passes a
safety shield. The shield simulates "apply this feed for one step, then stop the feed with cooling on" on worst-case corners of a
±20 % uncertainty set (rate constant, reaction enthalpy, heat-transfer coefficient), and accepts the move only if the reactor stays
below its limit until the Maximum Temperature of the Synthesis Reaction (MTSR) is below that limit too.

Two reactors are studied:

1. **Reactor 1:** acetic anhydride hydrolysis in a 1 L lab reactor (kinetics and thermochemistry from Ładosz et al., 2020).
2. **Reactor 2:** the A + B → C semi-batch benchmark of Kummer et al. (2020) as published by Fricz et al. (2026), with their reward and TD3 settings.

Main results: the shielded agent overheated in 0 of 1,905 mismatched test batches and 0 of 8,543 practice batches; penalty-only
agents overheated in 1.7–19.8 % of test batches. Whether the agent beats classic control depends on the reactor (details in the paper).

## Requirements

Python 3.12 and the packages in `code/requirements.txt` (CPU only; no graphics card needed):

```
pip install -r code/requirements.txt
```

Everything in the paper was run on a 4-core laptop (Intel Core i5-1135G7, 7.7 GB RAM).

## Reproducing the results

Run all commands inside `code/`. `code/README.md` is the full run log: every step, its command, run time, expected output and results.

| What | Command | Time |
|---|---|---|
| Physics checks, reactor 1 / reactor 2 | `python test_reactor_model.py` / `python r2_test_model.py` | seconds |
| Classic baselines | `python run_baselines.py` / `python r2_baselines.py` | 5-10 min |
| Train agents (reactor 1 / reactor 2), 5 seeds | `python train_agents.py --all --seeds 0-4` / `python r2_train.py --all --seeds 0-4` | hours |
| Test agents | `python evaluate_agents.py` / `python r2_evaluate.py` | 0.5-3 h |
| Shield stress test | `python validate_shield.py` | 20-40 min |
| Coolant-valve extension | `python r2_coolant.py` | about 1 h |
| Paper tables and figures (from stored results) | `python step8_compare.py` | 10 s |

The trained agents used in the paper are included in `code/results/agents/` and `code/results/r2/agents/`, so the test and
comparison steps can be rerun without retraining. Intermediate training checkpoints are not included (about 1.2 GB).

## Repository layout

| Path | Contents |
|---|---|
| `code/reactor_model.py`, `code/r2_model.py` | Reactor models (mass and energy balances, kinetics, jacket, MTSR, MDC) |
| `code/shield.py`, `code/r2_shield.py` | The safety shield (backup-policy look-ahead on corner models) |
| `code/reactor_env.py`, `code/r2_env.py` | Gymnasium training environments |
| `code/train_agents.py`, `code/r2_train.py` | TD3 training (Stable-Baselines3) |
| `code/evaluate_agents.py`, `code/r2_evaluate.py`, `code/r2_coolant.py`, `code/r2_cf_aware.py` | Testing and extensions |
| `code/step8_compare.py` | Final tables, statistics and figures |
| `code/results/` | Result tables (CSV, JSON), figures and final trained agents |

## License

MIT (see `LICENSE`).

## How to cite

See `CITATION.cff`, or use the Zenodo DOI shown on this page once a release is archived.
