"""
Step 6: stress test of the emergency brake on many random 'wrong-chemistry' reactors.

The 27-plant grid only checks x0.8 / x1.0 / x1.2. Here every reactor gets random
reaction speed (k), reaction heat (dH) and cooling (UA) factors, drawn uniformly:

  inside the design range   x0.8 - x1.2   (what the careful brake was built for)
  outside the design range  x0.7 - x1.3   (to see what happens beyond it)

Two kinds of driver push the brake as hard as possible:
  full speed   : always asks for the maximum feed (the brake does all the work)
  random       : asks for a random feed every 10 s (erratic, like an untrained AI)

    python validate_shield.py            # about 10-20 min on a laptop
    python validate_shield.py --quick    # 50 reactors per case, about 1-2 min

Output: results/shield_validation.csv and results/shield_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from reactor_env import ReactorDosingEnv, run_episode

OUT = "results"


def make_driver(kind: str, seed: int):
    if kind == "full speed":
        return lambda obs, env: np.array([1.0], dtype=np.float32)
    rng = np.random.default_rng(seed)
    return lambda obs, env: np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="random reactors per case")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 50 if args.quick else args.n
    os.makedirs(OUT, exist_ok=True)

    rows, t0 = [], time.time()
    cases = [(brake, rng_box, driver)
             for brake in ("robust", "nominal")
             for rng_box in (0.2, 0.3)
             for driver in ("full speed", "random")]
    for brake, box, driver in cases:
        env = ReactorDosingEnv(safety="shield", shield_type=brake)   # brake built for +/-20 %
        rng = np.random.default_rng(2026)                             # same reactors for every case
        for i in range(n):
            f = dict(zip(("k_factor", "dH_factor", "UA_factor"), rng.uniform(1 - box, 1 + box, 3)))
            r = run_episode(env, make_driver(driver, i), options={"plant_factors": f})
            rows.append(dict(brake=brake, range_pct=int(round(box * 100)), driver=driver, reactor=i,
                             **{k: round(v, 4) for k, v in f.items()},
                             batch_min=r["batch_time_min"], T_peak_C=round(r["T_peak_C"], 4),
                             overheated=r["violated"], feed_tv=round(r["feed_tv"], 2)))
        sel = rows[-n:]
        print(f"{brake:8s} brake | chemistry +/-{int(box*100)}% | {driver:10s} driver: "
              f"overheated {sum(r['overheated'] for r in sel):3d}/{n}  "
              f"worst peak {max(r['T_peak_C'] for r in sel):5.2f} C  ({time.time()-t0:.0f} s)", flush=True)

    with open(f"{OUT}/shield_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = []
    for brake, box, driver in cases:
        sel = [r for r in rows if (r["brake"], r["range_pct"], r["driver"]) == (brake, int(round(box * 100)), driver)]
        done = [r["batch_min"] for r in sel if r["batch_min"] is not None]
        summary.append(dict(brake=brake, chemistry_range_pct=int(round(box * 100)), driver=driver,
                            reactors=len(sel), overheated=sum(r["overheated"] for r in sel),
                            worst_peak_C=max(r["T_peak_C"] for r in sel),
                            mean_batch_min=float(np.mean(done)) if done else None,
                            unfinished=len(sel) - len(done)))
    with open(f"{OUT}/shield_validation.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done in {time.time()-t0:.0f} s -> {OUT}/shield_validation.csv, .json")


if __name__ == "__main__":
    main()
