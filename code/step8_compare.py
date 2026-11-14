"""
Step 8: one combined comparison of both reactors (final numbers and figures for the paper).

Reads only finished results (nothing is simulated or trained):
    results/agent_eval.csv, results/agents/*_meta.json            reactor 1 (Steps 5-6, 5 seeds)
    results/r2/agent_eval.csv, results/r2/cooling_failure.csv,
    results/r2/agents/*_meta.json                                  reactor 2 (Step 7, 5 seeds)

    python step8_compare.py          (~10 s)

Writes results/step8/:
    main_table.csv / main_table.md   one row per method and reactor (safety, performance, practice)
    penalty_sensitivity.csv          reactor 2, override penalty mu = 0 / 1 / 5, per seed
    stats.json                       confidence intervals and significance tests
    fig_safety.png                   overheated share of test and practice batches, with 95 % intervals
    fig_tradeoff.png                 performance vs safety, one point per seed
    fig_penalty.png                  override penalty: brake cuts, tracking, unfinished, per seed

Definitions
    Test batches      = 27 grid + 100 random reactors per seed (the nominal reactor is reported separately).
    Overheated        = T above the limit at any time, including 1 h after feeding (reactor 2) / until the end (reactor 1).
    95 % interval     = Clopper-Pearson exact binomial interval for the overheated share. With 0 events in n
                        batches the upper bound is about 3/n ("rule of three").
    Performance gap   = reactor 1: batch time vs the oracle (%); reactor 2: tracking error (IAE) vs the oracle brake (%),
                        averaged over the test reactors that every method finished. Lower is better; negative = better
                        than the oracle.
    Fisher test       = is the overheated share of method X different from our method's? Batches pooled over seeds
                        (treated as independent: an approximation, stated in the paper).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.join("results", "step8")
OURS = {1: "Driver B, careful brake", 2: "Driver B, careful brake + override penalty"}

# (reactor, name in the result files) -> name in the paper, kind
METHODS = [
    (1, "PID (robust tuned)", "PID, robust tuned", "classic"),
    (1, "Max feed + robust brake (no AI)", "Careful brake, no AI", "brake"),
    (1, "Oracle", "Oracle brake (knows true reactor)", "oracle"),
    (1, "AI A_nominal", "Driver A, penalty, nominal practice", "A"),
    (1, "AI A_random", "Driver A, penalty, random practice", "A"),
    (1, "AI B_nominal", "Driver B, normal brake", "Bn"),
    (1, "AI B_robust", "Driver B, careful brake", "B"),
    (2, "PI (best tracking)", "PI, best tracking (no safe PI exists)", "classic"),
    (2, "PI + robust brake", "PI + careful brake", "brake"),
    (2, "Max feed + robust brake (no AI)", "Careful brake, no AI", "brake"),
    (2, "Oracle brake", "Oracle brake (knows true reactor)", "oracle"),
    (2, "A_nominal", "Driver A, penalty, nominal practice (= Fricz 2026)", "A"),
    (2, "A_random", "Driver A, penalty, random practice", "A"),
    (2, "B_nominal", "Driver B, normal brake", "Bn"),
    (2, "B_robust", "Driver B, careful brake", "B"),
    (2, "B_robust_pen1", "Driver B, careful brake + override penalty", "B"),
]
CONFIG_OF = {"AI A_nominal": "A_nominal", "AI A_random": "A_random", "AI B_nominal": "B_nominal",
             "AI B_robust": "B_robust"}
COLORS = {"classic": "#7f7f7f", "brake": "#2ca02c", "oracle": "#17becf", "A": "#d62728", "Bn": "#ff7f0e", "B": "#1f77b4"}


def ci(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95 % interval for k events in n trials (as fractions)."""
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def load():
    r1 = pd.read_csv(os.path.join("results", "agent_eval.csv"))
    r1["finished"] = r1.batch_min.notna()
    r2 = pd.read_csv(os.path.join("results", "r2", "agent_eval.csv"))
    r2["finished"] = r2.feed_min.notna() & (r2.feed_min < 240)
    # reactor 2 performance: IAE gap vs oracle brake on test reactors every method finished
    test = r2[r2.group != "nominal"].copy()
    ok = test.groupby(["group", "reactor"]).finished.all()
    common = set(ok[ok].index)
    test["common"] = [(g, r) in common for g, r in zip(test.group, test.reactor)]
    orc = test[(test.method == "Oracle brake") & test.common].set_index(["group", "reactor"]).iae
    test["gap"] = [100 * (i / orc[(g, r)] - 1) if c else np.nan
                   for i, g, r, c in zip(test.iae, test.group, test.reactor, test.common)]
    r1t = r1[r1.group != "nominal"].copy()
    r1t["gap"] = r1t.slower_than_oracle_pct
    cf = pd.read_csv(os.path.join("results", "r2", "cooling_failure.csv"))
    practice = {}
    for reactor, folder in ((1, os.path.join("results", "agents")), (2, os.path.join("results", "r2", "agents"))):
        for f in glob.glob(os.path.join(folder, "*_meta.json")):
            m = json.load(open(f))
            if "_quick" in f:
                continue
            p = practice.setdefault((reactor, m["config"]), [0, 0])
            p[0] += m["practice_overheated"]
            p[1] += m["practice_batches"]
    return {1: r1t, 2: test}, {1: r1, 2: r2}, cf, practice, len(common)


def main():
    os.makedirs(OUT, exist_ok=True)
    tests, full, cf, practice, n_common = load()
    rows, per_seed = [], []
    for reactor, src, name, kind in METHODS:
        t = tests[reactor][tests[reactor].method == src]
        if t.empty:
            continue
        nom = full[reactor][(full[reactor].method == src) & (full[reactor].group == "nominal")]
        k, n = int(t.overheated.sum()), len(t)
        lo, hi = ci(k, n)
        seeds = sorted(t.seed.dropna().unique())
        gaps = [t[t.seed == s].gap.mean() for s in seeds] if seeds else [t.gap.mean()]
        cfg = CONFIG_OF.get(src, src)
        pk = practice.get((reactor, cfg))
        c = cf[cf.method == src] if reactor == 2 else None
        peak = t.T_peak_C.max() if reactor == 1 else t.T_peak_K.max() - 273.15
        rows.append(dict(
            reactor=reactor, method=name, kind=kind, seeds=len(seeds) or 1,
            test_batches=n, overheated=k, overheated_pct=round(100 * k / n, 2),
            ci95_low_pct=round(100 * lo, 2), ci95_high_pct=round(100 * hi, 2),
            worst_peak_C=round(peak, 1), unfinished=int((~t.finished).sum()),
            perf_gap_pct_mean=round(float(np.nanmean(gaps)), 1),
            perf_gap_pct_min=round(float(np.nanmin(gaps)), 1), perf_gap_pct_max=round(float(np.nanmax(gaps)), 1),
            nominal_overheated=int(nom.overheated.sum()), nominal_runs=len(nom),
            practice_overheated=pk[0] if pk else None, practice_batches=pk[1] if pk else None,
            cooling_failure=f"{int(c.overheated.sum())}/{len(c)}" if c is not None and len(c) else "",
            brake_cuts_per_batch=round(t.interventions.mean(), 1) if t.interventions.notna().any() else None))
        for s, g in zip(seeds, gaps):
            ts = t[t.seed == s]
            per_seed.append(dict(reactor=reactor, method=name, kind=kind, seed=int(s), gap=g,
                                 overheated_pct=100 * ts.overheated.mean(), unfinished=int((~ts.finished).sum()),
                                 cuts=ts.interventions.mean()))
        if not seeds:
            per_seed.append(dict(reactor=reactor, method=name, kind=kind, seed=-1, gap=gaps[0],
                                 overheated_pct=100 * k / n, unfinished=int((~t.finished).sum()), cuts=np.nan))
    table = pd.DataFrame(rows)
    seeds_df = pd.DataFrame(per_seed)
    table.to_csv(os.path.join(OUT, "main_table.csv"), index=False)

    # significance: our method vs every other method, overheated share (Fisher exact, two-sided)
    st = {"note": "Fisher exact tests pool batches over seeds (treated as independent). 95 % intervals: Clopper-Pearson.",
          "reactor2_common_test_reactors_for_IAE": n_common, "tests": []}
    for reactor in (1, 2):
        tr = table[table.reactor == reactor]
        ours = tr[tr.method == OURS[reactor]].iloc[0]
        for _, r in tr.iterrows():
            if r.method == ours.method:
                continue
            p = stats.fisher_exact([[ours.overheated, ours.test_batches - ours.overheated],
                                    [r.overheated, r.test_batches - r.overheated]])[1]
            st["tests"].append(dict(reactor=reactor, ours=ours.method, other=r.method,
                                    ours_overheated=f"{ours.overheated}/{ours.test_batches}",
                                    other_overheated=f"{r.overheated}/{r.test_batches}", p_value=float(p)))
        # practice
        for _, r in tr[tr.practice_batches.notna()].iterrows():
            if r.method == ours.method:
                continue
            p = stats.fisher_exact([[ours.practice_overheated, ours.practice_batches - ours.practice_overheated],
                                    [r.practice_overheated, r.practice_batches - r.practice_overheated]])[1]
            st["tests"].append(dict(reactor=reactor, ours=ours.method, other=r.method, what="practice",
                                    ours_overheated=f"{int(ours.practice_overheated)}/{int(ours.practice_batches)}",
                                    other_overheated=f"{int(r.practice_overheated)}/{int(r.practice_batches)}",
                                    p_value=float(p)))
    # reactor 2 performance: our method vs careful brake alone, paired per test reactor (Wilcoxon)
    t2 = tests[2]
    a = t2[(t2.method == "B_robust_pen1") & t2.common].groupby(["group", "reactor"]).gap.mean()
    b = t2[(t2.method == "Max feed + robust brake (no AI)") & t2.common].set_index(["group", "reactor"]).gap
    a, b = a.align(b, join="inner")
    w = stats.wilcoxon(a, b)
    st["reactor2_ours_vs_brake_alone_IAE"] = dict(
        reactors=len(a), ours_mean_gap_pct=round(float(a.mean()), 1), brake_alone_mean_gap_pct=round(float(b.mean()), 1),
        ours_better_on=int((a < b).sum()), wilcoxon_p=float(w.pvalue),
        note="paired over test reactors; ours = mean over 5 seeds per reactor")
    with open(os.path.join(OUT, "stats.json"), "w") as fh:
        json.dump(st, fh, indent=2)

    # penalty sensitivity (reactor 2)
    sens = []
    for mu, src in ((0, "B_robust"), (1, "B_robust_pen1"), (5, "B_robust_pen")):
        t = t2[t2.method == src]
        for s in sorted(t.seed.unique()):
            ts = t[t.seed == s]
            sens.append(dict(mu=mu, seed=int(s), brake_cuts_per_batch=round(ts.interventions.mean(), 1),
                             iae_gap_vs_oracle_pct=round(ts.gap.mean(), 1), unfinished=int((~ts.finished).sum()),
                             overheated=int(ts.overheated.sum())))
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(OUT, "penalty_sensitivity.csv"), index=False)

    write_markdown(table, st, sens)
    plot_safety(table)
    plot_tradeoff(seeds_df)
    plot_penalty(sens)
    print(open(os.path.join(OUT, "main_table.md"), encoding="utf-8").read())
    print(f"Files written to {OUT}/")


def write_markdown(table, st, sens):
    L = ["# Step 8: combined comparison (5 seeds; test = 27 grid + 100 random reactors per seed)", ""]
    for reactor, title, perf in ((1, "Reactor 1: acetic anhydride hydrolysis (goal: finish fast, stay below 50 C)",
                                  "batch time vs oracle"),
                                 (2, "Reactor 2: Fricz/Kummer benchmark (goal: hold 363 K, stay below 373 K)",
                                  "tracking error vs oracle brake")):
        L += [f"## {title}", "",
              f"| Method | Overheated (test) | 95 % interval | Worst peak | Unfinished | {perf}: mean [seed range] | "
              f"Practice overheated | Cooling failure |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in table[table.reactor == reactor].iterrows():
            rng = f" [{r.perf_gap_pct_min:+.1f}, {r.perf_gap_pct_max:+.1f}]" if r.seeds > 1 else ""
            prac = f"{int(r.practice_overheated)} / {int(r.practice_batches)}" if pd.notna(r.practice_batches) else "-"
            name = f"**{r.method}**" if r.method == OURS[reactor] else r.method
            L.append(f"| {name} | {r.overheated} / {r.test_batches} | {r.ci95_low_pct:.1f}-{r.ci95_high_pct:.1f} % | "
                     f"{r.worst_peak_C:.1f} C | {r.unfinished} | {r.perf_gap_pct_mean:+.1f} %{rng} | {prac} | "
                     f"{r.cooling_failure or '-'} |")
        L.append("")
    s = st["reactor2_ours_vs_brake_alone_IAE"]
    L += ["## Significance", "",
          f"- Reactor 2 tracking, our method vs careful brake alone (paired over {s['reactors']} test reactors): "
          f"gap {s['ours_mean_gap_pct']:+.1f} % vs {s['brake_alone_mean_gap_pct']:+.1f} %, ours better on "
          f"{s['ours_better_on']} of {s['reactors']}, Wilcoxon p = {s['wilcoxon_p']:.1e}.",
          "- Overheated share, our method vs others (Fisher exact, p-value):"]
    for t in st["tests"]:
        L.append(f"  - reactor {t['reactor']}, {t.get('what', 'test')}: vs {t['other']}: "
                 f"{t['ours_overheated']} vs {t['other_overheated']}, p = {t['p_value']:.1e}")
    L += ["", "## Override penalty sensitivity (reactor 2, careful brake, 5 seeds each)", "",
          "| mu | brake cuts per batch [seed range] | tracking gap vs oracle [seed range] | unfinished (of 635) | overheated |",
          "|---|---|---|---|---|"]
    for mu, g in sens.groupby("mu"):
        L.append(f"| {mu} | {g.brake_cuts_per_batch.mean():.0f} [{g.brake_cuts_per_batch.min():.0f}-"
                 f"{g.brake_cuts_per_batch.max():.0f}] | {g.iae_gap_vs_oracle_pct.mean():+.1f} % "
                 f"[{g.iae_gap_vs_oracle_pct.min():+.1f}, {g.iae_gap_vs_oracle_pct.max():+.1f}] | "
                 f"{g.unfinished.sum()} | {g.overheated.sum()} |")
    with open(os.path.join(OUT, "main_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def plot_safety(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, reactor in zip(axes, (1, 2)):
        t = table[table.reactor == reactor].reset_index(drop=True)
        y = np.arange(len(t))[::-1]
        err = [t.overheated_pct - t.ci95_low_pct, t.ci95_high_pct - t.overheated_pct]
        ax.barh(y + 0.2, t.overheated_pct, height=0.38, xerr=err, color=[COLORS[k] for k in t.kind],
                ecolor="k", capsize=2, label="test reactors (95 % interval)")
        prac = 100 * t.practice_overheated / t.practice_batches
        ax.barh(y - 0.2, prac.fillna(0), height=0.38, color=[COLORS[k] for k in t.kind], alpha=0.4, hatch="//",
                label="practice batches (while learning)")
        for yi, v, pv in zip(y, t.overheated_pct, prac):
            ax.text(v + 0.5, yi + 0.2, f"{v:.1f} %", va="center", fontsize=7)
            if pd.notna(pv):
                ax.text(pv + 0.5, yi - 0.2, f"{pv:.1f} %", va="center", fontsize=7, color="0.3")
        ax.set_yticks(y)
        ax.set_yticklabels([("* " if m == OURS[reactor] else "") + m for m in t.method], fontsize=8)
        ax.set_xlabel("batches that overheated (%)")
        ax.set_title(f"Reactor {reactor}" + (" (limit 50 C)" if reactor == 1 else " (limit 373 K)"))
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Safety: share of batches above the temperature limit (5 seeds; * = our method)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_safety.png"), dpi=150)
    plt.close(fig)


def plot_tradeoff(seeds_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, reactor in zip(axes, (1, 2)):
        d = seeds_df[seeds_df.reactor == reactor]
        seen = {}
        for name, g in d.groupby("method", sort=False):
            kind = g.kind.iloc[0]
            seen[kind] = seen.get(kind, -1) + 1          # second method of the same kind: other marker
            marker = "*" if name == OURS[reactor] else ("o", "^", "s")[seen[kind] % 3]
            ax.scatter(g.gap, g.overheated_pct, s=90 if marker == "*" else 45,
                       color=COLORS[kind], marker=marker,
                       alpha=0.8, label=name, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("batch time vs oracle (%)" if reactor == 1 else "tracking error vs oracle brake (%)")
        ax.set_ylabel("test batches overheated (%)")
        ax.set_title(f"Reactor {reactor}: lower-left is better (one point per seed)")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tradeoff.png"), dpi=150)
    plt.close(fig)


def plot_penalty(sens):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, lab in zip(axes, ("brake_cuts_per_batch", "iae_gap_vs_oracle_pct", "unfinished"),
                            ("brake cuts per batch", "tracking error vs oracle brake (%)", "unfinished test batches (of 127)")):
        for mu in (0, 1, 5):
            g = sens[sens.mu == mu]
            ax.scatter(np.full(len(g), mu) + np.linspace(-0.15, 0.15, len(g)), g[col], s=40, zorder=3)
            ax.plot([mu - 0.25, mu + 0.25], [g[col].mean()] * 2, "k-", lw=2)
        ax.set_xticks([0, 1, 5])
        ax.set_xlabel("override penalty mu")
        ax.set_ylabel(lab)
    fig.suptitle("Reactor 2: override penalty. mu=0 lazy seeds lean on the brake, mu=5 timid seeds under-feed, "
                 "mu=1 best (dots = seeds, bar = mean)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_penalty.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
