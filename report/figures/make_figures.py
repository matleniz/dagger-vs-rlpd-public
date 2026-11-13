"""Figures of the report, from runs/eval_all.csv (run `python -m lab.registry` first).

    python report/figures/make_figures.py      # from dagger-vs-rlpd/, needs pandas + matplotlib
"""
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif"})
C = {"dagger": "#2a78d6", "rlpd": "#eb6834", "sac": "#1baf7a"}
FAIL = 1500  # episode limit: an unfinished lap is counted 1500 steps
df = pd.read_csv("runs/eval_all.csv")
df["lot"] = df.run.str.extract(r"runs/(\w+/[^/]+)/")[0]
TEST = ["hacienda", "scotland"]


def perm_p(x, y):
    """Exact two-sided permutation test on the difference of means."""
    z = np.r_[x, y]
    obs, n, hits, tot = abs(np.mean(x) - np.mean(y)), len(x), 0, 0
    for idx in combinations(range(len(z)), n):
        m = np.zeros(len(z), bool)
        m[list(idx)] = True
        hits += abs(z[m].mean() - z[~m].mean()) >= obs - 1e-12
        tot += 1
    return hits / tot


def lap_curve(lot, grid):
    """Per seed, lap steps (FAIL if unfinished) of the last evaluation <= each grid point."""
    g = df[(df.lot == lot) & (df.track == "lighthouse")].copy()
    g["t"] = np.where(g.finished == 1, g.lap_steps, FAIL)
    out = []
    for _, h in g.groupby("seed"):
        h = h.sort_values("env_steps")
        idx = np.searchsorted(h.env_steps.values, grid, side="right") - 1
        out.append(np.where(idx >= 0, h.t.values[np.clip(idx, 0, None)], np.nan))
    return np.array(out)


# Figure 1: learning curve on the training track
grid = np.unique(np.r_[np.arange(415, 10_001, 50), np.arange(10_000, 100_001, 1000)])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for lot, key, lab in [("dagger/t4_lighthouse_100k", "dagger", "DAgger"),
                      ("rlpd/t4_lighthouse_1tour_kaggle", "rlpd", "RLPD (1 demo lap)"),
                      ("rlpd/t4_lighthouse_sac_kaggle", "sac", "SAC only (no demo)")]:
    a = lap_curve(lot, grid)
    ok = ~np.isnan(a).any(0)
    ax.fill_between(grid[ok], np.nanmin(a[:, ok], 0), np.nanmax(a[:, ok], 0), color=C[key], alpha=.18, lw=0)
    ax.plot(grid[ok], np.nanmedian(a[:, ok], 0), color=C[key], lw=1.4, label=lab)
ax.axhline(415, color="0.2", ls=":", lw=.9, zorder=5)
ax.text(1.1e5, 415, "master\n415", va="center", fontsize=7, color="0.3")
ax.set_xscale("log")
ax.set_xlim(400, 1e5)
ax.set_ylim(370, 1000)
ax.set_yticks([400, 500, 600, 700, 800, 900, 1000])
ax.set_yticklabels(["400", "500", "600", "700", "800", "900", "$\\geq$1000"])
ax.set_xlabel("environment steps (log scale, demos included)")
ax.set_ylabel("lap time on lighthouse (steps)")
ax.set_title("Training track: learning curves")
ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1, .98))
fig.tight_layout()
fig.savefig("report/figures/fig1_learning.pdf", bbox_inches="tight")

# Figure 2: generalisation, per-seed finished fraction on the test tracks
PAIRS = [("team4, lighthouse", "dagger/t4_lighthouse", "rlpd/t4_lighthouse_1tour_kaggle", 20_500),
         ("team4, lighthouse, 100k", "dagger/t4_lighthouse_100k", "rlpd/t4_lighthouse_1tour_kaggle", None),
         ("team4, stats 16 tracks", "dagger/t4_lighthouse_stats-16pistes", "rlpd/t4_lighthouse_1tour_stats-16pistes_kaggle", 20_500),
         ("team4, 11 demo episodes, 100k", "dagger/t4_lighthouse_100k_demos-rlpd", "rlpd/t4_lighthouse_stats-demos_kaggle", None),
         ("team1, lighthouse", "dagger/t1_lighthouse", "rlpd/t1_lighthouse_wsl", 20_500),
         ("team3, lighthouse$^\\dagger$", "dagger/t3_lighthouse", "rlpd/t3_lighthouse_kaggle", 20_500),
         ("team4, snowtuxpeak", "dagger/t4_snowtuxpeak", "rlpd/t4_snowtuxpeak_kaggle", 20_500),
         ("team4, 4 tracks", "dagger/t4_4pistes", "rlpd/t4_4pistes_kaggle", 20_700),
         ("team4, 16 tracks, 100k", "dagger/t4_16pistes_100k", "rlpd/t4_16pistes_kaggle", None)]


def frac(lot, cap):
    g = df[(df.lot == lot) & df.track.isin(TEST) & (df.env_steps >= 5000)]
    if cap:
        g = g[g.env_steps <= cap]
    return g.groupby("seed").finished.mean().values


fig, ax = plt.subplots(figsize=(7.0, 2.6))
rows = []
for i, (name, d, r, cap) in enumerate(PAIRS):
    x, y = frac(d, cap), frac(r, cap)
    p = perm_p(x, y)
    rows.append((name, x.mean(), y.mean(), p))
    yy = len(PAIRS) - 1 - i
    for v, key, off in [(x, "dagger", .14), (y, "rlpd", -.14)]:
        ax.scatter(v, np.full(len(v), yy + off), s=12, color=C[key], alpha=.8, lw=0, zorder=3)
        ax.plot([v.mean()] * 2, [yy + off - .16, yy + off + .16], color=C[key], lw=1.6, zorder=4)
ax.set_yticks(range(len(PAIRS)))
ax.set_yticklabels([f"{n}  ($p$={p:.3f})" for n, _, _, p in rows][::-1], fontsize=7)
ax.set_xlim(-.03, 1.03)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_ylim(-.6, len(PAIRS) - .2)
ax.set_xlabel("fraction of evaluations finished on hacienda + scotland (dot = one seed, bar = mean of 5 seeds)")
ax.set_title("Generalisation to unseen tracks", loc="left")
ax.scatter([], [], color=C["dagger"], s=12, label="DAgger")
ax.scatter([], [], color=C["rlpd"], s=12, label="RLPD")
ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, borderaxespad=0)
ax.grid(axis="x", color="0.9", lw=.6)
fig.tight_layout()
fig.savefig("report/figures/fig2_generalisation.pdf", bbox_inches="tight")
for r in rows:
    print(f"{r[0]:32s} DAgger {r[1]:.2f}  RLPD {r[2]:.2f}  p={r[3]:.3f}")
