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
df = pd.read_csv("runs/eval_all.csv")
df["lot"] = df.run.str.replace("\\", "/", regex=False).str.extract(r"runs/(\w+/[^/]+)/")[0]
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


# Figure 1: learning curve on the training track.
# At a few checkpoints: median lap time of the seeds that finish, with the number of finishing seeds when < 5.
MASTER = 415
CHECKPOINTS = [1000, 2000, 3000, 5000, 10_000, 15_000, 20_000, 25_000, 50_000, 75_000, 100_000]
CURVES = [("rlpd/t4_lighthouse_1tour_kaggle", "rlpd", "RLPD, 1 demo lap"),
          ("rlpd/t4_lighthouse_sac_kaggle", "sac", "SAC only")]


def checkpoint_medians(lot):
    """(step, median lap of finishing seeds, n finishing, n seeds) at each checkpoint with >= 1 finisher."""
    g = df[(df.lot == lot) & (df.track == "lighthouse")]
    out = []
    for s in CHECKPOINTS:
        h = g[g.env_steps == s]
        fin = h[h.finished == 1]
        if len(fin):
            out.append((s, fin.lap_steps.median(), len(fin), len(h)))
    return out


fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.grid(axis="y", color="0.9", lw=.6, zorder=0)
ax.axhline(MASTER, color="0.2", ls=(0, (3, 2)), lw=.8, zorder=4)
ax.text(430, MASTER + 10, "DAgger = master", ha="left", va="bottom", fontsize=6.5, color=C["dagger"])
dag = df[(df.lot == "dagger/t4_lighthouse_100k") & (df.track == "lighthouse") & (df.finished == 1)]
dag_lap = dag.lap_steps.median()
ax.plot([415, 1e5], [dag_lap] * 2, color=C["dagger"], lw=1.6, zorder=3)
ax.plot(415, dag_lap, "o", ms=4.5, color=C["dagger"], mec="white", mew=.8, zorder=5)
for lot, key, lab in CURVES:
    pts = checkpoint_medians(lot)
    x, y = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(x, y, color=C[key], lw=1.4, zorder=4)
    ax.plot(x, y, "o", ms=4.5, color=C[key], mec="white", mew=.8, zorder=5)
    for s, m, k, n in pts:
        if k < n:
            off, ha = ((5, 2), "left") if key == "sac" else ((-5, -9), "right")
            ax.annotate(f"{k}/{n}", (s, m), xytext=off, textcoords="offset points", ha=ha,
                        fontsize=6.5, color="0.35", family="monospace")
ax.set_xscale("log")
ax.set_xlim(380, 1.15e5)
ax.set_xticks([415, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
ax.set_xticklabels(["415", "1k", "2k", "5k", "10k", "20k", "50k", "100k"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(360, 1010)
ax.set_yticks(range(400, 1001, 100))
ax.tick_params(labelsize=7, length=0)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("0.6")
ax.set_xlabel("environment steps (log scale, demos included)")
ax.text(-.02, 1.03, "lap time (steps)", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.3")
handles = [plt.Line2D([], [], color=C[k], marker="o", ms=4.5, mec="white", lw=1.4) for k in ("rlpd", "sac", "dagger")]
handles.append(plt.Line2D([], [], color="0.45", ls=(0, (3, 2)), lw=.9))
ax.legend(handles, ["RLPD, 1 demo lap", "SAC only", "DAgger", "master team4"], frameon=False, fontsize=7,
          loc="lower left", bbox_to_anchor=(-.02, 1.1), ncol=2, handlelength=1.8, columnspacing=1.2,
          handletextpad=.4, borderaxespad=0)
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
