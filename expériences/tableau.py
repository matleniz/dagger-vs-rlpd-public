"""Tableau markdown de lots de runs, à partir de ``runs/eval_all.csv`` (après ``python -m lab.registry``).

    python expériences/tableau.py rlpd/t4_lighthouse_1tour_kaggle rlpd/t4_lighthouse_1tour_kaggle@20000 dagger/t4_lighthouse

``méthode/nom@N`` : seuls les points d'évaluation à ``env_steps <= N`` sont gardés. Par lot :
  « fin »      dernier point d'évaluation de chaque graine : tours finis / graines, tour médian [min-max] en pas ;
  « ≥ 5 000 »  évaluations finies / évaluations, à partir de 5 000 pas, toutes graines confondues ;
  « par graine » plus petite et plus grande part d'évaluations finies (≥ 5 000 pas) parmi les graines ;
  « entraînement » les mêmes comptes, toutes pistes d'entraînement confondues ; « centre » = distance moyenne
  au centre de la piste (``mean_center_dist``) au dernier point, médiane des tours finis.
"""
import sys

import pandas as pd

TEST = sys.argv[sys.argv.index("--test") + 1].split(",") if "--test" in sys.argv else ["hacienda", "scotland"]
lots = [x for x in sys.argv[1:] if "/" in x]
df = pd.read_csv("runs/eval_all.csv")


def cell(g):
    last = g[g.env_steps == g.groupby("seed").env_steps.transform("max")]
    late = g[g.env_steps >= 5000]
    laps = last.lap_steps.dropna()
    med = f", {laps.median():.0f} [{laps.min():.0f}-{laps.max():.0f}]" if len(laps) else ""
    per = late.groupby("seed").finished.mean()
    return f"{int(last.finished.sum())}/{len(last)}{med}", f"{int(late.finished.sum())}/{len(late)}", f"{per.min():.2f}-{per.max():.2f}"


print("| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | "
      + " | ".join(f"{t}, fin | {t} ≥ 5 000 | {t} par graine" for t in TEST) + " |")
print("|" + "---|" * (6 + 3 * len(TEST)))
for lot in lots:
    path, _, cap = lot.partition("@")
    method, name = path.split("/")
    g = df[df.run.str.startswith(f"runs/{path}/")]  # le dossier, pas la colonne name (runs déplacés)
    if cap:
        g = g[g.env_steps <= int(cap)]
    if g.empty:
        print(f"| `{path}` | absent |")
        continue
    train = g[~g.track.isin(TEST)]
    tl = train[train.env_steps == train.groupby("seed").env_steps.transform("max")]
    a, b, _ = cell(train)
    cells = [x for t in TEST for x in cell(g[g.track == t])]
    print(f"| `{path}` | {g.seed.nunique()} | {int(g.env_steps.max())} | {a} | {b} | {tl[tl.finished == 1].mean_center_dist.median():.2f} | " + " | ".join(cells) + " |")
