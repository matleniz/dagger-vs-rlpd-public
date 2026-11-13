"""Tableau de exp3 : par configuration et par piste, sur les 5 graines, au dernier point d'évaluation
et sur tous les points à partir de 5 000 pas (tours finis / total, tour médian [min-max]).

    python expériences/exp3_dagger_grille/resume.py [méthode]     # après python -m lab.registry
"""
import sys
import pandas as pd

method = sys.argv[1] if len(sys.argv) > 1 else "dagger"
df = pd.read_csv("runs/eval_all.csv")
df = df[df.method == method]
for name, g in df.groupby("name", sort=False):
    last = g[g.env_steps == g.groupby("seed").env_steps.transform("max")]
    late = g[g.env_steps >= 5000]
    print(f"\n{name}  (graines {g.seed.nunique()}, fin {int(g.env_steps.max())} pas, {g.wall_s.max():.0f} s)")
    for track in g.track.unique():
        a, b = last[last.track == track], late[late.track == track]
        laps = a.lap_steps.dropna()
        med = f"{laps.median():.0f} [{laps.min():.0f}-{laps.max():.0f}]" if len(laps) else "-"
        print(f"  {track:22s} fin : {int(a.finished.sum())}/{len(a)} finis, tour {med:16s} | >=5000 pas : {int(b.finished.sum())}/{len(b)} finis")
