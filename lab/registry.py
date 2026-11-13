"""Registre des runs : relit ``runs/`` et dit où en est chaque run.

    python -m lab.registry            # tableau + runs/index.csv, runs/eval_all.csv, runs/train_all.csv

État : ``fini`` (dernier point d'évaluation = budget atteint), ``en cours`` (journal écrit depuis
moins de ``STALE`` secondes), sinon ``arrêté`` (planté, tué ou figé : lire son .log).
Les trois fichiers produits sont reconstruits à chaque appel à partir des dossiers de runs, qui
restent la seule source de vérité.
"""

import csv
import json
import time
from pathlib import Path

STALE = 900


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def scan(root="runs"):
    """Retourne (index, eval, train) : une ligne par run, et toutes les lignes des journaux."""
    index, evals, trains = [], [], []
    for cfg_path in sorted(Path(root).glob("**/config.json")):
        d, cfg = cfg_path.parent, json.loads(cfg_path.read_text())
        ev = read(d / "eval.csv") if (d / "eval.csv").exists() else []
        tr = read(d / "train.csv") if (d / "train.csv").exists() else []
        files = [f for f in (d / "eval.csv", d / "train.csv") if f.exists()]
        age = time.time() - max((f.stat().st_mtime for f in files), default=cfg_path.stat().st_mtime)
        steps = max([int(r["env_steps"]) for r in ev + tr], default=0)
        done = bool(ev) and int(ev[-1]["env_steps"]) >= cfg["steps"]
        last = {r["track"]: r for r in ev if r["env_steps"] == ev[-1]["env_steps"]} if ev else {}
        index.append({
            "run": str(d), "method": cfg["method"], "name": cfg["name"], "seed": cfg["seed"], "commit": cfg["commit"],
            "host": cfg.get("host", ""), "team": cfg["team"], "tracks": " ".join(cfg["tracks"]),
            "état": "fini" if done else "en cours" if age < STALE else "arrêté",
            "env_steps": steps, "budget": cfg["steps"], "wall_s": (ev + tr)[-1]["wall_s"] if ev or tr else "",
            "dernière éval": " ".join(f"{t}:{r['lap_steps'] if r['finished'] == '1' else 'échec'}" for t, r in last.items()),
        })
        evals += [{"run": str(d), **r} for r in ev]
        trains += [{"run": str(d), **r} for r in tr]
    return index, evals, trains


if __name__ == "__main__":
    index, evals, trains = scan()
    for name, rows in (("index", index), ("eval_all", evals), ("train_all", trains)):
        write(f"runs/{name}.csv", rows)
    for r in index:
        print(f"{r['état']:9s} {r['env_steps']:>7}/{r['budget']:<7} {str(r['wall_s']):>8}s  {r['run']}  {r['dernière éval']}")
    print(f"{len(index)} runs : " + ", ".join(f"{sum(r['état'] == e for r in index)} {e}" for e in ("fini", "en cours", "arrêté")))
