"""Métriques d'un épisode et journal CSV, communs à DAgger et RLPD.

Principe : on journalise tout ce qui est bon marché, la métrique principale se choisit après coup.
La récompense cumulée est celle de l'environnement, brute (jamais celle, modifiée, de
l'entraînement), pour que toutes les méthodes se mesurent exactement de la même façon.
"""

import csv
from pathlib import Path

import numpy as np

from .envs import get_raw_obs, play

#: Avancement par pas en dessous duquel on dit que le kart est bloqué
STALL_EPS = 1e-3


def episode_metrics(env, act, seed=None, track=None, on_reset=None) -> dict:
    """Joue un épisode avec ``act(obs) -> action`` et retourne ses métriques."""
    reward_cum, finished, actions, center, dists = 0.0, 0, [], [], [0.0]
    for _, action, reward, _, terminated, info in play(env, act, seed, track, on_reset):
        reward_cum += float(reward)
        finished = int(terminated)
        actions.append(np.asarray(action, dtype=np.float64))
        dists.append(float(info["distance"]))
        center.append(abs(float(np.ravel(get_raw_obs(env)["center_path_distance"])[0])))
    actions, dists, steps = np.array(actions), np.array(dists), len(actions)
    return {
        "reward_cum": reward_cum,
        "finished": finished,
        "lap_steps": steps if finished else float("nan"),
        "steps": steps,
        "distance": dists[-1],  # avancement final le long de la piste
        "mean_speed": dists[-1] / steps,
        "stalled": int((np.diff(dists) < STALL_EPS).sum()),  # pas sans avancement
        "steer_jerk": float(np.abs(np.diff(actions[:, 1])).mean()) if steps > 1 else 0.0,  # |Δdirection| moyen
        "mean_abs_steer": float(np.abs(actions[:, 1]).mean()),
        "mean_accel": float(actions[:, 0].mean()),
        "mean_center_dist": float(np.mean(center)),  # écart moyen au centre de la piste
    }


class CsvLog:
    """Une ligne par appel à ``log`` ; l'en-tête vient de la première ligne. Refuse d'écrire dans un
    fichier existant : un CSV = un run."""

    def __init__(self, path, **context):
        self.path, self.context, self.writer = Path(path), context, None

    def log(self, **row):
        row = {**self.context, **row}
        if self.writer is None:
            self.file = open(self.path, "x", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
