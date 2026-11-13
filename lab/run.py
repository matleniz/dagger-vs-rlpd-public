"""Un run d'entraînement = un dossier neuf, commun à DAgger et RLPD :

    config.json   arguments, méthode, commit git
    eval.csv      une ligne par (point d'évaluation, piste, épisode) : courbes d'apprentissage
    train.csv     une ligne par épisode d'entraînement (statistiques propres à la méthode)
    model.pth     ``state_dict`` de ``MLPPolicy``, réécrit à chaque évaluation (un run interrompu
                  garde son dernier modèle)

Abscisse commune : ``env_steps`` = pas d'env consommés, démonstrations comprises.
"""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from stk_actor.actors import MLPPolicy

from . import demos
from .evaluate import eval_tracks, policy_state
from .metrics import CsvLog


def parser() -> argparse.ArgumentParser:
    """Arguments communs à toutes les méthodes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base", help="nom de la variante -> runs/<méthode>/<name>/seed<k>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000, help="budget en pas d'env, démos comprises")
    ap.add_argument("--team", type=int, default=4, help="maître")
    ap.add_argument("--tracks", nargs="+", default=["lighthouse"], help="pistes d'entraînement (en alternance)")
    ap.add_argument("--test-tracks", nargs="*", default=["hacienda", "scotland"])
    ap.add_argument("--demos", nargs="+", required=True, help="fichiers de lab.demos")
    ap.add_argument("--stats", nargs="+", help="fichiers dont on tire la normalisation (défaut : --demos)")
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--eval-early", type=int, default=500, help="pas d'évaluation jusqu'à --early-until")
    ap.add_argument("--early-until", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", help="dossier du run (défaut : runs/<méthode>/<name>/seed<k>)")
    return ap


class Run:
    def __init__(self, a, method):
        self.a, self.t0 = a, time.time()
        self.dir = Path(a.out or f"runs/{method}/{a.name}/seed{a.seed}")
        self.dir.mkdir(parents=True)  # échoue si le dossier existe : jamais deux runs mélangés
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        (self.dir / "config.json").write_text(json.dumps({"method": method, "commit": commit, "host": socket.gethostname(), **vars(a)}, indent=1))
        context = {"method": method, "name": a.name, "seed": a.seed}
        self.eval_log, self.train_log = CsvLog(self.dir / "eval.csv", **context), CsvLog(self.dir / "train.csv", **context)
        early = range(a.eval_early, a.early_until + 1, a.eval_early)
        self.pending = sorted({*early, *range(a.eval_every, a.steps, a.eval_every)})
        torch.set_num_threads(a.threads)
        torch.manual_seed(a.seed)
        self.rng = np.random.default_rng(a.seed)

    def load_demos(self, model: MLPPolicy) -> dict:
        """Fixe la normalisation de ``model`` (stats de ``--stats``, sinon des démos) et retourne les
        démos en tenseurs, observations normalisées."""
        d = demos.load(self.a.demos)
        mean, std = demos.obs_stats(demos.load(self.a.stats)["obs"] if self.a.stats else d["obs"])
        model.obs_mean.copy_(torch.as_tensor(mean))
        model.obs_std.copy_(torch.as_tensor(std))
        d = {k: torch.as_tensor(v) for k, v in d.items()}
        return {**d, "obs": model.normalize(d["obs"]), "next_obs": model.normalize(d["next_obs"])}

    def track(self, episode: int) -> str:
        return self.a.tracks[episode % len(self.a.tracks)]

    def seed(self) -> int:
        return int(self.rng.integers(1 << 30))

    def log_train(self, env_steps, **row):
        self.train_log.log(env_steps=env_steps, wall_s=round(time.time() - self.t0, 1), **row)
        print(f"[{env_steps}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    def evaluate(self, model, env_steps, force=False):
        """Évalue et sauve ``model`` si un point d'évaluation est atteint (ou ``force``)."""
        due = [p for p in self.pending if p <= env_steps]
        if not (due or force):
            return
        self.pending = self.pending[len(due):]
        state = policy_state(model)
        torch.save(state, self.dir / "model.pth")
        tracks = dict.fromkeys([*self.a.tracks, *self.a.test_tracks])
        for row in eval_tracks(list(tracks), state=state, episodes=self.a.eval_episodes):
            self.eval_log.log(env_steps=env_steps, wall_s=round(time.time() - self.t0, 1), **row)
            print(f"  éval [{env_steps}] {row['track']}: fini={row['finished']} tour={row['lap_steps']} R={row['reward_cum']:.1f}", flush=True)
