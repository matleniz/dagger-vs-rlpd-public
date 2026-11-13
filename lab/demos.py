"""Démonstrations : le maître joue seul, on enregistre ses transitions.

    python -m lab.demos --team 4 --tracks lighthouse                         # 1 épisode propre
    python -m lab.demos --team 4 --tracks lighthouse --episodes 10 --noise 0.1

Un fichier ``data/team<T>_<piste>_n<bruit>.npz`` par piste, avec pour chaque pas :
  obs, next_obs : (N, OBS_DIM) vecteurs BRUTS (``obs_to_vector``, non normalisés) ;
  action        : (N, 2) action exécutée ; reward : (N,) récompense brute de l'env ;
  terminated    : (N,) ; episode : (N,) indice d'épisode ; lap_steps : (n_episodes,) (-1 = échec).

L'env est déterministe : sans bruit, tous les épisodes sont identiques. ``--noise`` ajoute un bruit
gaussien à l'action exécutée (et enregistrée) pour diversifier les états visités.
``--clean-labels`` enregistre à la place l'action du maître avant bruit (étiquettes propres pour
l'imitation ; fichier ``..._n<bruit>_clean.npz``, à ne pas donner à RLPD : ce n'est pas l'action jouée).
"""

import argparse
from pathlib import Path

import numpy as np

from stk_actor.actors import CONTINUOUS_DIM, OBS_DIM, obs_to_vector

from .envs import make_env, play
from .experts import ExpertPolicy

KEYS = ("obs", "action", "reward", "next_obs", "terminated")


def collect(env, team, track, episodes, noise=0.0, seed=0, clean_labels=False) -> dict:
    rng = np.random.default_rng(seed)
    expert = ExpertPolicy(env, team)

    def act(obs):
        action = label[0] = expert.act()
        if noise > 0:
            action = np.clip(action + rng.normal(0.0, noise, size=2), [0.0, -1.0], [1.0, 1.0]).astype(np.float32)
        return action

    rows, lap_steps, label = [], [], [None]
    for ep in range(episodes):
        n = len(rows)
        for obs, action, reward, next_obs, terminated, _ in play(env, act, seed + ep, track, expert.reset):
            rows.append((obs_to_vector(obs), label[0] if clean_labels else action, reward, obs_to_vector(next_obs), terminated, ep))
        lap_steps.append(len(rows) - n if terminated else -1)
    data = {k: np.array(col, dtype=np.float32) for k, col in zip((*KEYS, "episode"), zip(*rows))}
    return {**data, "lap_steps": np.array(lap_steps)}


def load(paths) -> dict:
    """Concatène des fichiers de démonstrations -> dict de tableaux float32 (clés ``KEYS``)."""
    files = [np.load(p) for p in paths]
    return {k: np.concatenate([f[k] for f in files]) for k in KEYS}


def obs_stats(obs: np.ndarray):
    """(mean, std) de normalisation à partir de vecteurs bruts. Seules les colonnes continues sont
    normalisées (one-hot : mean 0, std 1) ; colonnes quasi constantes : std = 1 (pas d'explosion)."""
    mean, std = np.zeros(OBS_DIM, dtype=np.float32), np.ones(OBS_DIM, dtype=np.float32)
    c = obs[:, :CONTINUOUS_DIM]
    mean[:CONTINUOUS_DIM] = c.mean(0)
    std[:CONTINUOUS_DIM] = np.where(c.std(0) < 1e-3, 1.0, c.std(0))
    return mean, std


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, required=True)
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-labels", action="store_true", help="enregistre l'action du maître avant bruit")
    ap.add_argument("--dir", default="data")
    a = ap.parse_args()
    Path(a.dir).mkdir(exist_ok=True)
    env = make_env(a.tracks[0])
    for track in a.tracks:
        d = collect(env, a.team, track, a.episodes, a.noise, a.seed, a.clean_labels)
        out = f"{a.dir}/team{a.team}_{track}_n{a.noise}{"_clean" if a.clean_labels else ""}.npz"
        np.savez_compressed(out, **d)
        print(f"{out}: {len(d['obs'])} transitions, tours {d['lap_steps'].tolist()}")
    env.close()
