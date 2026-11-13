"""Évaluation d'une politique : métriques par épisode (``lab/metrics.py``), une ou plusieurs pistes.

    python -m lab.evaluate --team 4 --tracks lighthouse hacienda            # un maître
    python -m lab.evaluate --weights runs/dagger/base/seed0/model.pth --tracks lighthouse

``eval_tracks`` sert aussi pendant l'entraînement (``lab/run.py``) : une piste par processus, parce
qu'un processus ne peut ouvrir qu'un env pystk à la fois et que celui de l'entraînement reste ouvert.
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from stk_actor.actors import MLPPolicy

from .envs import make_env
from .experts import ExpertPolicy
from .metrics import CsvLog, episode_metrics

_POOL = None


def policy_state(model) -> dict:
    """Poids d'un ``MLPPolicy`` (ou d'une sous-classe : seules les clés de ``MLPPolicy`` sont gardées)."""
    keys = MLPPolicy().state_dict().keys()
    return {k: v for k, v in model.state_dict().items() if k in keys}


def eval_track(track, state=None, team=None, episodes=1) -> list:
    """``episodes`` épisodes sur ``track`` (graines 0, 1, ...) pour des poids ou pour un maître."""
    torch.set_num_threads(1)
    env = make_env(track)
    if team:
        expert = ExpertPolicy(env, team)
        act, on_reset = expert.act, expert.reset
    else:
        model = MLPPolicy()
        model.load_state_dict(state)
        act, on_reset = model.act, None
    rows = [{"track": track, "episode": i, **episode_metrics(env, act, seed=i, on_reset=on_reset)} for i in range(episodes)]
    env.close()
    return rows


def eval_tracks(tracks, **kw) -> list:
    """``eval_track`` sur chaque piste, en parallèle ; retourne toutes les lignes."""
    global _POOL
    if _POOL is None:
        workers = int(os.environ.get("LAB_EVAL_WORKERS", 8))  # 1 instance STK (~0,3 Go) par processus
        _POOL = ProcessPoolExecutor(min(len(tracks), workers), mp_context=mp.get_context("spawn"))
    return [row for rows in _POOL.map(partial(eval_track, **kw), tracks) for row in rows]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--team", type=int)
    who.add_argument("--weights", help="state_dict d'un MLPPolicy (.pth)")
    ap.add_argument("--tracks", nargs="+", default=["lighthouse"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--csv", help="écrit les métriques de chaque épisode dans ce CSV (nouveau fichier)")
    a = ap.parse_args()
    state = torch.load(a.weights, map_location="cpu") if a.weights else None
    rows = eval_tracks(a.tracks, state=state, team=a.team, episodes=a.episodes)
    log = CsvLog(a.csv, who=a.weights or f"team{a.team}") if a.csv else None
    for row in rows:
        print(f"{row['track']} ép. {row['episode']}: fini={row['finished']} tour={row['lap_steps']} R={row['reward_cum']:.1f}")
        if log:
            log.log(**row)
