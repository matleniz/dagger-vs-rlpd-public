"""Environnement d'étude : un kart seul sur une piste.

On reconstruit la même chaîne de wrappers que ``supertuxkart/flattened_continuous_actions-v0``
(celle que ML-Arena applique à l'agent), mais avec une prise (``RawObsTap``) juste au-dessus de
l'environnement de base : elle garde l'observation brute pour que le maître puisse la lire pendant
que l'élève voit l'observation aplatie. ``play`` est la boucle d'épisode commune à tout le projet.
"""

import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import (
    ConstantSizedObservations,
    OnlyContinuousActionsWrapper,
    PolarObservations,
)
from pystk2_gymnasium.wrappers import FlattenerWrapper

#: Même valeur que dans l'enregistrement de l'environnement ML-Arena
MAX_EPISODE_STEPS = 1500

# AgentSpec doit être hashable avec le multi-env (comme dans les scripts de pystk_project)
AgentSpec.__hash__ = lambda self: hash(
    (self.name, self.rank_start, self.use_ai, self.camera_mode)
)


class RawObsTap(gym.Wrapper):
    """Garde la dernière observation brute dans ``raw_obs`` (lue par le maître)."""

    raw_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_obs = obs
        return obs, reward, terminated, truncated, info


class PadEmptyKarts(gym.ObservationWrapper):
    """Kart seul : ``karts_position`` est vide et fait planter ``ConstantSizedObservations``.

    On le remplace par une ligne de zéros, ce que ce wrapper ferait de toute façon pour les
    karts absents (remplissage à 0) : l'observation finale est identique.
    """

    def observation(self, obs):
        if len(obs["karts_position"]) == 0:
            obs = {**obs, "karts_position": np.zeros((1, 3), dtype=np.float32)}
        return obs


def make_env(track: str) -> gym.Env:
    """Env élève sur ``track`` : un kart seul, un tour, obs aplatie, action Box(2) = (accélération, direction)."""
    env = gym.make(
        "supertuxkart/full-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False, name="student"),
        track=track,
        num_kart=1,
        laps=1,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    env = RawObsTap(env)
    env = PadEmptyKarts(env)
    env = ConstantSizedObservations(env)
    env = PolarObservations(env)
    env = OnlyContinuousActionsWrapper(env)
    env = FlattenerWrapper(env)
    env.default_track = track  # lu par certains maîtres (team3)
    return env


def set_track(env: gym.Env, track: str):
    """Change la piste du prochain ``reset`` (un processus ne peut ouvrir qu'un env pystk à la fois)."""
    env.default_track = env.unwrapped.default_track = track


def get_raw_obs(env: gym.Env):
    """Observation brute (pas aplatie) du dernier reset/step : celle que lit le maître."""
    while not isinstance(env, RawObsTap):
        env = env.env
    return env.raw_obs


def play(env, act, seed=None, track=None, on_reset=None):
    """Joue un épisode avec ``act(obs) -> action`` et produit, à chaque pas,
    ``(obs, action, reward, next_obs, terminated, info)``. S'arrête à la fin du tour (``terminated``)
    ou à ``MAX_EPISODE_STEPS`` (troncature). ``on_reset`` est appelé juste après ``env.reset``."""
    if track:
        set_track(env, track)
    obs, _ = env.reset(seed=seed)
    if on_reset:
        on_reset()
    done = False
    while not done:
        action = act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        yield obs, action, reward, next_obs, terminated, info
        obs, done = next_obs, terminated or truncated
