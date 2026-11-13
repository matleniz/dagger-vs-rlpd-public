from pathlib import Path

import torch

from .actors import MLPPolicy, make_random_actor

#: Même environnement que celui de l'étude (voir lab/envs.py)
env_name = "supertuxkart/flattened_continuous_actions-v0"

#: Player name (you must change that)
player_name = "dagger-vs-rlpd"


def get_actor(module_dir: Path, observation_space, action_space):
    weights = Path(module_dir) / "pystk_actor.pth"
    if not weights.exists():  # pas encore entraîné : acteur aléatoire
        return make_random_actor(action_space)

    policy = MLPPolicy()
    policy.load_state_dict(torch.load(weights, map_location="cpu"))
    policy.eval()

    def actor(state, obs):
        return policy.act(obs)

    return actor
