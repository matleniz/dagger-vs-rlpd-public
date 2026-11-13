"""Code partagé entre l'entraînement (lab/) et la soumission ML-Arena (pystk_actor.py).

Seulement torch + numpy : c'est tout ce qu'on peut utiliser sur la plateforme.
"""

import numpy as np
import torch
import torch.nn as nn

#: Tailles des variables discrètes de l'observation (``MultiDiscrete`` de l'env aplati)
DISCRETE_NVEC = (10, 7, 7, 7, 7, 7, 2, 4, 6, 11)
CONTINUOUS_DIM = 92
#: Dimension du vecteur d'entrée du réseau : 92 continues + one-hot des 10 variables discrètes
OBS_DIM = CONTINUOUS_DIM + sum(DISCRETE_NVEC)
ACTION_DIM = 2  # (acceleration dans [0, 1], steer dans [-1, 1])

_EYES = [np.eye(n, dtype=np.float32) for n in DISCRETE_NVEC]

#: Les entrées normalisées sont bornées à ±CLIP (une piste jamais vue peut sortir des stats).
CLIP = 10.0


def obs_to_vector(obs) -> np.ndarray:
    """Observation aplatie (dict ``continuous`` / ``discrete`` / ``kart_type``) -> vecteur float32 brut.

    Les continues sont recopiées, les variables discrètes encodées en one-hot, ``kart_type`` (texte)
    est ignoré. La normalisation est faite par le modèle (``MLPPolicy.normalize``).

    L'env sort parfois une valeur discrète égale à la taille annoncée (vu le 13/11 : 6 pour la variable
    d'indice 8, de taille 6, sur mines) : elle est ramenée à la dernière catégorie au lieu de planter.
    """
    continuous = np.asarray(obs["continuous"], dtype=np.float32)
    discrete = np.clip(np.asarray(obs["discrete"], dtype=int), 0, np.array(DISCRETE_NVEC) - 1)
    return np.concatenate([continuous] + [eye[d] for eye, d in zip(_EYES, discrete)])


def squash(y):
    """Sortie tanh -> action de l'env : accélération dans [0, 1], direction dans [-1, 1]."""
    return torch.cat([(y[..., :1] + 1) / 2, y[..., 1:]], -1)


class MLPPolicy(nn.Module):
    """Politique déterministe : vecteur d'observation normalisé -> (acceleration, steer).

    Les stats de normalisation (``obs_mean``, ``obs_std``) sont des tampons du modèle : elles sont
    sauvées avec les poids, donc chaque modèle garde les stats de ses propres données d'entraînement.
    """

    def __init__(self, obs_dim=OBS_DIM, hidden=(256, 256), act_dim=ACTION_DIM):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.obs_mean) / self.obs_std).clamp(-CLIP, CLIP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` : vecteurs déjà normalisés."""
        return squash(torch.tanh(self.net(x)))

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        """Action numpy (float32, taille 2) pour une observation de l'env."""
        return self(self.normalize(torch.as_tensor(obs_to_vector(obs)))).numpy()


def make_random_actor(action_space):
    def actor(state, obs):
        return action_space.sample()

    return actor
