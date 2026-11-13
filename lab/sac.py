"""SAC (acteur gaussien à tanh, critiques jumelles avec LayerNorm, température apprise).

Indépendant de l'env et de la provenance des données : ``SAC.update(batch)`` prend un dict de
tenseurs ``obs, action, reward, next_obs, terminated`` (observations normalisées).
Simplifications par rapport au papier RLPD : 2 critiques (pas 10), log_std indépendant de l'état.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stk_actor.actors import ACTION_DIM, OBS_DIM, MLPPolicy, squash

HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


class Actor(MLPPolicy):
    """``MLPPolicy`` (la moyenne, c'est elle qu'on évalue et qu'on soumet) + un log_std global."""

    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def sample(self, x):
        """Action de l'env (acc, steer) échantillonnée, et sa log-probabilité (dans l'espace tanh)."""
        mu, log_std = self.net(x), self.log_std.clamp(-5, 1)
        eps = torch.randn_like(mu)
        u = mu + log_std.exp() * eps
        gauss = (-0.5 * eps**2 - log_std - HALF_LOG_2PI).sum(-1)
        tanh_corr = (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)  # log(1 - tanh(u)²), stable
        return squash(torch.tanh(u)), gauss - tanh_corr


def critic(h=256):
    return nn.Sequential(nn.Linear(OBS_DIM + ACTION_DIM, h), nn.LayerNorm(h), nn.ReLU(),
                         nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, 1))


def q_values(critics, obs, action):
    """(n_critiques, batch)"""
    x = torch.cat([obs, action], -1)
    return torch.stack([q(x).squeeze(-1) for q in critics])


def descend(opt, loss):
    opt.zero_grad()
    loss.backward()
    opt.step()


class SAC:
    def __init__(self, lr=3e-4, gamma=0.99, tau=0.005, alpha0=0.1, target_entropy=-1.0):
        self.gamma, self.tau, self.target_entropy = gamma, tau, target_entropy
        self.actor = Actor()
        self.critics = nn.ModuleList([critic(), critic()])
        self.targets = deepcopy(self.critics)
        self.log_alpha = torch.tensor(np.log(alpha0), dtype=torch.float32, requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critics = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def update(self, b) -> float:
        """Une mise à jour (critiques, acteur, température, cibles) ; retourne la perte des critiques."""
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():  # cible : seul ``terminated`` coupe le bootstrap (pas la troncature)
            next_action, next_logp = self.actor.sample(b["next_obs"])
            next_q = q_values(self.targets, b["next_obs"], next_action).min(0).values
            y = b["reward"] + self.gamma * (1 - b["terminated"]) * (next_q - alpha * next_logp)
        critic_loss = ((q_values(self.critics, b["obs"], b["action"]) - y) ** 2).mean()
        descend(self.opt_critics, critic_loss)

        action, logp = self.actor.sample(b["obs"])
        descend(self.opt_actor, (alpha * logp - q_values(self.critics, b["obs"], action).min(0).values).mean())
        descend(self.opt_alpha, -(self.log_alpha * (logp.detach() + self.target_entropy)).mean())

        with torch.no_grad():
            for p, target in zip(self.critics.parameters(), self.targets.parameters()):
                target.lerp_(p, self.tau)
        return critic_loss.item()
