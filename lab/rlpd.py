"""RLPD : SAC en ligne dont chaque batch est tiré pour moitié dans les démonstrations du maître.

    python -m lab.rlpd --seed 0 --demos data/team4_lighthouse_n0.0.npz data/team4_lighthouse_n0.1.npz

Budget = pas en ligne + transitions des démos (même règle que DAgger). Récompense d'entraînement :
celle de l'env, plafonnée par pas (``--reward-cap``).
``--demo-ratio 0`` = SAC seul (référence sans maître) : aucun batch ne contient de démo et le budget part
de 0 ; les démos ne servent plus qu'à la normalisation des observations.
"""

import torch

from stk_actor.actors import obs_to_vector

from .envs import make_env, play
from .run import Run, parser
from .sac import SAC


class Buffer:
    """Tampon de transitions pré-alloué, mêmes clés et formes que ``like``."""

    def __init__(self, like: dict, capacity: int):
        self.data = {k: torch.zeros(capacity, *v.shape[1:]) for k, v in like.items()}
        self.n = 0

    def add(self, **row):
        for k, v in row.items():
            self.data[k][self.n] = torch.as_tensor(v)
        self.n += 1


def symmetric_batch(offline: dict, online: Buffer, size: int, demo_ratio: float = 0.5) -> dict:
    """Part ``demo_ratio`` de démos (RLPD : la moitié), le reste en ligne (tout en démos tant que le
    tampon en ligne est vide)."""
    n_on = size - int(size * demo_ratio) if online.n else 0
    i_off, i_on = torch.randint(len(offline["obs"]), (size - n_on,)), torch.randint(max(online.n, 1), (n_on,))
    return {k: torch.cat([offline[k][i_off], online.data[k][i_on]]) for k in offline}


def main(a):
    run = Run(a, "rlpd")
    sac = SAC(a.lr, a.gamma, a.tau, a.alpha0, a.target_entropy)
    actor = sac.actor
    offline = run.load_demos(actor)
    offline["reward"].clamp_(max=a.reward_cap)
    online = Buffer(offline, a.steps)
    env_steps, updates, episode = len(offline["obs"]) if a.demo_ratio > 0 else 0, 0, 0
    run.evaluate(actor, env_steps, force=True)

    def vector(obs):
        return actor.normalize(torch.as_tensor(obs_to_vector(obs)))

    @torch.no_grad()
    def act(obs):
        return actor.sample(vector(obs))[0].numpy()

    env = make_env(a.tracks[0])
    while env_steps < a.steps:
        ret, steps, critic_loss = 0.0, 0, float("nan")
        for obs, action, reward, next_obs, terminated, info in play(env, act, run.seed(), run.track(episode)):
            online.add(obs=vector(obs), action=action, reward=min(reward, a.reward_cap), next_obs=vector(next_obs),
                       terminated=float(terminated))
            env_steps, steps, ret = env_steps + 1, steps + 1, ret + reward
            if online.n >= a.warmup:
                for _ in range(a.utd):
                    critic_loss = sac.update(symmetric_batch(offline, online, a.batch, a.demo_ratio))
                updates += a.utd
            run.evaluate(actor, env_steps, force=env_steps >= a.steps)
            if env_steps >= a.steps:
                break
        episode += 1
        run.log_train(env_steps, episode=episode, track=run.track(episode - 1), steps=steps, finished=int(terminated),
                      reward_cum=ret, distance=info["distance"], alpha=sac.alpha, critic_loss=critic_loss, grad_steps=updates)
    env.close()


if __name__ == "__main__":
    ap = parser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--alpha0", type=float, default=0.1)
    ap.add_argument("--target-entropy", type=float, default=-1.0)
    ap.add_argument("--utd", type=int, default=4, help="mises à jour par pas d'env")
    ap.add_argument("--demo-ratio", type=float, default=0.5, help="part de démos dans chaque batch (0 : SAC seul)")
    ap.add_argument("--warmup", type=int, default=500, help="pas en ligne avant la première mise à jour")
    #: Au pas final, l'env ajoute parfois ~+88 (distance comptée deux fois, voir expériences/exp2) alors
    #: que la valeur normale est ~10 ; les démos du maître culminent à 10,14.
    ap.add_argument("--reward-cap", type=float, default=10.2, help="plafond de la récompense par pas (inf : aucun)")
    main(ap.parse_args())
