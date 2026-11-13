"""DAgger : le maître étiquette les états visités par l'élève, on agrège et on réapprend.

    python -m lab.dagger --seed 0 --demos data/team4_lighthouse_n0.0.npz

Par itération : un épisode où chaque pas exécute l'action du maître avec probabilité beta (sinon
celle de l'élève), puis ``--grad-steps`` pas de gradient (MSE) sur toutes les données agrégées.
``--rollout-noise`` ajoute un bruit gaussien à l'action exécutée (pas à l'étiquette) : l'env est
déterministe, sans bruit un élève qui égale le maître revisite toujours la même trajectoire.
"""

import numpy as np
import torch
import torch.nn.functional as F

from stk_actor.actors import ACTION_DIM, OBS_DIM, MLPPolicy, obs_to_vector

from .envs import make_env, play
from .experts import ExpertPolicy
from .run import Run, parser


def fit(model, opt, X, Y, steps, batch) -> float:
    """``steps`` pas de gradient sur des mini-batchs tirés avec remise ; retourne la dernière perte."""
    for _ in range(steps):
        i = torch.randint(len(X), (min(batch, len(X)),))
        loss = F.mse_loss(model(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def rollout(env, model, expert, beta, rng, seed, track, noise=0.0):
    """Un épisode du mélange maître/élève. Retourne les états visités (vecteurs bruts), les étiquettes
    du maître, le nombre de pas joués et si le tour est fini.

    Les états où le maître freine (récupération de blocage) ne sont pas étiquetés : l'élève n'a pas de
    frein. Ces pas comptent quand même dans le budget.
    """
    xs, ys, steps, finished = [], [], 0, False

    def act(obs):
        label = expert.act()
        if not expert.braking:
            xs.append(obs_to_vector(obs))
            ys.append(label)
        action = label if rng.random() < beta else model.act(obs)
        if noise > 0:
            action = np.clip(action + rng.normal(0.0, noise, size=2), [0.0, -1.0], [1.0, 1.0]).astype(np.float32)
        return action

    for *_, finished, _ in play(env, act, seed, track, expert.reset):
        steps += 1
    return np.reshape(xs, (-1, OBS_DIM)), np.reshape(ys, (-1, ACTION_DIM)), steps, finished


def main(a):
    run = Run(a, "dagger")
    model = MLPPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    demo = run.load_demos(model)
    X, Y, env_steps = demo["obs"], demo["action"], len(demo["obs"])
    fit(model, opt, X, Y, a.init_steps, a.batch)
    run.evaluate(model, env_steps, force=True)

    env = make_env(a.tracks[0])
    expert = ExpertPolicy(env, a.team)
    it = 0
    while env_steps < a.steps:
        beta = a.beta0 * a.beta_decay**it
        xs, ys, steps, finished = rollout(env, model, expert, beta, run.rng, run.seed(), run.track(it), a.rollout_noise)
        X = torch.cat([X, model.normalize(torch.as_tensor(xs, dtype=torch.float32))])
        Y = torch.cat([Y, torch.as_tensor(ys, dtype=torch.float32)])
        loss = fit(model, opt, X, Y, a.grad_steps, a.batch)
        it, env_steps = it + 1, env_steps + steps
        run.log_train(env_steps, iteration=it, track=run.track(it - 1), beta=beta, steps=steps,
                      finished=int(finished), dataset=len(X), loss=loss, grad_steps=a.init_steps + it * a.grad_steps)
        run.evaluate(model, env_steps, force=env_steps >= a.steps)
    env.close()


if __name__ == "__main__":
    ap = parser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--init-steps", type=int, default=1000, help="pas de gradient sur les démos seules")
    ap.add_argument("--grad-steps", type=int, default=200, help="pas de gradient par itération")
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--rollout-noise", type=float, default=0.0, help="écart-type du bruit sur l'action exécutée")
    main(ap.parse_args())
