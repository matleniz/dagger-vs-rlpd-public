"""Maîtres de ``pystk_project`` (importés, jamais modifiés) joués dans l'env élève.

Le maître lit l'observation *brute* ; l'élève voit l'observation *aplatie*. L'action du maître
(dict avec acceleration, steer, brake, drift, nitro, rescue, fire) est projetée sur le Box(2)
(acceleration, steer) de l'élève : brake, drift, nitro, rescue et fire sont perdus.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np

from .envs import get_raw_obs

#: ``src/`` du dépôt pystk_project : variable d'environnement, sinon le dossier voisin de ce projet
PROJECT_SRC = Path(os.environ.get("PYSTK_PROJECT_SRC") or Path(__file__).resolve().parents[2] / "pystk_project" / "src")

#: team -> (module, classe). team6 et team7 jouent au hasard : ce ne sont pas de vrais maîtres.
#: team2 est INUTILISABLE ici : il sort d'un blocage en reculant avec ``brake``, absent de Box(2),
#: et se croit bloqué dès le départ (voir expériences/exp0_maitres).
EXPERTS = {
    1: ("agents.team1.agent1", "Agent1"),
    2: ("agents.team2.agent2", "Agent2"),
    3: ("agents.team3.agent3", "Agent3"),
    4: ("agents.team4.agent4", "Agent4"),
    5: ("agents.team5.agent5", "Agent5"),
}

#: Kart factice très loin derrière : les maîtres supposent qu'il y a au moins un autre kart
#: (ex. team1 lit ``karts_position[0]``), alors que le nôtre est seul. Ils ne s'en servent que pour
#: les items et l'évitement. (L'élève, lui, voit une ligne de zéros : ``envs.PadEmptyKarts``.)
DUMMY_KART = np.array([[0.0, 0.0, -1000.0]], dtype=np.float32)


def load_expert_class(team: int):
    """Importe la classe du maître ``team``.

    ``PROJECT_SRC`` est ajouté *en dernier* dans ``sys.path`` : la copie locale de
    ``pystk2_gymnasium`` du dépôt ne doit pas masquer celle du venv.
    team2 charge sa config avec un chemin relatif à ``src/main`` : on s'y place pendant l'import.
    """
    if str(PROJECT_SRC) not in sys.path:
        sys.path.append(str(PROJECT_SRC))
    module, cls = EXPERTS[team]
    cwd = os.getcwd()
    os.chdir(PROJECT_SRC / "main")
    try:
        return getattr(importlib.import_module(module), cls)
    finally:
        os.chdir(cwd)


class ExpertPolicy:
    """Maître ``team`` dans ``env`` : ``reset()`` après chaque ``env.reset()``, puis ``act()``.

    Le maître reçoit seulement l'observation brute (``get_raw_obs``) : on n'appelle pas
    ``Agent.step()`` de pystk_project, qui lit ``env.unwrapped.world`` et attend un env multi-karts.
    """

    def __init__(self, env, team: int):
        self.env, self.cls = env, load_expert_class(team)
        self.braking = False

    def reset(self):
        """Recrée le maître : état interne remis à zéro, et piste courante relue (team3)."""
        self.expert = self.cls(self.env, path_lookahead=3)

    def full_action(self) -> dict:
        """Action complète du maître (dict) pour l'observation courante."""
        raw = get_raw_obs(self.env)
        if len(raw["karts_position"]) == 0:
            raw = {**raw, "karts_position": DUMMY_KART}
        return self.expert.choose_action(raw)

    def act(self, obs=None) -> np.ndarray:
        """Action Box(2). ``braking`` dit si le maître freinait (récupération de blocage, non
        reproductible par l'élève)."""
        action = self.full_action()
        self.braking = bool(action.get("brake"))
        return np.array([np.clip(action["acceleration"], 0, 1), np.clip(action["steer"], -1, 1)], dtype=np.float32)
