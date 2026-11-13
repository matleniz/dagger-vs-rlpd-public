"""Compare les maîtres : action tronquée (Box(2), env élève) contre action complète (env de base).

    python -m expériences.exp0_maitres.compare_experts --tracks lighthouse       # détail sur une piste
    python -m expériences.exp0_maitres.compare_experts --tracks all --no-detail  # toutes les pistes

Résultat toutes pistes : ``compare_experts_all_tracks.md`` (même dossier).

L'env est déterministe : une seule exécution par (maître, piste, mode) suffit.
"""

import argparse
import time

import numpy as np

import gymnasium as gym
from pystk2_gymnasium import AgentSpec

from lab.envs import MAX_EPISODE_STEPS, RawObsTap, make_env
from lab.experts import ExpertPolicy

MAPS = ['abyss', 'black_forest', 'candela_city', 'cocoa_temple', 'cornfield_crossing', 'fortmagma',
        'gran_paradiso_island', 'hacienda', 'lighthouse', 'mines', 'minigolf', 'olivermath',
        'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak', 'stk_enterprise',
        'volcano_island', 'xr591', 'zengarden']
DISCRETE = ("brake", "drift", "nitro", "rescue", "fire")


def make_full_env(track):
    """Env de base (``full-v0``) : l'action est le dict complet (brake, drift, nitro, rescue, fire)."""
    env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False, name="full"),
                   track=track, num_kart=1, laps=1, max_episode_steps=MAX_EPISODE_STEPS)
    env = RawObsTap(env)
    env.default_track = track
    return env


def run(track, team, full):
    """Un épisode. Retourne dict : lap (None si échec), distance, stats d'action, temps de calcul."""
    env = make_full_env(track) if full else make_env(track)
    ex = ExpertPolicy(env, team)
    env.reset(seed=0)
    ex.reset()
    used = dict.fromkeys(DISCRETE, 0)
    steer, acc, t_calc, lap = [], [], 0.0, None
    for step in range(1, MAX_EPISODE_STEPS + 1):
        t0 = time.perf_counter()
        a = ex.full_action()
        t_calc += time.perf_counter() - t0
        for k in DISCRETE:
            used[k] += bool(a[k])
        steer.append(float(a["steer"]))
        acc.append(float(a["acceleration"]))
        if full:
            action = {**a, **{k: int(bool(a[k])) for k in DISCRETE},
                      "acceleration": np.float32(np.clip(a["acceleration"], 0, 1)),
                      "steer": np.float32(np.clip(a["steer"], -1, 1))}
        else:
            action = np.array([np.clip(a["acceleration"], 0, 1), np.clip(a["steer"], -1, 1)], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if terminated:
            lap = step
        if terminated or truncated:
            break
    env.close()
    n = step
    return dict(lap=lap, distance=float(info.get("distance", 0.0)),
                **{f"%{k}": 100 * v / n for k, v in used.items()},
                d_steer=float(np.mean(np.abs(np.diff(steer)))) if n > 1 else 0.0,
                acc_lt1=100 * float(np.mean(np.array(acc) < 0.99)),
                ms=1000 * t_calc / n)


def fmt(lap):
    return "ÉCHEC" if lap is None else str(lap)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="lighthouse")
    ap.add_argument("--experts", default="1,2,3,4,5")
    ap.add_argument("--no-detail", action="store_true")
    a = ap.parse_args()
    tracks = MAPS if a.tracks == "all" else a.tracks.split(",")
    teams = [int(t) for t in a.experts.split(",")]
    res = {}
    for tr in tracks:
        for t in teams:
            for full in (False, True):
                try:
                    res[tr, t, full] = run(tr, t, full)
                except Exception as e:
                    res[tr, t, full] = dict(lap=None, error=type(e).__name__)
        print(f"[{tr}] " + " ".join(f"team{t}:{fmt(res[tr,t,False]['lap'])}/{fmt(res[tr,t,True]['lap'])}" for t in teams), flush=True)
    print("\n(tronqué / complet, en pas)\n")
    print("| piste | " + " | ".join(f"team{t}" for t in teams) + " |")
    print("|---|" + "---|" * len(teams))
    for tr in tracks:
        print(f"| {tr} | " + " | ".join(f"{fmt(res[tr,t,False]['lap'])} / {fmt(res[tr,t,True]['lap'])}" for t in teams) + " |")
    print("\n| maître | pistes finies (tronqué) | pistes finies (complet) | gain moyen complet vs tronqué |")
    print("|---|---|---|---|")
    for t in teams:
        both = [(res[tr,t,False]['lap'], res[tr,t,True]['lap']) for tr in tracks]
        ok_t = sum(x is not None for x, _ in both); ok_f = sum(y is not None for _, y in both)
        gains = [100 * (x - y) / x for x, y in both if x and y]
        print(f"| team{t} | {ok_t}/{len(tracks)} | {ok_f}/{len(tracks)} | {np.mean(gains):+.1f} % (sur {len(gains)} pistes) |" if gains else f"| team{t} | {ok_t}/{len(tracks)} | {ok_f}/{len(tracks)} | n/a |")
    if not a.no_detail:
        for tr in tracks:
            print(f"\nDétail sur {tr} (action complète) :\n")
            print("| maître | tour tronqué | tour complet | %brake | %drift | %nitro | %rescue | %fire | |Δsteer| moy | %pas accél.<1 | ms/pas |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for t in teams:
                r, f = res[tr,t,False], res[tr,t,True]
                if "error" in f:
                    print(f"| team{t} | {fmt(r['lap'])} | erreur {f['error']} |"); continue
                print(f"| team{t} | {fmt(r['lap'])} | {fmt(f['lap'])} | " + " | ".join(f"{f['%'+k]:.0f}" for k in DISCRETE)
                      + f" | {f['d_steer']:.3f} | {f['acc_lt1']:.0f} | {f['ms']:.2f} |")
