"""Script d'investigation du smoke test, tel qu'exécuté au commit a8431ab (API de ``lab/`` d'alors :
``lab.expert_adapter``, poids sans stats de normalisation). Gardé comme trace, pas maintenu."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from stk_actor.actors import MLPPolicy
from lab.envs import make_env, get_raw_obs, MAX_EPISODE_STEPS
m=MLPPolicy(); m.load_state_dict(torch.load('runs/rlpd/pilot_utd4/model.pth')); m.eval()
env=make_env("lighthouse"); rng=np.random.default_rng(0)
for ep in range(6):
    obs,_=env.reset(seed=int(rng.integers(1<<30))); hist=[]
    for t in range(MAX_EPISODE_STEPS):
        a=np.clip(m.act(obs)+rng.normal(0,0.1,2).astype(np.float32),[0,-1],[1,1])
        obs,r,te,tr,info=env.step(a)
        k=env.unwrapped.world.karts[0]
        hist.append((t+1,round(r,2),round(info["distance"],1),round(float(k.distance_down_track),1),round(float(k.overall_distance),1),k.has_finished_race))
        if te or tr: break
    tag="DOUBLÉ" if info["distance"]>1000 else "normal"
    print(f"ep{ep} {tag}"); [print("   (pas, récompense, info.distance, distance_down_track, overall_distance, fini)",h) for h in hist[-3:]]
