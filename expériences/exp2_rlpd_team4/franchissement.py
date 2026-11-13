"""Script d'investigation du smoke test, tel qu'exécuté au commit a8431ab (API de ``lab/`` d'alors :
``lab.expert_adapter``, poids sans stats de normalisation). Gardé comme trace, pas maintenu."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
torch.set_num_threads(1)
from stk_actor.actors import MLPPolicy
from lab.envs import make_env, get_raw_obs, MAX_EPISODE_STEPS
from lab.expert_adapter import ExpertPolicy
env=make_env("lighthouse"); ex=ExpertPolicy(env,4)
def mdl(p):
    m=MLPPolicy(); m.load_state_dict(torch.load(p)); m.eval(); return lambda o: m.act(o)
pols={"maître":lambda o: ex.act(),"DAgger":mdl(sys.argv[1]),"RLPD":mdl(sys.argv[2])}
f=lambda x: float(np.asarray(x).ravel()[0])
for name,pol in pols.items():
    rows=[]; rng=np.random.default_rng(0)
    for ep in range(12):
        obs,_=env.reset(seed=int(rng.integers(1<<30))); ex.reset()
        prev=None
        for t in range(MAX_EPISODE_STEPS):
            a=pol(obs); obs,r,te,tr,info=env.step(a); raw=get_raw_obs(env)
            if te or tr: break
            prev=(f(raw["distance_down_track"]),f(raw["center_path_distance"]),float(np.linalg.norm(raw["velocity"])))
        k=env.unwrapped.world.karts[0]
        rows.append((info["distance"]>1000, t+1, prev[0], prev[1], prev[2], float(k.distance_down_track), float(a[1]), float(a[0]), f(raw["center_path_distance"]), float(np.linalg.norm(raw["velocity"]))))
    print(f"\n== {name} : [avant-dernier pas: down_track, écart centre, vitesse] -> [dernier pas: down_track, écart centre, vitesse]")
    for d,n,pd,pc,pv,fd,st,ac,fc,fv in rows:
        print(f"  {'DOUBLÉ' if d else 'normal'} pas {n:3d} | {pd:6.1f} {pc:+5.1f} {pv:4.1f} -> {fd:6.1f} {fc:+5.1f} {fv:4.1f} | steer {st:+.2f}")
