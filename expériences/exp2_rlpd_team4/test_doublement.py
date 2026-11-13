"""Script d'investigation du smoke test, tel qu'exécuté au commit a8431ab (API de ``lab/`` d'alors :
``lab.expert_adapter``, poids sans stats de normalisation). Gardé comme trace, pas maintenu."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from stk_actor.actors import MLPPolicy
from lab.envs import make_env, MAX_EPISODE_STEPS
from lab.expert_adapter import ExpertPolicy
env=make_env("lighthouse")
def run(policy,noise,n=15,seed=0):
    rng=np.random.default_rng(seed); out=[]
    for ep in range(n):
        obs,_=env.reset(seed=int(rng.integers(1<<30)))
        if hasattr(policy,'reset'): policy.reset()
        for t in range(MAX_EPISODE_STEPS):
            a=np.clip(policy(obs)+rng.normal(0,noise,2),[0,-1],[1,1]).astype(np.float32)
            obs,r,te,tr,info=env.step(a)
            if te or tr: break
        out.append(te and info["distance"]>1000)
    return sum(out),n
def mdl(path):
    m=MLPPolicy(); m.load_state_dict(torch.load(path)); m.eval(); return lambda o: m.act(o)
ex=ExpertPolicy(env,4)
class E:
    def __call__(s,o): return ex.act()
    def reset(s): ex.reset()
for name,pol in (("maître team4",E()),("DAgger",mdl(sys.argv[1])),("RLPD",mdl(sys.argv[2]))):
    for noise in (0.0,0.1):
        d,n=run(pol,noise); print(f"{name:13s} bruit {noise}: {d}/{n} épisodes doublés")
