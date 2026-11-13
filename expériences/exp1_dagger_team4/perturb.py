"""Script d'investigation du smoke test, tel qu'exécuté au commit a8431ab (API de ``lab/`` d'alors :
``lab.expert_adapter``, poids sans stats de normalisation). Gardé comme trace, pas maintenu."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from stk_actor.actors import MLPPolicy
from lab.envs import make_env, MAX_EPISODE_STEPS
from lab.expert_adapter import ExpertPolicy


m=MLPPolicy(); m.load_state_dict(torch.load(sys.argv[1])); m.eval()
env=make_env("lighthouse"); ex=ExpertPolicy(env,4)


def run(who,t0,k,seed):
    rng=np.random.default_rng(seed); obs,_=env.reset(seed=0)
    for s in range(1,MAX_EPISODE_STEPS+1):
        ae=ex.act()
        if t0<=s<t0+k: a=np.array([rng.uniform(0,1),rng.uniform(-1,1)],dtype=np.float32)
        else: a=ae if who=="expert" else m.act(obs)
        obs,r,te,tr,_=env.step(a)
        if te: return s
        if tr: return None
R={"expert":[],"student":[]}
for i in range(20):
    rng=np.random.default_rng(1000+i); t0=int(rng.integers(20,350)); k=int(rng.choice([10,20,40]))
    for who in R: R[who].append(run(who,t0,k,i))
for w,v in R.items():
    ok=[x for x in v if x]; print(w,"finis",len(ok),"/20, tour médian",np.median(ok) if ok else None)
both=[(e,s) for e,s in zip(R["expert"],R["student"]) if e and s]
print("pistes où les deux finissent:",len(both),"écart moyen élève-maître (pas):",np.mean([s-e for e,s in both]))
print("détail (maître,élève):",list(zip(R["expert"],R["student"])))
