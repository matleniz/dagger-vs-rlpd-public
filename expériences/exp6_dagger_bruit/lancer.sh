#!/bin/sh
# DAgger avec bruit sur l'action exécutée (12/11, après la revue adverse : sans bruit, DAgger = clonage ici).
# Étiquettes = action propre du maître. Niveaux de bruit fixés d'avance (0,1 = celui des démos de RLPD ; 0,3).
# Mêmes démos, stats, budget (20 000 pas) et graines que le lot sans bruit de exp3.
#   LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp6_dagger_bruit/lancer.sh > runs/logs/dagger_bruit.log 2>&1 &
PY=${PY:-../.venv/bin/python}
d() { printf "data/team%s_%s_n0.0.npz" "$1" "$2"; }
# 7,8 Go : 2 runs à la fois tant que le lot RLPD local tourne, 5 sinon
run() { name=$1; shift
  if pgrep -f 'm lab.rlpd --seed' >/dev/null; then chunks="0-1 2-3 4"; else chunks="0-4"; fi
  for s in $chunks; do $PY -m lab.launch dagger --seeds $s --wait -- --name "$name" --threads 1 --steps 20000 "$@"; done; }

# B0 : démos de RLPD réétiquetées par le maître (sépare le bruit d'étiquette du reste) <-> t4_lighthouse_100k_demos-rlpd
run t4_lighthouse_demos-bruit-propres --demos $(d 4 lighthouse) data/team4_lighthouse_n0.1_clean.npz
for noise in 0.1 0.3; do
  run t4_lighthouse_bruit$noise  --rollout-noise $noise --demos $(d 4 lighthouse)
  run t3_lighthouse_bruit$noise  --rollout-noise $noise --team 3 --demos $(d 3 lighthouse)
  run t4_abyss_bruit$noise       --rollout-noise $noise --tracks abyss --demos $(d 4 abyss)
  run t4_snowtuxpeak_bruit$noise --rollout-noise $noise --tracks snowtuxpeak --demos $(d 4 snowtuxpeak)
  run t1_lighthouse_bruit$noise  --rollout-noise $noise --team 1 --demos $(d 1 lighthouse)
done
