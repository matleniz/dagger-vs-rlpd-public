#!/bin/sh
# RLPD sur Kaggle (12/11) : 4 configurations × 5 graines, 100 000 pas, mêmes --demos/--stats que le lot DAgger
# correspondant de exp3. Kernels détachés (4 vCPU chacun ; Kaggle en accepte 5 à la fois : chaque push est
# retenté toutes les 5 min jusqu'à ce qu'une place se libère). Depuis la racine du dépôt :
#   setsid nohup sh expériences/exp4_rlpd_kaggle/lancer.sh > runs/logs/kaggle_push.log 2>&1 &
# Récupération : kaggle kernels output matleniz/<slug> -p <dossier>, puis copier <dossier>/runs/rlpd/* dans runs/rlpd/.
PY=${PY:-../.venv/bin/python}
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }
P4="lighthouse snowtuxpeak cornfield_crossing abyss"
push() { slug=$1; seeds=$2; name=$3; shift 3
  kaggle kernels status "matleniz/$slug" >/dev/null 2>&1 && { echo "déjà poussé : $slug"; return; }
  until $PY scripts/kaggle_push.py "$slug" rlpd "$seeds" -- --name "$name" "$@" 2>&1 | grep -q "successfully pushed"; do sleep 300; done
  echo "$(date '+%F %T') poussé : $slug (graines $seeds, $name)"; }

# R0 <-> dagger/t4_lighthouse_100k_demos-rlpd
for s in 0-1 2-3 4; do push rlpd-t4-lh-demos-rlpd-s$(echo $s | tr -d -) $s t4_lighthouse_stats-demos_kaggle --demos $(d 4 lighthouse) data/team4_lighthouse_n0.1.npz; done
# R1 <-> dagger/t4_lighthouse et t4_lighthouse_100k (démo d'un seul tour)
for s in 0-1 2-3 4; do push rlpd-t4-lh-1tour-s$(echo $s | tr -d -) $s t4_lighthouse_1tour_kaggle --demos $(d 4 lighthouse); done
# R2 <-> dagger/t3_lighthouse
for s in 0-2 3-4; do push rlpd-t3-lh-s$(echo $s | tr -d -) $s t3_lighthouse_kaggle --team 3 --demos $(d 3 lighthouse); done
# R3 <-> dagger/t4_4pistes
for s in 0-2 3-4; do push rlpd-t4-4pistes-s$(echo $s | tr -d -) $s t4_4pistes_kaggle --tracks $P4 --demos $(d 4 "$P4"); done
