#!/bin/sh
# exp7, partie Kaggle : RLPD sur les 16 pistes hors test de team4, 100 000 pas (~6 000 pas par piste)
#   <-> dagger/t4_16pistes_100k (mêmes --demos, donc mêmes stats)
#   setsid nohup sh expériences/exp7_sac_et_16pistes/lancer_kaggle.sh > runs/logs/kaggle_push3.log 2>&1 &
PY=${PY:-../.venv/bin/python}
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }
S16="abyss black_forest candela_city cocoa_temple cornfield_crossing gran_paradiso_island lighthouse mines minigolf olivermath ravenbridge_mansion sandtrack snowmountain snowtuxpeak volcano_island xr591"
push() { slug=$1; seeds=$2; name=$3; shift 3
  kaggle kernels status "matleniz/$slug" >/dev/null 2>&1 && { echo "déjà poussé : $slug"; return; }
  until $PY scripts/kaggle_push.py "$slug" rlpd "$seeds" -- --name "$name" "$@" 2>&1 | grep -q "successfully pushed"; do sleep 300; done
  echo "$(date '+%F %T') poussé : $slug (graines $seeds, $name)"; }
for s in 0-1 2-3 4; do push rlpd-t4-16pistes-s$(echo $s | tr -d -) $s t4_16pistes_kaggle --tracks $S16 --demos $(d 4 "$S16"); done
