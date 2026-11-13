#!/bin/sh
# Suite de lancer.sh (12/11 après-midi, après la revue adverse) : attend la fin de la première file, puis pousse
#   R6 : R1 avec les stats de normalisation des 16 pistes hors test (la mauvaise généralisation de RLPD vient-elle
#        de la normalisation ?)  <-> dagger/t4_lighthouse_stats-16pistes et rlpd/t4_lighthouse_1tour_kaggle
#   R5 : snowtuxpeak, déplacé du local vers Kaggle  <-> dagger/t4_snowtuxpeak
#   setsid nohup sh expériences/exp4_rlpd_kaggle/lancer2.sh > runs/logs/kaggle_push2.log 2>&1 &
PY=${PY:-../.venv/bin/python}
while pgrep -f 'exp4_rlpd_kaggle/lancer.s[h]' >/dev/null; do sleep 60; done
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }
S16="abyss black_forest candela_city cocoa_temple cornfield_crossing gran_paradiso_island lighthouse mines minigolf olivermath ravenbridge_mansion sandtrack snowmountain snowtuxpeak volcano_island xr591"
push() { slug=$1; seeds=$2; name=$3; shift 3
  kaggle kernels status "matleniz/$slug" >/dev/null 2>&1 && { echo "déjà poussé : $slug"; return; }
  until $PY scripts/kaggle_push.py "$slug" rlpd "$seeds" -- --name "$name" "$@" 2>&1 | grep -q "successfully pushed"; do sleep 300; done
  echo "$(date '+%F %T') poussé : $slug (graines $seeds, $name)"; }
for s in 3-4 0-2; do push rlpd-t4-lh-stats16-s$(echo $s | tr -d -) $s t4_lighthouse_1tour_stats-16pistes_kaggle --demos $(d 4 lighthouse) --stats $(d 4 "$S16"); done
for s in 3-4 0-2; do push rlpd-t4-snow-s$(echo $s | tr -d -) $s t4_snowtuxpeak_kaggle --tracks snowtuxpeak --demos $(d 4 snowtuxpeak); done
