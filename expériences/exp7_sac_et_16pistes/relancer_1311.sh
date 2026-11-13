#!/bin/sh
# Relance du 13/11 matin, après le redémarrage de la machine (01:28) et le correctif de obs_to_vector :
#   - SAC seul, déplacé sur Kaggle (le lot local s'est arrêté vers 27 000 pas ; il est gardé sous son nom _wsl)
#   - graine 2 de RLPD 16 pistes (Kaggle) et graine 0 de DAgger 16 pistes (local), plantées sur l'IndexError
#   LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp7_sac_et_16pistes/relancer_1311.sh > runs/logs/exp7_relance.log 2>&1 &
PY=${PY:-../.venv/bin/python}
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }
S16="abyss black_forest candela_city cocoa_temple cornfield_crossing gran_paradiso_island lighthouse mines minigolf olivermath ravenbridge_mansion sandtrack snowmountain snowtuxpeak volcano_island xr591"
push() { slug=$1; seeds=$2; name=$3; shift 3
  kaggle kernels status "matleniz/$slug" >/dev/null 2>&1 && { echo "déjà poussé : $slug"; return; }
  until $PY scripts/kaggle_push.py "$slug" rlpd "$seeds" -- --name "$name" "$@" 2>&1 | grep -q "successfully pushed"; do sleep 300; done
  echo "$(date '+%F %T') poussé : $slug (graines $seeds, $name)"; }
for s in 0-1 2-3 4; do push sac-t4-lh-s$(echo $s | tr -d -) $s t4_lighthouse_sac_kaggle --demo-ratio 0 --demos $(d 4 lighthouse); done
push rlpd-t4-16pistes-s2b 2 t4_16pistes_kaggle --tracks $S16 --demos $(d 4 "$S16")
$PY -m lab.launch dagger --seeds 0 --wait -- --name t4_16pistes_100k --threads 1 --steps 100000 --tracks $S16 --demos $(d 4 "$S16")
