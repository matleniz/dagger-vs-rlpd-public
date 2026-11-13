#!/bin/sh
# exp7, partie locale (12/11 soir, demandes de l'utilisateur) :
#   D16 : DAgger sur les 16 pistes hors test de team4, 100 000 pas (à 20 000, le lot 8 pistes était sous-entraîné)
#   S1  : SAC seul (--demo-ratio 0 : aucun batch ne contient de démo) sur lighthouse, 100 000 pas
#         <-> rlpd/t4_lighthouse_1tour_kaggle (mêmes stats de normalisation, tirées du tour de démo)
#   LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp7_sac_et_16pistes/lancer_local.sh > runs/logs/exp7_local.log 2>&1 &
PY=${PY:-../.venv/bin/python}
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }
S16="abyss black_forest candela_city cocoa_temple cornfield_crossing gran_paradiso_island lighthouse mines minigolf olivermath ravenbridge_mansion sandtrack snowmountain snowtuxpeak volcano_island xr591"
$PY -m lab.launch dagger --seeds 0-4 --wait -- --name t4_16pistes_100k --threads 1 --steps 100000 --tracks $S16 --demos $(d 4 "$S16")
$PY -m lab.launch rlpd --seeds 0-4 --wait -- --name t4_lighthouse_sac_wsl --threads 1 --demo-ratio 0 --demos $(d 4 lighthouse)
