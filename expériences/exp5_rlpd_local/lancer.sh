#!/bin/sh
# RLPD en local sur WSL (12/11, après la grille DAgger) : 2 configurations de plus que exp4, 5 graines,
# 100 000 pas, l'une après l'autre (7,8 Go : pas plus de 5 runs à la fois). Mêmes --demos/--stats que le lot DAgger.
#   LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp5_rlpd_local/lancer.sh > runs/logs/rlpd_local.log 2>&1 &
PY=${PY:-../.venv/bin/python}
run() { name=$1; shift; $PY -m lab.launch rlpd --seeds 0-4 --wait -- --name "$name" --threads 1 "$@"; }
run t1_lighthouse_wsl --team 1 --demos data/team1_lighthouse_n0.0.npz                    # R4 <-> dagger/t1_lighthouse
run t4_snowtuxpeak_wsl --tracks snowtuxpeak --demos data/team4_snowtuxpeak_n0.0.npz      # R5 <-> dagger/t4_snowtuxpeak
