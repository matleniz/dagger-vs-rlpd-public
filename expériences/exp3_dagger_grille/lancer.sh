#!/bin/sh
# Grille DAgger du 12/11 (voir ROADMAP.md) : 5 graines par configuration, un lot après l'autre.
# Depuis la racine du dépôt :  setsid nohup sh expériences/exp3_dagger_grille/lancer.sh > runs/logs/grille_dagger.log 2>&1 &
# Découpages fixés avant tout résultat. Stats de normalisation = démos du run (choix A), sauf le lot D2.
PY=${PY:-../.venv/bin/python}
T4="abyss black_forest candela_city cocoa_temple cornfield_crossing gran_paradiso_island hacienda lighthouse mines minigolf olivermath ravenbridge_mansion sandtrack scotland snowmountain snowtuxpeak volcano_island xr591"
d() { for t in $2; do printf "data/team%s_%s_n0.0.npz " "$1" "$t"; done; }   # fichiers de démos : équipe, pistes
run() { name=$1; shift; $PY -m lab.launch dagger --seeds 0-4 --wait -- --name "$name" --threads 1 "$@"; }

$PY -m lab.demos --team 4 --tracks $T4
$PY -m lab.demos --team 3 --tracks lighthouse
$PY -m lab.demos --team 1 --tracks lighthouse

# D1 : maître
run t4_lighthouse --steps 20000 --demos $(d 4 lighthouse)
run t3_lighthouse --steps 20000 --team 3 --demos $(d 3 lighthouse)
run t1_lighthouse --steps 20000 --team 1 --demos $(d 1 lighthouse)
# D2 : stats sur les 16 pistes hors test (ancien protocole) au lieu des démos du run
S16=$(echo $T4 | tr ' ' '\n' | grep -v -e hacienda -e scotland | tr '\n' ' ')
run t4_lighthouse_stats-16pistes --steps 20000 --demos $(d 4 lighthouse) --stats $(d 4 "$S16")
# D3 : piste d'entraînement
for t in snowtuxpeak cornfield_crossing abyss; do run t4_$t --steps 20000 --tracks $t --demos $(d 4 $t); done
# D4 : nombre de pistes d'entraînement, à pas totaux égaux
P2="lighthouse snowtuxpeak"; P4="$P2 cornfield_crossing abyss"; P8="$P4 candela_city cocoa_temple minigolf xr591"
run t4_2pistes --steps 20000 --tracks $P2 --demos $(d 4 "$P2")
run t4_4pistes --steps 20000 --tracks $P4 --demos $(d 4 "$P4")
run t4_8pistes --steps 20000 --tracks $P8 --demos $(d 4 "$P8")
# D5 : second découpage de test
run t4_lighthouse_test-mines-sandtrack --steps 20000 --test-tracks mines sandtrack --demos $(d 4 lighthouse)
# D6 : 100 000 pas, pour la comparaison avec RLPD (démo d'un tour, puis mêmes démos que RLPD)
run t4_lighthouse_100k --steps 100000 --demos $(d 4 lighthouse)
run t4_lighthouse_100k_demos-rlpd --steps 100000 --demos $(d 4 lighthouse) data/team4_lighthouse_n0.1.npz
