# Exp 4 : RLPD sur Kaggle (6 configurations × 5 graines, 100 000 pas)

Date : 2026-11-12 11:20 → 2026-11-13 ~03:00. Kernels Kaggle CPU (4 vCPU, Python 3.13, torch 2.11), code
`lab/rlpd.py` du commit `0771731` (inchangé pendant ces lots). `config.json` (ou `runs/logs/<slug>_commit.txt` sur Kaggle) porte le commit HEAD du lancement, qui change d'un lot à l'autre parce que `runs/` est commité en continu ; le code d'entraînement (`lab/`, `stk_actor/`) n'a changé qu'aux commits `0771731`, `f7da380` (bruit DAgger), `5bc2c66` (`--demo-ratio`, défaut inchangé) et `c71ad0a` (bornage des discrètes). Le champ `commit` de `config.json` est vide sur Kaggle. Statut : **tous les runs finis (30/30)**, tableaux bruts, sans interprétation.

## Question
Que donne RLPD avec les mêmes démos et les mêmes stats de normalisation que les lots DAgger de exp3 ?

## Dispositif
- `lancer.sh` (R0 à R3) puis `lancer2.sh` (R6, R5) ; `scripts/kaggle_push.py` embarque `lab/` et `stk_actor/`, les démos
  viennent du dataset privé `matleniz/dagger-vs-rlpd-demos` ; `scripts/kaggle_fetch.sh` rapatrie dans `runs/rlpd/`.
- Code `lab/rlpd.py`, arguments par défaut : Adam 3e-4, batch 256 (moitié démos, moitié en ligne), γ 0,99, τ 0,005,
  UTD 4, 500 pas avant la première mise à jour, α initial 0,1 appris (entropie cible −1), 2 critiques 256×256 avec
  LayerNorm, récompense plafonnée à 10,2 par pas, `--threads 2`. Pas de préentraînement hors ligne.
- Budget 100 000 pas d'env, démos comprises. Test : hacienda et scotland, jamais dans les démos ni dans les stats.

| id | lot | démos (transitions) | stats | lot DAgger aux mêmes démos |
|---|---|---|---|---|
| R0 | `t4_lighthouse_stats-demos_kaggle` | 1 tour propre + 10 épisodes bruités σ 0,1 (415 + 4 167) | démos | `t4_lighthouse_100k_demos-rlpd` |
| R1 | `t4_lighthouse_1tour_kaggle` | 1 tour propre (415) | démos | `t4_lighthouse`, `t4_lighthouse_100k` |
| R6 | `t4_lighthouse_1tour_stats-16pistes_kaggle` | 1 tour propre (415) | 16 pistes hors test | `t4_lighthouse_stats-16pistes` |
| R2 | `t3_lighthouse_kaggle` | team3, 1 tour (401) | démos | `t3_lighthouse` |
| R5 | `t4_snowtuxpeak_kaggle` | 1 tour snowtuxpeak (466) | démos | `t4_snowtuxpeak` |
| R3 | `t4_4pistes_kaggle` | 1 tour sur lighthouse, snowtuxpeak, cornfield_crossing, abyss | démos | `t4_4pistes` |

Le lot local R4 (team1) est dans `expériences/exp5_rlpd_local/`. Les configurations ont été choisies après lecture
des résultats de exp3, pistes de test comprises.

## Résultats
Colonnes (`expériences/tableau.py`) : « fin » = dernier point d'évaluation de chaque graine (tours finis / graines, tour médian [min-max] en pas) ; « ≥ 5 000 » = évaluations finies / évaluations à partir de 5 000 pas, toutes graines confondues (pas un échantillon indépendant) ; « par graine » = plus petite et plus grande part d'évaluations finies parmi les graines ; « centre » = distance moyenne au centre de la piste au dernier point (médiane des tours finis, pistes d'entraînement). Une évaluation = 1 épisode déterministe par piste.
Maîtres (1 épisode, Box(2)) : team4 lighthouse 415, snowtuxpeak 466, abyss 585, hacienda 542, scotland 574 ; team3 lighthouse 401 ; team1 lighthouse 412.

### RLPD à 100 000 pas
| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rlpd/t4_lighthouse_stats-demos_kaggle` | 5 | 100000 | 5/5, 382 [381-384] | 125/145 | 3.20 | 1/5, 820 [820-820] | 15/145 | 0.00-0.34 | 1/5, 589 [589-589] | 34/145 | 0.00-0.48 |
| `rlpd/t4_lighthouse_1tour_kaggle` | 5 | 100000 | 5/5, 381 [380-383] | 145/145 | 3.32 | 2/5, 782 [522-1041] | 29/145 | 0.07-0.34 | 1/5, 547 [547-547] | 44/145 | 0.03-0.66 |
| `rlpd/t4_lighthouse_1tour_stats-16pistes_kaggle` | 5 | 100000 | 5/5, 381 [380-383] | 145/145 | 3.29 | 0/5 | 40/145 | 0.14-0.41 | 3/5, 558 [548-614] | 128/145 | 0.72-1.00 |
| `rlpd/t3_lighthouse_kaggle` | 5 | 100000 | 5/5, 382 [380-385] | 145/145 | 3.40 | 0/5 | 13/145 | 0.00-0.21 | 1/5, 686 [686-686] | 23/145 | 0.00-0.45 |
| `rlpd/t4_snowtuxpeak_kaggle` | 5 | 100000 | 5/5, 446 [444-449] | 143/145 | 2.02 | 3/5, 530 [525-548] | 69/145 | 0.10-0.93 | 2/5, 608 [602-614] | 80/145 | 0.24-0.86 |
| `rlpd/t4_4pistes_kaggle` | 5 | 100000 | 20/20, 511 [388-587] | 533/580 | 2.07 | 5/5, 530 [524-539] | 98/145 | 0.34-0.79 | 1/5, 655 [655-655] | 89/145 | 0.28-0.79 |

### RLPD, points d'évaluation jusqu'à 20 000 pas (budget des lots DAgger)
| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rlpd/t4_lighthouse_stats-demos_kaggle` | 5 | 20000 | 5/5, 394 [391-409] | 45/65 | 2.52 | 0/5 | 8/65 | 0.00-0.38 | 2/5, 611 [606-616] | 14/65 | 0.00-0.62 |
| `rlpd/t4_lighthouse_1tour_kaggle` | 5 | 20000 | 5/5, 395 [385-398] | 65/65 | 2.83 | 2/5, 844 [536-1151] | 11/65 | 0.00-0.31 | 1/5, 572 [572-572] | 25/65 | 0.00-0.92 |
| `rlpd/t4_lighthouse_1tour_stats-16pistes_kaggle` | 5 | 20000 | 5/5, 389 [386-393] | 65/65 | 2.87 | 1/5, 529 [529-529] | 11/65 | 0.00-0.31 | 5/5, 569 [559-602] | 60/65 | 0.85-1.00 |
| `rlpd/t3_lighthouse_kaggle` | 5 | 20000 | 5/5, 394 [390-401] | 65/65 | 2.68 | 1/5, 553 [553-553] | 5/65 | 0.00-0.23 | 0/5 | 5/65 | 0.00-0.23 |
| `rlpd/t4_snowtuxpeak_kaggle` | 5 | 20000 | 5/5, 459 [448-510] | 64/65 | 1.90 | 3/5, 539 [532-555] | 26/65 | 0.08-1.00 | 4/5, 640 [567-822] | 37/65 | 0.23-0.92 |
| `rlpd/t4_4pistes_kaggle` | 5 | 20000 | 19/20, 576 [406-846] | 221/260 | 1.84 | 3/5, 553 [545-556] | 29/65 | 0.23-0.54 | 4/5, 744 [649-890] | 46/65 | 0.46-1.00 |

### Lots DAgger aux mêmes démos (exp3)
`@21000` : points jusqu'à 21 000 pas du lot à 100 000 pas.

| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dagger/t4_lighthouse_100k_demos-rlpd` | 5 | 100048 | 5/5, 415 [415-415] | 145/145 | 0.73 | 2/5, 554 [547-560] | 46/145 | 0.10-0.76 | 5/5, 672 [612-851] | 127/145 | 0.59-1.00 |
| `dagger/t4_lighthouse_100k_demos-rlpd` | 5 | 20368 | 5/5, 415 [415-415] | 65/65 | 0.74 | 1/5, 549 [549-549] | 33/65 | 0.15-0.77 | 4/5, 713 [608-761] | 58/65 | 0.69-1.00 |
| `dagger/t4_lighthouse_100k` | 5 | 100015 | 5/5, 415 [415-415] | 145/145 | 0.74 | 5/5, 543 [541-544] | 135/145 | 0.86-1.00 | 5/5, 578 [574-584] | 145/145 | 1.00-1.00 |
| `dagger/t4_lighthouse` | 5 | 20335 | 5/5, 415 [415-415] | 65/65 | 0.74 | 5/5, 543 [541-543] | 65/65 | 1.00-1.00 | 5/5, 578 [576-582] | 65/65 | 1.00-1.00 |
| `dagger/t4_lighthouse_stats-16pistes` | 5 | 20335 | 5/5, 415 [415-415] | 65/65 | 0.74 | 5/5, 540 [540-544] | 65/65 | 1.00-1.00 | 5/5, 576 [576-582] | 65/65 | 1.00-1.00 |
| `dagger/t3_lighthouse` | 5 | 20113 | 5/5, 401 [400-404] | 65/65 | 1.86 | 0/5 | 1/65 | 0.00-0.08 | 2/5, 692 [614-770] | 3/65 | 0.00-0.15 |
| `dagger/t4_snowtuxpeak` | 5 | 20041 | 5/5, 466 [466-466] | 65/65 | 0.81 | 1/5, 545 [545-545] | 30/65 | 0.00-1.00 | 5/5, 580 [573-607] | 65/65 | 1.00-1.00 |
| `dagger/t4_4pistes` | 5 | 20592 | 20/20, 526 [415-664] | 247/248 | 0.78 | 5/5, 544 [542-547] | 55/62 | 0.54-1.00 | 5/5, 587 [576-600] | 46/62 | 0.38-1.00 |

### Autres mesures
- α (température de SAC) en fin de run : entre 0,0035 et 0,0071 selon les lots (départ 0,1).
- Durée d'un run (`wall_s`) : 9 100 à 15 500 s dans les kernels à 1 ou 2 graines, 27 400 à 36 200 s dans les
  kernels à 3 graines (4 vCPU pour 3 processus à 2 threads). Limite Kaggle : 12 h par session.

## Problèmes
- 5 kernels CPU à la fois au plus : file d'attente locale (`lancer.sh` retente le push toutes les 5 min).
- Les kernels à 3 graines ont pris jusqu'à 10 h (36 153 s pour `t3_lighthouse_kaggle`), près de la limite de 12 h.
- Versions différentes du local (Python 3.13 / torch 2.11 contre 3.12 / 2.15) : pas de comparaison bit à bit ni de
  temps d'horloge entre Kaggle et local.
- `runs/rlpd/t4_lighthouse_stats-demos` (msi-matteo, arrêté vers 14 000 pas le 12/11) est à exclure : R0 le refait.
- Le point « fin » est 1 épisode à 1 instant : il varie d'une évaluation à l'autre (voir la colonne « par graine »).
- `lap_steps` n'existe que pour les tours finis ; `reward_cum` contient l'artefact de fin de tour (exp2).
- DAgger interroge le maître à chaque pas ; RLPD n'a que les démos et la récompense.
- Écarts au papier RLPD non réglés ici : α initial, UTD, absence de préentraînement (valeurs du papier à vérifier
  avant de les citer).

## Reproductibilité
```sh
setsid nohup sh expériences/exp4_rlpd_kaggle/lancer.sh  > runs/logs/kaggle_push.log  2>&1 &
setsid nohup sh expériences/exp4_rlpd_kaggle/lancer2.sh > runs/logs/kaggle_push2.log 2>&1 &
sh scripts/kaggle_fetch.sh && python -m lab.registry
python expériences/tableau.py rlpd/t4_lighthouse_1tour_kaggle rlpd/t4_lighthouse_1tour_kaggle@20000
```
Runs : `runs/rlpd/<lot>/seed<k>/` (config, `eval.csv`, `train.csv` ; poids non versionnés).
