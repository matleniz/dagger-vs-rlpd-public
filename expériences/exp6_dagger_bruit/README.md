# Exp 6 : DAgger avec bruit d'exécution, et démos bruitées réétiquetées

Date : 2026-11-12 15:30 → ~21:00, machine WSL, code du commit `f7da380`. 11 lots × 5 graines = 55 runs, tous finis.
Statut : tableaux bruts, sans interprétation.

## Question
Dans exp3, sur une piste, l'élève égale le maître dès la démo et chaque itération de DAgger rejoue la même
trajectoire (env déterministe). Que se passe-t-il quand l'élève est poussé hors de cette trajectoire ?

## Dispositif
- `lancer.sh` ; `lab/dagger.py --rollout-noise σ` : bruit gaussien d'écart-type σ ajouté à l'action **jouée** pendant les
  rollouts (puis écrêtée) ; l'étiquette reste l'action propre du maître. σ = 0,1 et 0,3, fixés avant tout résultat.
- 5 configurations de exp3, mêmes démos, stats, budget (20 000 pas) et graines : `t4_lighthouse`, `t3_lighthouse`,
  `t4_abyss`, `t4_snowtuxpeak`, `t1_lighthouse`. Le lot sans bruit de exp3 sert de référence.
- Lot de contrôle `t4_lighthouse_demos-bruit-propres` : démos = tour propre + les 10 épisodes bruités des démos de
  RLPD, **réétiquetés** par le maître (`lab.demos --clean-labels`, fichier `data/team4_lighthouse_n0.1_clean.npz`,
  mêmes observations que `_n0.1.npz`). À comparer à `t4_lighthouse_100k_demos-rlpd` de exp3, dont les démos bruitées
  contiennent l'action exécutée (51 % des accélérations < 1 contre 0 % après réétiquetage).

## Résultats
Colonnes (`expériences/tableau.py`) : « fin » = dernier point d'évaluation de chaque graine (tours finis / graines, tour médian [min-max] en pas) ; « ≥ 5 000 » = évaluations finies / évaluations à partir de 5 000 pas, toutes graines confondues (pas un échantillon indépendant) ; « par graine » = plus petite et plus grande part d'évaluations finies parmi les graines ; « centre » = distance moyenne au centre de la piste au dernier point (médiane des tours finis, pistes d'entraînement). Une évaluation = 1 épisode déterministe par piste.
Maîtres (1 épisode, Box(2)) : team4 lighthouse 415, snowtuxpeak 466, abyss 585, hacienda 542, scotland 574 ; team3 lighthouse 401 ; team1 lighthouse 412.

### Bruit d'exécution (20 000 pas)
| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dagger/t4_lighthouse` | 5 | 20335 | 5/5, 415 [415-415] | 65/65 | 0.74 | 5/5, 543 [541-543] | 65/65 | 1.00-1.00 | 5/5, 578 [576-582] | 65/65 | 1.00-1.00 |
| `dagger/t4_lighthouse_bruit0.1` | 5 | 20412 | 5/5, 415 [415-415] | 65/65 | 0.73 | 4/5, 542 [534-544] | 62/65 | 0.77-1.00 | 5/5, 578 [573-582] | 65/65 | 1.00-1.00 |
| `dagger/t4_lighthouse_bruit0.3` | 5 | 20280 | 5/5, 415 [415-415] | 65/65 | 0.74 | 4/5, 542 [540-545] | 63/65 | 0.85-1.00 | 5/5, 578 [576-619] | 65/65 | 1.00-1.00 |
| `dagger/t3_lighthouse` | 5 | 20113 | 5/5, 401 [400-404] | 65/65 | 1.86 | 0/5 | 1/65 | 0.00-0.08 | 2/5, 692 [614-770] | 3/65 | 0.00-0.15 |
| `dagger/t3_lighthouse_bruit0.1` | 5 | 20246 | 5/5, 401 [400-405] | 65/65 | 1.86 | 2/5, 528 [527-530] | 9/65 | 0.00-0.31 | 3/5, 614 [613-997] | 17/65 | 0.00-1.00 |
| `dagger/t3_lighthouse_bruit0.3` | 5 | 20123 | 5/5, 401 [400-401] | 65/65 | 1.84 | 2/5, 522 [522-523] | 3/65 | 0.00-0.08 | 5/5, 606 [574-620] | 12/65 | 0.15-0.23 |
| `dagger/t4_abyss` | 5 | 20595 | 5/5, 585 [585-585] | 57/57 | 0.46 | 0/5 | 0/57 | 0.00-0.00 | 0/5 | 6/57 | 0.00-0.55 |
| `dagger/t4_abyss_bruit0.1` | 5 | 20252 | 5/5, 585 [584-585] | 55/55 | 0.46 | 0/5 | 0/55 | 0.00-0.00 | 0/5 | 6/55 | 0.00-0.55 |
| `dagger/t4_abyss_bruit0.3` | 5 | 20520 | 5/5, 585 [584-614] | 55/55 | 0.46 | 0/5 | 0/55 | 0.00-0.00 | 0/5 | 0/55 | 0.00-0.00 |
| `dagger/t4_snowtuxpeak` | 5 | 20041 | 5/5, 466 [466-466] | 65/65 | 0.81 | 1/5, 545 [545-545] | 30/65 | 0.00-1.00 | 5/5, 580 [573-607] | 65/65 | 1.00-1.00 |
| `dagger/t4_snowtuxpeak_bruit0.1` | 5 | 20481 | 5/5, 466 [465-466] | 64/64 | 0.81 | 5/5, 544 [542-545] | 45/64 | 0.38-0.92 | 5/5, 575 [573-578] | 64/64 | 1.00-1.00 |
| `dagger/t4_snowtuxpeak_bruit0.3` | 5 | 20477 | 5/5, 466 [466-466] | 65/65 | 0.82 | 5/5, 542 [541-551] | 56/65 | 0.46-1.00 | 5/5, 581 [581-615] | 65/65 | 1.00-1.00 |
| `dagger/t1_lighthouse` | 5 | 20185 | 5/5, 412 [412-412] | 65/65 | 1.01 | 2/5, 575 [547-603] | 35/65 | 0.08-0.92 | 5/5, 611 [575-632] | 65/65 | 1.00-1.00 |
| `dagger/t1_lighthouse_bruit0.1` | 5 | 20333 | 5/5, 412 [411-413] | 65/65 | 1.01 | 1/5, 596 [596-596] | 36/65 | 0.54-0.62 | 5/5, 592 [581-627] | 65/65 | 1.00-1.00 |
| `dagger/t1_lighthouse_bruit0.3` | 5 | 20302 | 5/5, 412 [412-412] | 65/65 | 1.02 | 0/5 | 21/65 | 0.00-0.69 | 2/5, 600 [580-619] | 34/65 | 0.00-1.00 |

### Démos bruitées : étiquettes exécutées contre étiquettes propres
`@21000` : points jusqu'à 21 000 pas du lot à 100 000 pas.

| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dagger/t4_lighthouse_100k_demos-rlpd` | 5 | 20368 | 5/5, 415 [415-415] | 65/65 | 0.74 | 1/5, 549 [549-549] | 33/65 | 0.15-0.77 | 4/5, 713 [608-761] | 58/65 | 0.69-1.00 |
| `dagger/t4_lighthouse_demos-bruit-propres` | 5 | 20352 | 5/5, 415 [415-415] | 65/65 | 0.73 | 5/5, 544 [541-580] | 65/65 | 1.00-1.00 | 5/5, 580 [575-585] | 65/65 | 1.00-1.00 |
| `dagger/t4_lighthouse` | 5 | 20335 | 5/5, 415 [415-415] | 65/65 | 0.74 | 5/5, 543 [541-543] | 65/65 | 1.00-1.00 | 5/5, 578 [576-582] | 65/65 | 1.00-1.00 |

### Autres mesures
- Longueur des rollouts de `t4_lighthouse_bruit0.1`, graine 0 : 415 à 418 pas (414 à 415 sans bruit), tous finis.

## Problèmes
- Pas de run d'essai avant le lancement ; le premier lot a été vérifié à sa sortie.
- 2 runs à la fois tant que le RLPD local (exp5) tournait, 5 ensuite : temps d'horloge non comparables entre lots.
- Les points d'évaluation ne sont pas alignés entre graines (ils dépendent de la longueur des épisodes).
- Mêmes limites de mesure que exp3 (1 épisode par point ; « ≥ 5 000 » agrège le temps et les graines).

## Reproductibilité
```sh
python -m lab.demos --team 4 --tracks lighthouse --episodes 10 --noise 0.1 --clean-labels
LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp6_dagger_bruit/lancer.sh > runs/logs/dagger_bruit.log 2>&1 &
python -m lab.registry && python expériences/tableau.py dagger/t4_snowtuxpeak dagger/t4_snowtuxpeak_bruit0.1 dagger/t4_snowtuxpeak_bruit0.3
```
