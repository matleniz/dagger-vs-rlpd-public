# Exp 3 : grille DAgger (maître, normalisation, piste, nombre de pistes, budget)

Date : 2026-11-12, machine WSL (12 cœurs, 7,8 Go), commit `0771731`. 13 configurations × 5 graines = 65 runs,
tous finis (11:06 → 14:00). Statut : **résultats bruts**, analyse statistique à faire le 13/11.

## Question
Les smoke tests disaient « la tâche est trop facile pour DAgger » (team4 sur lighthouse). Est-ce encore vrai
quand on change le maître, la piste d'entraînement, le nombre de pistes, la source des stats de normalisation
et les données de départ ?

## Dispositif
- `lancer.sh` (découpages fixés avant tout résultat). Code : `lab/dagger.py`, arguments par défaut.
- Élève `MLPPolicy` 256×256, action Box(2). Test : hacienda et scotland (sauf lot D5 : mines, sandtrack),
  jamais dans les démos ni dans les stats.
- Budget 20 000 pas d'env (démos comprises), 100 000 pour le lot D6. Évaluation déterministe, 1 épisode par
  piste, tous les 500 pas jusqu'à 10 000 puis tous les 5 000.
- Stats de normalisation = démos du run, sauf `stats-16pistes` (démos team4 des 16 pistes hors test).

| lot | configurations | ce qui varie |
|---|---|---|
| D1 | `t4_lighthouse`, `t3_lighthouse`, `t1_lighthouse` | maître |
| D2 | `t4_lighthouse_stats-16pistes` | source des stats |
| D3 | `t4_snowtuxpeak`, `t4_cornfield_crossing`, `t4_abyss` | piste d'entraînement |
| D4 | `t4_2pistes`, `t4_4pistes`, `t4_8pistes` | nombre de pistes, à pas totaux égaux |
| D5 | `t4_lighthouse_test-mines-sandtrack` | pistes de test |
| D6 | `t4_lighthouse_100k`, `t4_lighthouse_100k_demos-rlpd` | budget ; démos de RLPD (propre + 10 ép. bruités) |

## Résultats
`python -m lab.registry && python expériences/exp3_dagger_grille/resume.py` (sortie complète : `resume.txt`).
« fin » = dernier point d'évaluation, sur 5 graines : tours finis, tour médian en pas [min-max].
« ≥ 5 000 » = part des évaluations finies à partir de 5 000 pas, toutes graines confondues.

Pistes d'entraînement : finies 5/5 partout (sauf xr591 3/5 dans `t4_8pistes`), au temps du maître.

| configuration | hacienda, fin | hacienda ≥ 5 000 | scotland, fin | scotland ≥ 5 000 |
|---|---|---|---|---|
| `t4_lighthouse` | 5/5, 543 [541-543] | 65/65 | 5/5, 578 [576-582] | 65/65 |
| `t3_lighthouse` | 0/5 | 1/65 | 2/5, 692 [614-770] | 3/65 |
| `t1_lighthouse` | 2/5, 575 [547-603] | 35/65 | 5/5, 611 [575-632] | 65/65 |
| `t4_lighthouse_stats-16pistes` | 5/5, 540 [540-544] | 65/65 | 5/5, 576 [576-582] | 65/65 |
| `t4_snowtuxpeak` | 1/5, 545 | 30/65 | 5/5, 580 [573-607] | 65/65 |
| `t4_cornfield_crossing` | 4/5, 536 [530-539] | 43/55 | 4/5, 600 [581-748] | 45/55 |
| `t4_abyss` | 0/5 | 0/57 | 0/5 | 6/57 |
| `t4_2pistes` | 4/5, 542 [541-544] | 56/65 | 5/5, 576 [576-582] | 65/65 |
| `t4_4pistes` | 5/5, 544 [542-547] | 55/62 | 5/5, 587 [576-600] | 46/62 |
| `t4_8pistes` | 3/5, 543 [542-544] | 52/61 | 5/5, 574 [574-577] | 57/61 |
| `t4_lighthouse_100k` | 5/5, 543 [541-544] | 135/145 | 5/5, 578 [574-584] | 145/145 |
| `t4_lighthouse_100k_demos-rlpd` | 2/5, 554 [547-560] | 46/145 | 5/5, 672 [612-851] | 127/145 |

`t4_lighthouse_test-mines-sandtrack` : mines 5/5, 544 [543-560] ; sandtrack 5/5, 588 [588-589] ; 65/65 et 65/65.
Maître team4 (référence) : lighthouse 415, hacienda 542, scotland 574.

## Clonage seul contre fin de DAgger (ajout après revue adverse, 12/11)
Le premier point d'évaluation de chaque run (env_steps = taille des démos : 1 000 pas de gradient sur la démo,
avant toute itération) est un clonage simple (BC). Tours finis sur 5 graines, BC → fin de run :

| configuration | hacienda | scotland |
|---|---|---|
| `t4_lighthouse` | 5 → 5 | 5 → 5 (628 → 578 pas) |
| `t3_lighthouse` | 0 → 0 | 0 → 2 |
| `t1_lighthouse` | 5 → 2 | 5 → 5 |
| `t4_lighthouse_stats-16pistes` | 5 → 5 | 5 → 5 |
| `t4_snowtuxpeak` | 5 → 1 | 4 → 5 |
| `t4_cornfield_crossing` | 4 → 4 | 3 → 4 |
| `t4_abyss` | 0 → 0 | 1 → 0 |
| `t4_2pistes` / `4pistes` / `8pistes` | 5 → 4 / 5 → 5 / 5 → 3 | 5 → 5 / 3 → 5 / 5 → 5 |
| `t4_lighthouse_100k_demos-rlpd` | 5 → 2 | 5 → 5 (614 → 672 pas) |

## Lecture (à confirmer par l'analyse du 13)
- **Sur une piste, DAgger est ici un clonage simple.** L'env est déterministe et l'élève égale le maître dès la
  démo : dans `t4_lighthouse` chaque itération rejoue le même tour (415 pas, fini) et le jeu de données final est
  ~48 copies de la même trajectoire. La boucle n'a aucun état d'erreur à corriger.
- Ce que le tableau des résultats mesure est donc surtout la **généralisation du clonage**, qui dépend des données
  (maître, piste) : team3 et abyss échouent déjà au premier point, la boucle n'y change rien.
- La boucle **ne rend pas la généralisation meilleure de façon fiable** : sur hacienda elle la dégrade dans
  plusieurs configurations (t1 5 → 2, snowtuxpeak 5 → 1, 8 pistes 5 → 3, démos bruitées 5 → 2) ; sur scotland elle
  aide un peu (4 pistes 3 → 5, tour de `t4_lighthouse` 628 → 578). Cause non établie (hypothèse : sur-ajustement,
  200 pas de gradient par itération à lr constant sur des données presque dupliquées).
- Sur la piste d'entraînement, DAgger égale le maître, sauf dans `t4_8pistes` (xr591 fini 3/5, cornfield et xr591
  plus lents que le maître) : ce lot est sous-entraîné (2 500 pas par piste, perte 100 fois plus haute).
- Stats sur 16 pistes ou sur la seule démo : même résultat à 5 graines (l'écart vu sur 1 graine ne se confirme pas).
- `t4_lighthouse_100k_demos-rlpd` : les démos bruitées contiennent l'action **exécutée** (maître + bruit, écrêtée :
  51 % des accélérations < 1 contre 0 % dans la démo propre), pas l'étiquette propre du maître. Pour RLPD ce sont
  de vraies transitions ; pour DAgger ce sont des étiquettes bruitées et biaisées, et les stats de normalisation
  changent aussi. Le clonage de départ reste bon (5/5, 5/5) : c'est pendant la boucle que hacienda se dégrade.
- 100 000 pas n'apportent rien de plus que 20 000, parce que rien ne change après la démo.
- Le point « fin » (1 épisode, 1 instant) est instable : par graine, la part d'évaluations finies sur hacienda
  après 5 000 pas va de 0,00 à 1,00 dans `t4_snowtuxpeak`. Un 2/5 contre 5/5 n'est pas significatif (Fisher
  p = 0,17) ; 0/5 contre 5/5 l'est (p = 0,008). Dans le rapport : part finie par graine, pas le seul point final.

## Problèmes
- Crash mémoire du premier lancement (msi-matteo, 10 runs en parallèle). Résolu sur WSL par
  `LAB_EVAL_WORKERS=1` : un run = ~1,0 Go au lieu de 1,2 à 1,8 ; 5 graines = ~4 Go.
- 1 épisode d'évaluation par point : la colonne « ≥ 5 000 » agrège le temps et les graines, ce n'est pas un
  échantillon indépendant. Les tours sont souvent identiques au pas près entre graines : pas de test de Welch.
- DAgger interroge le maître à chaque pas (20 000 à 100 000 étiquettes) ; RLPD n'a que ses démos (415 ou 4 582
  transitions) et la récompense : « budget égal » en pas d'env n'est pas « information égale ».
- Les itérations (donc les pas de gradient) dépendent de la longueur du tour : 48 itérations sur lighthouse,
  34 sur abyss, 30 sur 8 pistes, à 20 000 pas.
- Les configurations RLPD lancées ensuite (exp4, exp5) ont été choisies après lecture de ce tableau, pistes de
  test comprises : c'est un choix de comparaison, pas un réglage, mais il est à signaler.

## Reproductibilité
```sh
LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp3_dagger_grille/lancer.sh > runs/logs/grille_dagger.log 2>&1 &
python -m lab.registry && python expériences/exp3_dagger_grille/resume.py
```
Runs : `runs/dagger/<configuration>/seed<k>/` (config, `eval.csv`, `train.csv` ; poids non versionnés).
