# Exp 7 : SAC seul (sans démos) et entraînement sur 16 pistes

Date : 2026-11-12 22:58 → 2026-11-13 13:17. Statut : **tous les runs finis** (DAgger 16 pistes 5/5, RLPD 16 pistes 5/5,
SAC seul 5/5 sur Kaggle ; le lot SAC local reste arrêté à ~27 000 pas). Tableaux bruts, sans interprétation.

## Questions (utilisateur, 12/11 soir)
1. Référence sans maître : que donne le même algorithme quand aucun batch ne contient de démo ?
2. Que donnent DAgger et RLPD entraînés sur toutes les pistes d'entraînement (16 pistes hors test de team4) ?

## Dispositif
- **SAC seul** : `lab/rlpd.py --demo-ratio 0` (commit `5bc2c66`) : batchs 100 % en ligne, budget compté à partir de 0 ;
  tout le reste identique à R1 de exp4 (`t4_lighthouse_1tour_kaggle`), y compris les stats de normalisation, tirées
  du tour de démo de team4 (le maître intervient donc encore par la normalisation). 100 000 pas, lighthouse.
  - `rlpd/t4_lighthouse_sac_wsl` : lot local, arrêté vers 27 000 pas par le redémarrage de la machine (13/11 01:28).
  - `rlpd/t4_lighthouse_sac_kaggle` : relancé sur Kaggle le 13/11 08:54 (kernels `sac-t4-lh-s{01,23,4}`), fini 13:17.
- **16 pistes** : abyss, black_forest, candela_city, cocoa_temple, cornfield_crossing, gran_paradiso_island, lighthouse,
  mines, minigolf, olivermath, ravenbridge_mansion, sandtrack, snowmountain, snowtuxpeak, volcano_island, xr591, en
  alternance d'un épisode à l'autre. Démos = stats = 1 tour propre de team4 par piste (9 513 transitions).
  100 000 pas pour les deux méthodes (RLPD : ~90 000 pas en ligne, soit ~5 700 par piste).
  - `dagger/t4_16pistes_100k` (local) ; `rlpd/t4_16pistes_kaggle` (kernels `rlpd-t4-16pistes-s{01,23,4}`, puis `-s2b`).
- Scripts : `lancer_local.sh`, `lancer_kaggle.sh`, `relancer_1311.sh`.

## Résultats
Colonnes (`expériences/tableau.py`) : « fin » = dernier point d'évaluation de chaque graine (tours finis / graines, tour médian [min-max] en pas) ; « ≥ 5 000 » = évaluations finies / évaluations à partir de 5 000 pas, toutes graines confondues (pas un échantillon indépendant) ; « par graine » = plus petite et plus grande part d'évaluations finies parmi les graines ; « centre » = distance moyenne au centre de la piste au dernier point (médiane des tours finis, pistes d'entraînement). Une évaluation = 1 épisode déterministe par piste.
Maîtres (1 épisode, Box(2)) : team4 lighthouse 415, snowtuxpeak 466, abyss 585, hacienda 542, scotland 574 ; team3 lighthouse 401 ; team1 lighthouse 412.

### 16 pistes (et lots à 4 et 8 pistes de exp3 et exp4)
`@N` : points d'évaluation jusqu'à N pas.

| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dagger/t4_16pistes_100k` | 5 | 100701 | 78/80, 585 [254-1287] | 1574/1600 | 0.77 | 5/5, 542 [542-546] | 99/100 | 0.95-1.00 | 5/5, 576 [575-584] | 100/100 | 1.00-1.00 |
| `dagger/t4_16pistes_100k` | 5 | 20645 | 80/80, 592 [253-1232] | 270/272 | 0.79 | 5/5, 543 [541-545] | 16/17 | 0.67-1.00 | 5/5, 579 [576-600] | 17/17 | 1.00-1.00 |
| `dagger/t4_8pistes` | 5 | 20177 | 38/40, 488 [391-672] | 476/488 | 0.85 | 3/5, 543 [542-544] | 52/61 | 0.73-0.92 | 5/5, 574 [574-577] | 57/61 | 0.77-1.00 |
| `dagger/t4_4pistes` | 5 | 20592 | 20/20, 526 [415-664] | 247/248 | 0.78 | 5/5, 544 [542-547] | 55/62 | 0.54-1.00 | 5/5, 587 [576-600] | 46/62 | 0.38-1.00 |
| `rlpd/t4_16pistes_kaggle` | 5 | 100000 | 70/80, 607 [238-1300] | 1181/1600 | 1.80 | 3/5, 544 [541-559] | 51/100 | 0.30-0.70 | 2/5, 589 [586-592] | 42/100 | 0.20-0.65 |
| `rlpd/t4_16pistes_kaggle` | 5 | 20000 | 57/80, 643 [256-1484] | 109/320 | 2.08 | 1/5, 561 [561-561] | 3/20 | 0.00-0.25 | 2/5, 606 [590-621] | 4/20 | 0.00-0.50 |
| `rlpd/t4_4pistes_kaggle` | 5 | 100000 | 20/20, 511 [388-587] | 533/580 | 2.07 | 5/5, 530 [524-539] | 98/145 | 0.34-0.79 | 1/5, 655 [655-655] | 89/145 | 0.28-0.79 |
| `rlpd/t4_4pistes_kaggle` | 5 | 20000 | 19/20, 576 [406-846] | 221/260 | 1.84 | 3/5, 553 [545-556] | 29/65 | 0.23-0.54 | 4/5, 744 [649-890] | 46/65 | 0.46-1.00 |

Dernier point, par piste (tours finis / graines, tour médian ; hacienda et scotland sont les pistes de test) :

| piste | maître team4 | DAgger 16 pistes | RLPD 16 pistes |
|---|---|---|---|
| abyss | 585 | 5/5, 585 | 5/5, 655 |
| black_forest | 1163 | 5/5, 1209 | 5/5, 1273 |
| candela_city | 488 | 5/5, 488 | 5/5, 530 |
| cocoa_temple | 590 | 5/5, 589 | 5/5, 630 |
| cornfield_crossing | 631 | 5/5, 648 | 3/5, 654 |
| gran_paradiso_island | 664 | 4/5, 656 | 3/5, 664 |
| hacienda | 542 | 5/5, 542 | 3/5, 544 |
| lighthouse | 415 | 5/5, 415 | 5/5, 411 |
| mines | 572 | 5/5, 572 | 5/5, 556 |
| minigolf | 385 | 5/5, 397 | 4/5, 404 |
| olivermath | 254 | 5/5, 255 | 4/5, 240 |
| ravenbridge_mansion | 616 | 5/5, 616 | 5/5, 681 |
| sandtrack | 558 | 5/5, 577 | 4/5, 580 |
| scotland | 574 | 5/5, 576 | 2/5, 589 |
| snowmountain | 541 | 5/5, 582 | 5/5, 593 |
| snowtuxpeak | 466 | 5/5, 466 | 3/5, 464 |
| volcano_island | 968 | 5/5, 952 | 5/5, 984 |
| xr591 | 617 | 4/5, 630 | 4/5, 646 |

### SAC seul et RLPD 1 tour (lighthouse, mêmes stats de normalisation)
| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rlpd/t4_lighthouse_sac_kaggle` | 5 | 100000 | 5/5, 383 [381-392] | 136/145 | 4.05 | 0/5 | 8/145 | 0.00-0.21 | 0/5 | 8/145 | 0.00-0.28 |
| `rlpd/t4_lighthouse_1tour_kaggle` | 5 | 100000 | 5/5, 381 [380-383] | 145/145 | 3.32 | 2/5, 782 [522-1041] | 29/145 | 0.07-0.34 | 1/5, 547 [547-547] | 44/145 | 0.03-0.66 |
| `rlpd/t4_lighthouse_sac_kaggle` | 5 | 20000 | 5/5, 400 [395-423] | 56/65 | 3.40 | 0/5 | 7/65 | 0.00-0.46 | 1/5, 670 [670-670] | 2/65 | 0.00-0.15 |
| `rlpd/t4_lighthouse_1tour_kaggle` | 5 | 20000 | 5/5, 395 [385-398] | 65/65 | 2.83 | 2/5, 844 [536-1151] | 11/65 | 0.00-0.31 | 1/5, 572 [572-572] | 25/65 | 0.00-0.92 |
| `rlpd/t4_lighthouse_sac_wsl` | 5 | 25000 | 5/5, 396 [389-416] | 63/70 | 3.49 | 0/5 | 3/70 | 0.00-0.07 | 1/5, 1028 [1028-1028] | 10/70 | 0.00-0.71 |

lighthouse, tours finis / graines et tour médian à pas fixés :

| pas | RLPD 1 tour (Kaggle) | SAC seul (Kaggle) | SAC seul (local, arrêté) |
|---|---|---|---|
| 1000 | 0/5 | 1/5, 969 | 1/5, 973 |
| 2000 | 2/5, 582 | 2/5, 854 | 2/5, 662 |
| 3000 | 2/5, 421 | 2/5, 633 | 2/5, 616 |
| 5000 | 5/5, 406 | 4/5, 448 | 3/5, 427 |
| 10000 | 5/5, 398 | 4/5, 418 | 4/5, 415 |
| 15000 | 5/5, 397 | 5/5, 405 | 5/5, 405 |
| 20000 | 5/5, 395 | 5/5, 400 | 5/5, 399 |
| 25000 | 5/5, 389 | 5/5, 397 | 5/5, 396 |
| 50000 | 5/5, 385 | 5/5, 386 | - |
| 75000 | 5/5, 382 | 5/5, 384 | - |
| 100000 | 5/5, 381 | 5/5, 383 | - |

Le budget de RLPD compte les 415 transitions de démo, celui du SAC seul part de 0 : à pas égal, RLPD a 415 pas en
ligne de moins.

## Problèmes
- **Plantage `IndexError` dans `obs_to_vector`** : l'env a sorti la valeur 6 pour la variable discrète d'indice 8,
  annoncée à 6 catégories. Vu deux fois, sur les lots à 16 pistes, à l'épisode qui suit lighthouse dans l'alternance
  (mines) : `dagger/t4_16pistes_100k` graine 0 (64 756 pas) et `rlpd/t4_16pistes_kaggle` graine 2 (29 921 pas).
  Corrigé au commit `c71ad0a` (valeur ramenée à la dernière catégorie, `stk_actor/actors.py`, partagé avec la
  soumission). Les deux runs plantés sont gardés sous `runs/dagger/t4_16pistes_100k_plante/` et
  `runs/rlpd/t4_16pistes_kaggle_plante/` ; les graines ont été relancées avec le correctif. Les autres graines
  de ces lots ont tourné sans le correctif et n'ont pas rencontré la valeur.
- Redémarrage de la machine (mise à jour) le 13/11 vers 01:28 : lot SAC local perdu après ~27 000 pas, relancé sur Kaggle.
  Le lot local et le lot Kaggle n'ont pas les mêmes versions de Python et de torch.
- RLPD 16 pistes : 70/80 pistes d'entraînement finies au dernier point (5 graines) ; budget par piste ~17 fois
  plus petit que dans les lots à 1 piste.
- Avec 18 pistes évaluées, les points « ≥ 5 000 » sont peu nombreux à 20 000 pas (évaluation tous les 5 000 pas
  au-delà de 10 000, et les démos comptent déjà 9 513 pas).

## Reproductibilité
```sh
LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp7_sac_et_16pistes/lancer_local.sh > runs/logs/exp7_local.log 2>&1 &
setsid nohup sh expériences/exp7_sac_et_16pistes/lancer_kaggle.sh > runs/logs/kaggle_push3.log 2>&1 &
LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp7_sac_et_16pistes/relancer_1311.sh > runs/logs/exp7_relance.log 2>&1 &
sh scripts/kaggle_fetch.sh && python -m lab.registry
python expériences/tableau.py rlpd/t4_lighthouse_sac_kaggle rlpd/t4_lighthouse_1tour_kaggle rlpd/t4_16pistes_kaggle dagger/t4_16pistes_100k
```
