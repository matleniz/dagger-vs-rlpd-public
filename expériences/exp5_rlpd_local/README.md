# Exp 5 : RLPD en local, maître team1 (5 graines, 100 000 pas)

Date : 2026-11-12 14:01 → 19:50, machine WSL (12 cœurs, 7,8 Go), code du commit `0771731`. Statut : **fini (5/5)**,
tableaux bruts, sans interprétation.

## Question
Même question que exp4, avec le maître team1 (direction saccadée, accélération modulée) : R4 `t1_lighthouse_wsl`.

## Dispositif
- `lancer.sh` ; `lab/rlpd.py`, mêmes arguments que exp4 sauf `--threads 1` et `LAB_EVAL_WORKERS=1`.
- Démos = stats = 1 tour de team1 sur lighthouse (412 transitions), comme le lot DAgger `t1_lighthouse`.
- Le second lot prévu (snowtuxpeak) a été annulé en local à 15:30 et fait sur Kaggle (exp4, R5).

## Résultats
Colonnes (`expériences/tableau.py`) : « fin » = dernier point d'évaluation de chaque graine (tours finis / graines, tour médian [min-max] en pas) ; « ≥ 5 000 » = évaluations finies / évaluations à partir de 5 000 pas, toutes graines confondues (pas un échantillon indépendant) ; « par graine » = plus petite et plus grande part d'évaluations finies parmi les graines ; « centre » = distance moyenne au centre de la piste au dernier point (médiane des tours finis, pistes d'entraînement). Une évaluation = 1 épisode déterministe par piste.
Maîtres (1 épisode, Box(2)) : team4 lighthouse 415, snowtuxpeak 466, abyss 585, hacienda 542, scotland 574 ; team3 lighthouse 401 ; team1 lighthouse 412.

| lot | graines | pas | entraînement, fin | entraînement ≥ 5 000 | centre | hacienda, fin | hacienda ≥ 5 000 | hacienda par graine | scotland, fin | scotland ≥ 5 000 | scotland par graine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rlpd/t1_lighthouse_wsl` | 5 | 100000 | 5/5, 381 [380-382] | 145/145 | 3.26 | 1/5, 590 [590-590] | 22/145 | 0.00-0.48 | 4/5, 620 [560-688] | 58/145 | 0.00-0.62 |
| `rlpd/t1_lighthouse_wsl` | 5 | 20000 | 5/5, 394 [387-402] | 65/65 | 2.94 | 0/5 | 8/65 | 0.00-0.62 | 1/5, 580 [580-580] | 28/65 | 0.00-1.00 |
| `dagger/t1_lighthouse` | 5 | 20185 | 5/5, 412 [412-412] | 65/65 | 1.01 | 2/5, 575 [547-603] | 35/65 | 0.08-0.92 | 5/5, 611 [575-632] | 65/65 | 1.00-1.00 |
| `dagger/t1_lighthouse_bruit0.1` | 5 | 20333 | 5/5, 412 [411-413] | 65/65 | 1.01 | 1/5, 596 [596-596] | 36/65 | 0.54-0.62 | 5/5, 592 [581-627] | 65/65 | 1.00-1.00 |
| `dagger/t1_lighthouse_bruit0.3` | 5 | 20302 | 5/5, 412 [412-412] | 65/65 | 1.02 | 0/5 | 21/65 | 0.00-0.69 | 2/5, 600 [580-619] | 34/65 | 0.00-1.00 |

Durée : 20 600 à 21 000 s par run (5 runs en parallèle, la machine partagée avec exp6 à partir de 15:30).
α en fin de run : 0,0038 à 0,0048.

## Problèmes
- Machine partagée avec les lots DAgger de exp6 : le temps d'horloge ne se compare pas aux runs Kaggle.
- Mêmes limites de mesure que exp4.

## Reproductibilité
```sh
LAB_EVAL_WORKERS=1 setsid nohup sh expériences/exp5_rlpd_local/lancer.sh > runs/logs/rlpd_local.log 2>&1 &
python -m lab.registry && python expériences/tableau.py rlpd/t1_lighthouse_wsl
```
