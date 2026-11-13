# Exp 2 : RLPD, démos team4, lighthouse (2 pilotes, anomalie de récompense)

Dates : 2026-11-11 et 12. Statut : **smoke tests** (2 pilotes d'une graine, 30 000 pas) : servent à faire sortir les problèmes (ici l'artefact de récompense), pas à conclure. L'API de `lab/` a changé depuis (voir README) ; les sorties console citées n'ont pas été gardées.

## Question
RLPD (SAC + 50 % de démos du maître dans chaque batch) apprend-il à finir lighthouse, aussi vite que le maître, et généralise-t-il à des pistes jamais vues ?

## Dispositif
- Même env, normalisation, évaluation et pistes de test que l'exp 1 (hacienda, scotland). Code : `lab/rlpd.py`.
- Acteur = `MLPPolicy` 256×256 ReLU + `log_std` global ; 2 critiques 256×256 avec LayerNorm (min) ; alpha automatique (départ 0,1, entropie cible −1) ; γ 0,99 ; τ 0,005 ; lr 3e-4 ; batch 256 (128 hors ligne + 128 en ligne) ; **UTD 4** ; 500 pas en ligne avant les mises à jour.
- Données hors ligne : `data/team4_lighthouse_n0.0.npz` (1 épisode propre) + `n0.1.npz` (10 épisodes, bruit 0,1) = 4 577 transitions, comptées dans le budget (RLPD part de 4 577 pas, DAgger de 415).
- Récompense = celle de l'env à chaque pas (kart seul) : `(Δ overall_distance)/10 − 0,1 + 10·[tour fini]`.
- Évaluation déterministe tous les 5 000 pas sur lighthouse, hacienda, scotland (1 épisode chacune, 3 processus en parallèle).

## Pilote 1 : récompense brute (`pilot1_sans_plafond_eval.csv`)
Tour en pas (maître : lighthouse 415, hacienda 542, scotland 574) ; « – » = non fini.

| pas | lighthouse | hacienda | scotland |
|---|---|---|---|
| 5 000 | – | – | – |
| 10 000 | 423 | – | 564 |
| 15 000 | 472 | – | – |
| 20 000 | 504 | – | 1132 |
| 25 000 | 434 | – | – |
| 30 000 | 420 | – | – |

Apprend lighthouse vite (dès ~5 400 pas en ligne) mais oscille, direction saccadée (jerk 0,24-0,30 contre 0,075), généralise mal et de façon instable. `reward_cum` anormal sur ~1/3 des tours (140 au lieu de 56).

## Problème 1 : récompense de fin de tour parasite
- **Symptôme** : épisodes finis avec `reward_cum` ≈ 140 et distance finale 1 760 (le double de la piste, 880). Sur 31 épisodes finis du pilote 1, 12 sont « doublés » (les 3 derniers de suite). Aucun autre écart : tous les tours finis sont à 880 ou 1 760.
- **Mécanisme (déduit de la lecture de `pystk2_gymnasium/envs.py:376-399` et des champs du kart, pas du code C++ de STK)** : `overall_distance` ≈ tours finis × longueur + `distance_down_track`. À l'image où la fin est détectée, soit `distance_down_track` est déjà repassée à ~0-1 (total ≈ 880, récompense finale ≈ 10), soit elle vaut encore 879,6-879,8 (total ≈ 1 760, récompense ≈ 98).
- **Test de fréquence** (15 épisodes par ligne, `test_doublement.py`) : maître 0/15 (bruit 0) et 0/15 (0,1) ; DAgger 0/15 et 0/15 ; RLPD 10/15 et 5/15. Propre à la politique de RLPD, pas à tout bruit.
- **Franchissement** (`franchissement.py`) : maître et DAgger passent la ligne au centre (±0,2 m) à 23 m/s, séquence identique à chaque épisode ; RLPD passe à **10-15 m du centre**, à 13-17 m/s, en braquant fort. Le doublement n'est pas une fonction de l'écart seul (cas normaux à 10,1-14,9 m, doublés à 10,9-14,7 m).
- **Triche par recul / aller-retour** : non observée (récompense = différence de distance, un aller-retour s'annule) ; non testée directement.
- **Fausse piste écartée** : un essai « récompense = coût de temps constant » (`--reward time`) venait d'une mauvaise lecture de la consigne (la récompense de RLPD est bien celle de l'env pas à pas, `reward_cum` n'est qu'une mesure). Deux pilotes lancés puis arrêtés et supprimés ; l'option reste, désactivée. Remarque notée : avec γ 0,99 un coût de temps seul aurait donné un signal presque nul (0,99^415 ≈ 0,015).
- **Résolution (décision utilisateur)** : plafonner la récompense par pas à `REWARD_CAP = 10,2` à l'entraînement (les démos culminent à 10,14). `reward_cum` journalisé reste brut ; la métrique principale devient `finished` + `lap_steps`.

## Pilote 2 : récompense plafonnée (`pilot2_plafond_eval.csv`)

| pas | lighthouse | hacienda | scotland |
|---|---|---|---|
| 5 000 | – | – | – |
| 10 000 | 423 | – | 564 |
| 15 000 | 403 | – | 623 |
| 20 000 | 390 | – | 633 |
| 25 000 | 400 | – | 624 |
| 30 000 | 388 | – | 626 |

- Doublement : **0/15 et 0/15** (bruit 0 / 0,1) ; 1 épisode sur 52 finis en entraînement.
- Franchissement redevenu centré (0,8-2,5 m) à 23 m/s, comme le maître. Lighthouse : 388 pas à 30 000, **6,5 % plus rapide que le maître** (distance finale ≈ 880, 25 pas « bloqués » comme lui, écart au centre moyen 2,7 m, jerk 0,19).
- Hacienda 0/6, scotland 5/6 mais lent (~625 contre 574) : généralise moins bien que DAgger (3/3 dès 1 245 pas).
- **Interprétation** (une graine, une intervention) : sans plafond RLPD allait vraisemblablement chercher le +88 en franchissant la ligne décentré ; avec le plafond il optimise le temps. Le maître est bridé par Box(2) (pas de nitro/dérapage), ce qui peut expliquer qu'il soit battu : non vérifié.

## Coût (mesuré, CPU 16 cœurs, pas de GPU compatible dans le venv)
Une mise à jour ≈ 22 ms seul, 27-34 ms avec 5 processus en parallèle ; XPU (venv à part, Arc B390) 17,5 ms seul, 20-22 ms à 5 processus : gain ×1,3-1,5, noyaux trop petits (batch 1024 ≈ batch 256). Évaluation de 3 pistes : 9,8 s en séquentiel, 6,1 s en parallèle (10 s au premier appel). Un pilote UTD 4 / 30 000 pas ≈ 45 min (durée exacte non relevée).

## Limites
Une graine par pilote, un épisode par point d'évaluation, pas d'incertitude ; RLPD simplifié (2 critiques au lieu de 10, UTD 4 au lieu de 20, `log_std` global) ; budget non aligné avec DAgger (démos 4 577 contre 415) ; mécanisme de l'artefact déduit, pas lu dans STK ; hacienda jamais finie.

## Reproductibilité
Depuis `dagger-vs-rlpd/`, `../.venv/bin/python` (Python 3.12.15, torch 2.15.0) :
```
python -m lab.rlpd --seed 0 --utd 4 --steps 30000 --out runs/rlpd/pilot_cap            # pilote 2
# pilote 1 : même commande avec REWARD_CAP = float("inf") dans lab/rlpd.py (code du commit bd5a73a)
python -I expériences/exp2_rlpd_team4/test_doublement.py <dagger/model.pth> <rlpd/model.pth>
python -I expériences/exp2_rlpd_team4/franchissement.py  <dagger/model.pth> <rlpd/model.pth>
python -I expériences/exp2_rlpd_team4/anomalie_distance.py   # lit runs/rlpd/pilot_utd4/model.pth
```
Les `.pth` ne sont pas versionnés. Entraînement CPU multi-thread : les graines fixent torch et numpy mais le résultat exact peut varier d'une exécution à l'autre (non vérifié). Les scripts doivent être lancés depuis `dagger-vs-rlpd/`.

## Suite
Planifier les 5 graines par méthode et les variantes sur la journée (voir `STATE.md`).
