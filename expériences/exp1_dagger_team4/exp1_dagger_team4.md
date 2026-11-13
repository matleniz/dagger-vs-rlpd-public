# Exp 1 : DAgger, maître team4, lighthouse

Date : 2026-11-11. Statut : **smoke test** (1 graine, 2 075 pas) : sert à faire sortir les problèmes de protocole, pas à conclure. Code du smoke test jamais commité ; l'API de `lab/` a changé depuis (voir README).

## Question
DAgger (élève MLP 256×256 ReLU, perte MSE) imite-t-il team4 sur lighthouse, et généralise-t-il à des pistes jamais vues ?

## Dispositif
- Env : 1 kart, 1 tour, action Box(2) = (accélération, direction) ; `lab/envs.py`. Maître : team4, actions discrètes ignorées.
- Entraînement : lighthouse. Test : hacienda, scotland (exclues des stats de normalisation).
- Observations : 92 continues normalisées `(x−m)/s` clip ±10 (`stk_actor/obs_stats.npz`, team4 sur 16 pistes hors test) + one-hot des 10 discrètes = 160 entrées.
- Données initiales : `data/team4_lighthouse_n0.0.npz` (1 épisode propre, 415 pas).
- Boucle (`lab/dagger.py`) : par itération, 1 épisode où chaque pas exécute l'action du maître avec proba β (0,5 puis ×0,5 par itération), sinon celle de l'élève ; le maître étiquette tous les états visités ; on agrège ; 200 pas de gradient (batch 256, Adam 1e-3, avec remise). Init : 1 000 pas de gradient sur la démo.
- Budget = pas d'env joués, démo comprise. Évaluation déterministe (`seed=0`) toutes les `--eval-every` pas, pistes train + test.

## Résultats (1 graine, `smoke_eval.csv`)
Référence maître (tronqué, pas) : lighthouse 415, hacienda 542, scotland 574.

| pas env | \|D\| | lighthouse | hacienda | scotland |
|---:|---:|---:|---:|---:|
| 1 245 | 1 245 | 416 (R 56,49) | 542 (R 72,54) | 575 (R 85,46) |
| 2 075 | 2 075 | 415 (R 56,58) | 541 (R 72,65) | 575 (R 85,49) |

- 1 245 pas = 3 tours (démo + 2 itérations), soit 1 400 pas de gradient de batch 256 (≈ 290 passages sur les données). Perte finale ≈ 2e-5.
- Perturbation (`perturb.py`, modèle à 2 075 pas) : 10–40 pas d'actions aléatoires à un instant tiré dans [20, 350], puis reprise par le maître ou l'élève, 20 perturbations identiques. Maître 19/20 finis, élève 19/20, tour médian 425 / 425, écart moyen élève−maître −0,1 pas sur 19 ; le seul échec est commun.
- Maître sur la démo : accélération moyenne 1,0 (toujours à fond), direction std 0,48 dans [−1, 1].

## Problème identifié
**La tâche est trop facile pour DAgger** : il égale le maître (au pas près) dès ~1 200 pas, sur piste vue, sur pistes de test, et après perturbation. Le budget de 100 000 pas ne discrimine rien de son côté.
Causes probables (non vérifiées) : accélération constante, donc une seule sortie (direction) ; le maître est sans doute une fonction déterministe de l'observation que le MLP reproduit exactement ; environnement déterministe, un kart seul, départ identique.
Conséquences : (1) comparer les performances finales de DAgger et RLPD ne sera pas informatif côté DAgger ; (2) la mesure utile est l'efficacité en échantillons (pas pour égaler le maître) ; (3) un seul épisode d'évaluation par point est insuffisant pour des tests statistiques, et le départ unique limite la diversité.
Limites : 1 graine, pas de barre d'erreur ; perturbation testée sur lighthouse seulement ; « départ n'importe où » non testé (pas de téléportation, seulement des perturbations par actions aléatoires).

## Reproductibilité
Depuis `dagger-vs-rlpd/`, avec `../.venv/bin/python` (Python 3.12.15, torch 2.15.0) :
```
python -m lab.obs_stats                                   # stats de normalisation (déterministes)
python -m lab.collect --expert 4 --track lighthouse --episodes 1 --noise 0.0 --out data/team4_lighthouse_n0.0.npz
python -m lab.dagger --seed 0 --steps 2000 --eval-every 1000 --out <dossier>
python -I expériences/exp1_dagger_team4/perturb.py <dossier>/model.pth
```
Graines : `--seed` fixe torch et le générateur des reset ; le simulateur est déterministe. Les chemins de `perturb.py` supposent d'être lancé depuis `dagger-vs-rlpd/`. Pas de dépôt git : version du code = état du dossier au 2026-11-11.

## Suite
Tenter RLPD dans le même dispositif (`lab/rlpd.py`) ; décider ensuite du budget et de la métrique (courbes d'efficacité en échantillons, évaluations plus fines au début).

## Addendum (2026-11-12) : résultats antérieurs aux correctifs
Cette exp. date d'avant la revue adverse : depuis, le maître est réinitialisé à chaque épisode (`ExpertPolicy.reset`), les étiquettes de récupération (frein) sont ignorées, la perte d'alpha de RLPD est corrigée. Les chiffres ci-dessus (smoke test, 1 graine) ont été revérifiés après correction à 1 245 et 1 660 pas (415 / 541-542 / 574) mais l'exp. n'a pas été refaite avec 5 graines. Suite : exp 2 (RLPD).
