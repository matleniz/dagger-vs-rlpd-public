# DAgger vs RLPD sur SuperTuxKart

Un kart seul sur une piste, un tour le plus vite possible. Les maîtres viennent de
[`pystk_project`](https://github.com/osigaud/pystk_project) (`src/agents`, importés, jamais modifiés) :
dossier voisin `../pystk_project`, ou variable `PYSTK_PROJECT_SRC=<chemin>/src`.
Environnement Python : `../.venv` (3.12, pystk2-gymnasium 0.10.0).

## Organisation
- `stk_actor/` : **ce qui part sur ML-Arena** (torch + numpy seulement).
  `actors.py` : `obs_to_vector` (obs -> vecteur brut de 160 valeurs) et `MLPPolicy`, partagés avec
  l'entraînement ; `pystk_actor.py` charge `pystk_actor.pth` s'il existe ; `agent.py` ne se modifie pas.
- `lab/` : l'étude, jamais soumise. Chaque module se lit seul :

| module | rôle |
|---|---|
| `envs.py` | `make_env(track)`, `set_track`, `get_raw_obs`, et `play` : la boucle d'épisode commune |
| `experts.py` | `ExpertPolicy(env, team)` : un maître joue dans l'env élève (action Box(2)) |
| `demos.py` | collecte des démonstrations (`data/*.npz`), chargement, stats de normalisation |
| `metrics.py` | métriques d'un épisode, journal CSV |
| `evaluate.py` | évalue un maître ou des poids, une piste par processus |
| `run.py` | dossier d'un run : arguments communs, config, calendrier d'évaluation, sauvegarde |
| `launch.py`, `registry.py` | lancer des runs détachés ; état de tous les runs et journaux concaténés |
| `dagger.py` | DAgger (`rollout` + `fit`) |
| `sac.py` | SAC, indépendant de l'env |
| `rlpd.py` | RLPD = SAC + batch moitié démos, moitié en ligne |

- `expériences/` : un dossier par expérience (question, dispositif, résultats, problèmes). exp0 à exp2
  sont des smoke tests ; leurs scripts sont des traces, pas du code maintenu.
- `ROADMAP.md` : idées d'expériences et plan ; `STATE.md` : état et décisions datées.
- `scripts/` : `check_submission.py`, `submit.py` (copiés du template).

## Commandes (depuis ce dossier, `PY=../.venv/bin/python`)
```sh
# 1. démonstrations du maître (un fichier par piste, vecteurs bruts)
$PY -m lab.demos --team 4 --tracks lighthouse                           # data/team4_lighthouse_n0.0.npz
$PY -m lab.demos --team 4 --tracks lighthouse --episodes 10 --noise 0.1 # data/team4_lighthouse_n0.1.npz
# 2. entraînement -> runs/<méthode>/<name>/seed<k>/{config.json, eval.csv, train.csv, model.pth}
$PY -m lab.dagger --seed 0 --demos data/team4_lighthouse_n0.0.npz
$PY -m lab.rlpd   --seed 0 --demos data/team4_lighthouse_n0.0.npz data/team4_lighthouse_n0.1.npz
# plusieurs graines détachées, puis état de tous les runs (runs/index.csv, eval_all.csv, train_all.csv)
$PY -m lab.launch dagger --seeds 0-4 -- --name base --demos data/team4_lighthouse_n0.0.npz
$PY -m lab.registry
# 3. évaluation à la main
$PY -m lab.evaluate --team 4 --tracks lighthouse hacienda
$PY -m lab.evaluate --weights runs/dagger/base/seed0/model.pth --tracks lighthouse
uv run scripts/check_submission.py     # avant toute soumission
```
Arguments communs aux deux méthodes (`lab/run.py`) : `--name` (variante), `--team`, `--tracks` (une ou
plusieurs pistes d'entraînement, en alternance), `--test-tracks`, `--demos`, `--stats`, `--steps`,
`--eval-every` / `--eval-early` (évaluations serrées au début). Un run refuse de démarrer si son dossier existe.

Pour soumettre : copier un `model.pth` en `stk_actor/pystk_actor.pth`.

## À savoir
- **L'env est déterministe** : un même maître refait exactement le même tour (graine sans effet). La
  variabilité vient de l'entraînement (initialisation, tirages), pas de l'env.
- **Abscisse des courbes** : `env_steps` = pas d'env consommés, démonstrations comprises.
- **Normalisation sans fuite** : les stats (moyenne, écart-type des 92 colonnes continues, clip ±10) sont
  calculées sur les fichiers `--stats` (défaut : `--demos`) et sauvées dans le modèle. Ne jamais y mettre
  une piste de test. Pour comparer deux méthodes, leur donner les mêmes `--stats`.
- **Action de l'élève** = (accélération, direction). Les maîtres produisent aussi brake/drift/nitro/rescue/fire :
  perdus. Conséquences : **team2 est inutilisable** ; DAgger n'étiquette pas les états où le maître freine.
- **Récompense RLPD** : celle de l'env, plafonnée à 10,2 par pas (`--reward-cap`) : au franchissement de la
  ligne l'env ajoute parfois ~+88 (voir `expériences/exp2_rlpd_team4`).
- Temps de tour des maîtres, toutes pistes : `expériences/exp0_maitres/compare_experts_all_tracks.md`
  (lighthouse : team3 401, team1 412, team4 415, team5 428 pas).
- Kart seul : `karts_position` est vide ; le maître reçoit un kart factice lointain, l'élève des zéros.
- Sur ML-Arena il y aura 4 karts et des adversaires : l'étude ne le teste pas.
