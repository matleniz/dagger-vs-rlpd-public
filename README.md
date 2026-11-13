# DAgger vs RLPD on SuperTuxKart

A single kart on a track, one lap as fast as possible. The masters come from
[`pystk_project`](https://github.com/osigaud/pystk_project) (`src/agents`, imported, never modified):
sibling folder `../pystk_project`, or the variable `PYSTK_PROJECT_SRC=<path>/src`.
Python environment: `../.venv` (3.12, pystk2-gymnasium 0.10.0).

## Layout
- `stk_actor/`: **what is submitted to ML-Arena** (torch + numpy only).
  `actors.py`: `obs_to_vector` (obs -> raw vector of 160 values) and `MLPPolicy`, shared with
  training; `pystk_actor.py` loads `pystk_actor.pth` if it exists; `agent.py` must not be modified.
- `lab/`: the study, never submitted. Each module can be read on its own:

| module | role |
|---|---|
| `envs.py` | `make_env(track)`, `set_track`, `get_raw_obs`, and `play`: the shared episode loop |
| `experts.py` | `ExpertPolicy(env, team)`: a master plays in the student env (Box(2) action) |
| `demos.py` | demonstration collection (`data/*.npz`), loading, normalisation stats |
| `metrics.py` | episode metrics, CSV log |
| `evaluate.py` | evaluates a master or weights, one track per process |
| `run.py` | run folder: common arguments, config, evaluation schedule, saving |
| `launch.py`, `registry.py` | launch detached runs; status of all runs and concatenated logs |
| `dagger.py` | DAgger (`rollout` + `fit`) |
| `sac.py` | SAC, independent of the env |
| `rlpd.py` | RLPD = SAC + batch half demos, half online |

- `expériences/`: one folder per experiment (question, setup, results, issues). exp0 to exp2
  are smoke tests; their scripts are traces, not maintained code.
- `scripts/`: `check_submission.py`, `submit.py` (copied from the template).

## Commands (from this folder, `PY=../.venv/bin/python`)
```sh
# 1. master demonstrations (one file per track, raw vectors)
$PY -m lab.demos --team 4 --tracks lighthouse                           # data/team4_lighthouse_n0.0.npz
$PY -m lab.demos --team 4 --tracks lighthouse --episodes 10 --noise 0.1 # data/team4_lighthouse_n0.1.npz
# 2. training -> runs/<method>/<name>/seed<k>/{config.json, eval.csv, train.csv, model.pth}
$PY -m lab.dagger --seed 0 --demos data/team4_lighthouse_n0.0.npz
$PY -m lab.rlpd   --seed 0 --demos data/team4_lighthouse_n0.0.npz data/team4_lighthouse_n0.1.npz
# several detached seeds, then status of all runs (runs/index.csv, eval_all.csv, train_all.csv)
$PY -m lab.launch dagger --seeds 0-4 -- --name base --demos data/team4_lighthouse_n0.0.npz
$PY -m lab.registry
# 3. manual evaluation
$PY -m lab.evaluate --team 4 --tracks lighthouse hacienda
$PY -m lab.evaluate --weights runs/dagger/base/seed0/model.pth --tracks lighthouse
uv run scripts/check_submission.py     # before any submission
```
Arguments common to both methods (`lab/run.py`): `--name` (variant), `--team`, `--tracks` (one or
more training tracks, alternated), `--test-tracks`, `--demos`, `--stats`, `--steps`,
`--eval-every` / `--eval-early` (dense evaluations at the start). A run refuses to start if its folder exists.

To submit: copy a `model.pth` to `stk_actor/pystk_actor.pth`.

## Good to know
- **The env is deterministic**: a given master drives exactly the same lap every time (the seed has no
  effect). Variability comes from training (initialisation, sampling), not from the env.
- **Curve x-axis**: `env_steps` = env steps consumed, demonstrations included.
- **Leak-free normalisation**: the stats (mean, std of the 92 continuous columns, clip ±10) are
  computed on the `--stats` files (default: `--demos`) and saved in the model. Never put a test track
  in them. To compare two methods, give them the same `--stats`.
- **Student action** = (acceleration, steering). The masters also output brake/drift/nitro/rescue/fire:
  these are lost. Consequences: **team2 is unusable**; DAgger does not label states where the master brakes.
- **RLPD reward**: the env's reward, capped at 10.2 per step (`--reward-cap`): when crossing the
  line the env sometimes adds ~+88 (see `expériences/exp2_rlpd_team4`).
- Master lap times, all tracks: `expériences/exp0_maitres/compare_experts_all_tracks.md`
  (lighthouse: team3 401, team1 412, team4 415, team5 428 steps).
- Single kart: `karts_position` is empty; the master receives a distant dummy kart, the student zeros.
- On ML-Arena there will be 4 karts and opponents: the study does not test this.
