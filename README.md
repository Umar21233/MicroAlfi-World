# Microalfi World: RL on a Cellular-Automaton Bioreactor

A reinforcement-learning project simulating algae cultivation as a 10x10
cellular-automaton (CA) grid world. A single RL agent sits **outside** the
grid and controls global bioreactor conditions (light, CO2, nutrients, pH,
toxin filtering, radiation shielding, humidity, growth pulses); colony
behaviour (spreading, mutation, toxin build-up, collapse) **emerges** from
local cell-to-cell rules the agent never touches directly.

This repository began as an internship project and continues into a
final-year project. The internship's experiments are **done** and are
preserved as described below. **Nothing in `envs/alphae_world_env_v2.py`
may change behaviourally**: every constant, CA rule, reward formula, and
reset/seed sequence must keep producing exactly the numbers already on
disk. See `docs/AUDIT.md` for the original audit of this codebase and
`docs/VERIFICATION.md` for the full rebuild and verification of its
results.

## Which results are trustworthy

Use **`results_verified/`**, not `results/`. `results/` is the
internship's original local output folder; its run-name convention
(`{algo}_{world}_{reward_mode}_{seed}`) has no drift-condition component,
so a with-drifts and a no-drifts run of the same algo/world/reward_mode/
seed silently overwrote each other's files (`docs/AUDIT.md` finding #12).
See `results/README_LEGACY.txt`. `results/` is kept in place rather than
archived, because `scripts/recompute_plot_numbers.py` still reads from it
as part of the audit trail.

`results_verified/` was rebuilt from scratch directly from the raw,
read-only Colab training downloads (`Colab training results/`, SHA-256
tracked in `data_manifest.json`), using a collision-free five-part run
id:

```
{algo}_{world}_{reward_mode}_{drift}_{seed}
```

where `drift` is `drifts` (`no_drifts=False`) or `nodrifts`
(`no_drifts=True`). See `docs/VERIFICATION.md` for the full rebuild
process, the verified result tables, and the old-versus-verified
plot-number comparison.

### Zip-stem to canonical-run-id mapping

| Raw Colab zip stem(s) | algo | world | reward_mode | drift |
|---|---|---|---|---|
| `a2c_seed{42,43,44}` | a2c | hard | multi_objective | drifts |
| `dqn_seed{42,43,44}` | dqn | hard | multi_objective | drifts |
| `ppo_seed{42,43,44}` | ppo | hard | multi_objective | drifts |
| `dqn_biomass_seed{42,43,44}` | dqn | hard | biomass_only | drifts |
| `dqn_efficiency_seed{42,43,44}` | dqn | hard | efficiency | drifts |
| `dqn_sustainability_seed{42,43,44}` | dqn | hard | sustainability | drifts |
| `dqn_turbidity_seed{42,43,44}` | dqn | hard | turbidity | drifts |
| `dqn_turbdelta_seed{42,43,44}` | dqn | hard | turbidity_delta | drifts |
| `dqn_easy_drifts_seed{42,43,44}` | dqn | easy | multi_objective | drifts |
| `dqn_easy_nodrifts_seed{42,43,44}` | dqn | easy | multi_objective | nodrifts |
| `dqn_medium_drifts_seed{42,43,44}` | dqn | medium | multi_objective | drifts |
| `dqn_medium_nodrifts_seed{42,43,44}` | dqn | medium | multi_objective | nodrifts |
| `dqn_hard_nodrifts_seed{42,43,44}` | dqn | hard | multi_objective | nodrifts |

Phase 3's "hard, with drifts" condition is the Phase 1 DQN run (referenced
by listing from `results_verified/phase3/_REFERENCED_FROM_PHASE1.txt`, not
duplicated). 9 (Phase 1) + 15 (Phase 2) + 15 (Phase 3) = **39 unique
models**, each under `results_verified/phase{1,2,3}/<run_id>/` with
`config.txt`, `model.zip`, `learning_curve.csv`, `training_time.txt`,
`eval.csv`, `eval_summary.csv`, `coverage_trajectory.csv`. Full provenance
(which file came from which zip, original versus regenerated) is in
`results_verified/PROVENANCE.csv`.

## Folder map

```
envs/
  alphae_world_env_v2.py   Frozen. The CA grid-world Gymnasium environment
                            (10x10 grid, 13-dim obs, 11 actions, 6 reward
                            modes). Every result in results_verified/
                            depends on this file never changing.
  alphae_world_env.py       Superseded v1 env. Kept for history; not used
                            by any current training/eval script.
  renderer.py               Pygame renderer for AlphaeWorldEnvV2 (--render).

agents/
  random_agent_v2.py        Baseline 1 (floor): uniform-random actions.
                            Defines WORLD_PRESETS (easy/medium/hard).
  fixed_setpoint_agent_v2.py Baseline 2 (ceiling): hand-coded priority
                            controller, no learning.
  complexity_check.py       Validates that easy/medium/hard form a real
                            difficulty ladder (random versus fixed
                            setpoint).
  random_agent.py, fixed_setpoint_agent.py
                            Equivalents for the superseded v1 env.

training/
  train_dqn.py, train_ppo.py, train_a2c.py
                            Train one Stable-Baselines3 model on
                            AlphaeWorldEnvV2 for a given world, reward
                            mode and seed. Saves model + config.txt under
                            trained_models/, plus a learning-curve CSV and
                            a training-time file under results/.
  hyperparameters.py        The tuned hyperparameters confirmed from the
                            Colab-trained models; used when a training
                            script is run with --tuned.
  evaluate.py               Loads a saved model and evaluates it
                            deterministically for 20 episodes on seeds
                            100-119 (disjoint from training seeds). Writes
                            *_eval.csv, *_eval_summary.csv and
                            *_coverage_trajectory.csv to results/.

results/                   129 CSVs (plus training-time .txt files) from
                            the internship's original training/evaluation
                            runs. Unverified and superseded, kept in place
                            because scripts/recompute_plot_numbers.py
                            still reads it; see results/README_LEGACY.txt.
results_verified/          The trustworthy, provenance-tracked dataset,
                            rebuilt from the raw "Colab training results/"
                            downloads. See "Which results are trustworthy"
                            above and docs/VERIFICATION.md.
Colab training results/    Raw Colab zip downloads. Read-only, never
                            committed (gitignored), integrity tracked via
                            data_manifest.json (SHA-256 of every file).
plots_verified/            Figures generated from results_verified/ only,
                            by generate_plots.py.
generate_plots.py          Builds Figures 1 to 7 entirely from
                            results_verified/ CSVs; no hardcoded numbers.
                            Writes to plots_verified/. Also writes
                            results_verified/PLOT_NUMBERS_OLD_VS_VERIFIED.csv,
                            comparing every number this script used to
                            hardcode against the verified value.

archive/
  plots/                    Figures generated from the unverified results/
                            folder, kept for history.
  trained_models/            Three old "_999" smoke-test model directories
                            from the original pipeline, kept for history.
  MOVES.csv                  Record of what was archived and why.

docs/
  AUDIT.md                  The original audit report (on the old
                            results/ data): findings, severity, fixed
                            versus report-only items, and proposed
                            (unimplemented) v3 changes.
  VERIFICATION.md            The results_verified/ rebuild report:
                            raw-data integrity, completeness, reproduction
                            of headline numbers, test results, and the
                            old-versus-verified plot table. Read this, not
                            AUDIT.md, for the trustworthy numbers.

scripts/
  build_data_manifest.py    SHA-256 manifest of every raw Colab file
                            ("check" or "write" mode).
  build_results_verified.py Rebuilds results_verified/phase{1,2,3}/ from
                            the raw Colab zips per the mapping table above.
  reevaluate_all.py          Re-evaluates all 39 models using evaluate.py's
                            own protocol; fills in missing or ambiguous
                            eval sets, cross-checks existing ones.
  run_baselines_verified.py  Fixed-setpoint and random baselines for Phase
                            1 and all six Phase 3 world x drift
                            combinations.
  build_eval_summary_all.py  Aggregates every run's eval.csv into
                            results_verified/EVAL_SUMMARY_PER_RUN.csv and
                            EVAL_SUMMARY_ALL.csv, grouped by algo/world/
                            reward_mode/drift and averaged over seeds.
  ppo_a2c_diagnostic.py       Evaluates every PPO/A2C model (and DQN, for
                            comparison) under both deterministic=True and
                            deterministic=False.
  verify_tuned_hyperparameters.py
                            Trains a 10,000-step smoke test per algorithm
                            with --tuned and checks the saved model's
                            hyperparameters against training/hyperparameters.py.
  run_naming.py               Parses and builds both the old four-part and
                            new five-part run-name conventions.
  prove_naming_equivalence.py
                            Proves the new five-part naming changes only
                            the output file name, not training behaviour.
  recompute_plot_numbers.py   Original (pre-results_verified) audit tool;
                            superseded by generate_plots.py's own
                            comparison table, kept for history.

tests/
  snapshot_baseline.py       Freezes a behavioural snapshot of the frozen
                            fixed-setpoint policy across every world
                            preset, reward mode, drift condition and seed
                            combination into baseline_snapshot.json.
  baseline_snapshot.json     The frozen snapshot itself, recorded before
                            any audit edits. Do not regenerate casually;
                            it is the ground truth every later change is
                            checked against.
  test_regression.py         Re-runs the same sweep and asserts bit-exact
                            equality against the snapshot. Must pass after
                            any change to envs/, agents/, or training/.
  test_env_properties.py     General correctness tests: Gymnasium
                            check_env, seed-determinism, observation-space
                            bounds, and full-episode runs for every reward
                            mode x world preset x drift condition.

data_manifest.json          SHA-256 and size of every file under
                            "Colab training results/", recorded before any
                            edit. Re-check any time with
                            scripts/build_data_manifest.py check.
requirements.txt            Exact pinned versions for training, evaluation
                            and plotting.
requirements-dev.txt        Adds pytest, needed only to run tests/.
```

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# to also run the test suite:
pip install -r requirements-dev.txt
```

Tested with Python 3.12.5 and Stable-Baselines3 2.9.0. See
`requirements.txt` for the full pinned set (numpy, gymnasium, torch,
pandas, matplotlib, pygame-ce).

## Regenerating figures from results_verified/

```
venv\Scripts\python.exe scripts\build_data_manifest.py check   # confirm raw zips are unchanged
python generate_plots.py                                        # figures -> plots_verified/
```

To rebuild `results_verified/` itself from the raw Colab zips (not usually
needed, since it is already committed):

```
venv\Scripts\python.exe scripts\build_results_verified.py
venv\Scripts\python.exe scripts\reevaluate_all.py
venv\Scripts\python.exe scripts\run_baselines_verified.py
venv\Scripts\python.exe scripts\build_eval_summary_all.py
venv\Scripts\python.exe scripts\ppo_a2c_diagnostic.py
python generate_plots.py
```

## Retraining

**Baselines** (no training; establish floor and ceiling):
```
python agents/random_agent_v2.py --world hard --reward_mode multi_objective
python agents/fixed_setpoint_agent_v2.py --world hard --reward_mode multi_objective
python agents/complexity_check.py          # validates easy/medium/hard ladder
```
Both baseline agents accept `--no_drifts` and `--render`.

**Training** (DQN, PPO or A2C, any world/reward_mode/seed):
```
python training/train_dqn.py --world hard --reward_mode multi_objective --seed 42
python training/train_ppo.py --world hard --reward_mode multi_objective --seed 42
python training/train_a2c.py --world hard --reward_mode multi_objective --seed 42
```

Add `--tuned` to use the hyperparameters confirmed from the Colab-trained
models (`training/hyperparameters.py`): `learning_rate=0.001` for all
three algorithms, `batch_size=64` for DQN, `n_steps=2048` for PPO,
`n_steps=64` for A2C, `gamma=0.99` throughout. Without `--tuned`, each
script keeps its own older CLI defaults (`learning_rate=0.0003`, and
`n_steps=2048` for A2C), unchanged from before this flag existed.
`scripts/verify_tuned_hyperparameters.py` confirms a `--tuned` run's saved
model matches these values for all three algorithms.

**Evaluation** (any trained model, 20 episodes on seeds 100-119):
```
python training/evaluate.py --model_path trained_models/dqn_hard_multi_objective_42/model \
    --algo dqn --world hard --reward_mode multi_objective --seed 42
```
`--seed` here is only the *training* seed, used to name the output files;
evaluation itself always uses seeds 100-119. Pass `--no_drifts` if the
model was trained with it, so evaluation uses the matching environment.

**Tests**:
```
python tests/test_regression.py                 # must pass after any edit
python -m pytest tests/test_env_properties.py -q
```

## Run-naming convention

For any run done before the verification pass (everything in `results/`),
run names follow `{algo}_{world}_{reward_mode}_{seed}`, with no
drift-condition component; a with-drifts and a no-drifts run of the same
algo/world/reward_mode/seed silently overwrote each other's results. See
`docs/AUDIT.md` finding #12 and `results/README_LEGACY.txt`.

For any new run, `train_dqn.py`, `train_ppo.py`, `train_a2c.py` and
`evaluate.py` accept an opt-in `--new_naming` flag that instead builds the
collision-free five-part name used throughout `results_verified/`:
`{algo}_{world}_{reward_mode}_{drift}_{seed}`, where `drift` is `drifts`
or `nodrifts`. The flag changes only the output file or folder name; see
`scripts/prove_naming_equivalence.py`, which trains the same configuration
with and without it and diffs the resulting learning curves byte for
byte. `scripts/run_naming.py` parses both the old four-part and new
five-part names.

## Headline results

From `results_verified/EVAL_SUMMARY_ALL.csv` (mean over 3 seeds x 20
evaluation episodes = 60 episodes per row; full detail and provenance in
`docs/VERIFICATION.md`):

| algo | world | reward_mode | drift | mean_reward | mean_biomass | mean_coverage % | extinctions |
|---|---|---|---|---|---|---|---|
| dqn | hard | multi_objective | drifts | 5220.89 | 497.24 | 100.00 | 0/60 |
| ppo | hard | multi_objective | drifts | 2836.02 | 275.03 | 94.33 | 1/60 |
| a2c | hard | multi_objective | drifts | 1180.21 | 95.59 | 90.93 | 0/60 |
| dqn | hard | multi_objective | nodrifts | 4958.39 | 468.86 | 99.90 | 0/60 |
| dqn | hard | biomass_only | drifts | 4690.70 | 473.03 | 100.00 | 0/60 |
| dqn | hard | sustainability | drifts | 4760.38 | 459.85 | 98.72 | 0/60 |
| dqn | hard | efficiency | drifts | 3467.60 | 25.98 | 46.82 | 1/60 |
| dqn | hard | turbidity | drifts | 1596.26 | 42.79 | 86.88 | 0/60 |
| dqn | hard | turbidity_delta | drifts | 7.45 | 54.01 | 89.57 | 0/60 |
| dqn | easy | multi_objective | drifts | 5302.80 | 497.39 | 100.00 | 0/60 |
| dqn | easy | multi_objective | nodrifts | 4574.02 | 424.84 | 99.38 | 0/60 |
| dqn | medium | multi_objective | drifts | 5132.95 | 487.34 | 100.00 | 0/60 |
| dqn | medium | multi_objective | nodrifts | 4806.42 | 459.46 | 100.00 | 0/60 |

## Known limitations

- **PPO/A2C deterministic versus stochastic evaluation.** `evaluate.py`
  always uses `deterministic=True`. Both PPO and A2C score substantially
  higher under `deterministic=False` (roughly 1.3 to 3.3 times for PPO,
  2.4 to 2.6 times for A2C), and the stochastic-mode reward tracks
  training reward far more closely than the deterministic-mode reward
  does. The published figures use the deterministic value throughout; see
  `results_verified/diagnostics/ppo_a2c_deterministic_vs_stochastic.csv`
  and `docs/VERIFICATION.md` Step 5.
- **Drift effect not statistically tested.** The with-drifts versus
  no-drifts comparison (Figures 5 and 6) reports descriptive means over
  three seeds only; no significance test has been run, so a numerical
  difference in the headline table above should not be read as a tested
  effect.
- **Unresolved `_999_` smoke-test reproducibility gap.** One early
  smoke-test model and its stored evaluation CSV, outside the 42/43/44
  protocol, could not be reproduced end to end when re-run against
  today's environment. This is unrelated to the 39 models in
  `results_verified/`; see `docs/AUDIT.md` finding #14.
- **Five mismatched re-evaluations.** Re-running evaluation for
  `dqn_hard_efficiency_drifts_42/43/44`, `dqn_hard_sustainability_drifts_43`
  and `dqn_hard_turbidity_drifts_43` gave small numerical differences from
  the originally stored results (under 0.5 percent of mean reward in every
  case). Both versions are kept side by side rather than one overwriting
  the other; see `results_verified/REEVALUATION_REPORT.csv`.

## Further reading

`docs/AUDIT.md` and `docs/VERIFICATION.md` hold the full findings,
verification tables, and methodology behind everything summarised above.
