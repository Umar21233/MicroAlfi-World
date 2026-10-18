# Microalfi World — RL on a Cellular-Automaton Bioreactor

A reinforcement-learning project simulating algae cultivation as a 10x10
cellular-automaton (CA) grid world. A single RL agent sits **outside** the
grid and controls global bioreactor conditions (light, CO2, nutrients, pH,
toxin filtering, radiation shielding, humidity, growth pulses); colony
behaviour (spreading, mutation, toxin buildup, collapse) **emerges** from
local cell-to-cell rules the agent never touches directly.

This repository continues a completed internship project into a final-year
project. The internship's experiments are **done**.
**Nothing in `envs/alphae_world_env_v2.py` may change behaviourally**: every
constant, CA rule, reward formula, and reset/seed sequence must keep
producing exactly the numbers already on disk. See `AUDIT.md` for the full
audit of this codebase and a list of proposed (not yet implemented) v3
improvements that would require a new, separate env file.

## Which results are trustworthy?

Use **`results_verified/`**, not `results/`. `results/` is the internship's
original local output folder; its run-name convention
(`{algo}_{world}_{reward_mode}_{seed}`) has no drift-condition component, so
a with-drifts and a no-drifts run of the same algo/world/reward_mode/seed
silently overwrote each other's files (AUDIT.md finding #12). See
`results/README_LEGACY.txt`.

`results_verified/` was rebuilt from scratch directly from the raw,
read-only Colab training downloads (`Colab training results/`, SHA-256
tracked in `data_manifest.json`), using a collision-free 5-part run id:

```
{algo}_{world}_{reward_mode}_{drift}_{seed}
```

where `drift` is `drifts` (`no_drifts=False`) or `nodrifts`
(`no_drifts=True`). See `VERIFICATION.md` for the full rebuild process, the
verified result tables, and the old-vs-verified plot-number comparison.

### Zip-stem -> canonical-run-id mapping

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
(which file came from which zip, original-vs-regenerated) is in
`results_verified/PROVENANCE.csv`.

### Regenerating results_verified/ and the figures

```
venv\Scripts\python.exe scripts\build_data_manifest.py check   # verify raw zips untouched
venv\Scripts\python.exe scripts\build_results_verified.py      # rebuild results_verified/ from the raw zips
venv\Scripts\python.exe scripts\reevaluate_all.py               # re-run evaluation for all 39 models
venv\Scripts\python.exe scripts\run_baselines_verified.py       # fixed-setpoint + random baselines
venv\Scripts\python.exe scripts\build_eval_summary_all.py       # aggregate per-run eval.csv -> summary tables
venv\Scripts\python.exe scripts\ppo_a2c_diagnostic.py           # deterministic vs stochastic diagnostic
python generate_plots.py                                        # figures -> plots_verified/ (reads results_verified/ only)
```

## Folder structure

```
envs/
  alphae_world_env_v2.py   FROZEN. The CA grid-world Gymnasium environment
                            (10x10 grid, 13-dim obs, 11 actions, 6 reward
                            modes). This is what every result in results/
                            and every model in trained_models/ depends on.
  alphae_world_env.py       Superseded v1 env (6-dim obs, 7 actions, single
                            reward formula, no CA grid). Kept for history;
                            not used by any current training/eval script.
  renderer.py               Pygame renderer for AlphaeWorldEnvV2 (--render).

agents/
  random_agent_v2.py        Baseline 1 (floor): uniform-random actions.
                            Defines WORLD_PRESETS (easy/medium/hard).
  fixed_setpoint_agent_v2.py Baseline 2 (ceiling): hand-coded priority
                            controller, no learning.
  complexity_check.py       Validates that easy/medium/hard form a real
                            difficulty ladder (random vs fixed setpoint).
  random_agent.py, fixed_setpoint_agent.py
                            Equivalents for the superseded v1 env.

training/
  train_dqn.py, train_ppo.py, train_a2c.py
                            Train one Stable-Baselines3 model on
                            AlphaeWorldEnvV2 for a given (world, reward_mode,
                            seed). Saves model + config.txt under
                            trained_models/, plus a learning-curve CSV and a
                            training-time file under results/.
  evaluate.py               Loads a saved model and evaluates it
                            deterministically for 20 episodes on seeds
                            100-119 (disjoint from training seeds). Writes
                            *_eval.csv, *_eval_summary.csv and
                            *_coverage_trajectory.csv to results/.

results/                    129 CSVs (+ training-time .txt files) from the
                            internship's training/evaluation runs.
                            UNVERIFIED/superseded — see
                            results/README_LEGACY.txt. Kept, unmodified,
                            for history.
results_verified/           The trustworthy, provenance-tracked dataset,
                            rebuilt from the raw "Colab training results/"
                            downloads. See "Which results are trustworthy?"
                            above and VERIFICATION.md.
Colab training results/     Raw Colab zip downloads. READ-ONLY, never
                            committed (gitignored) — integrity tracked via
                            data_manifest.json (SHA-256 of every file).
trained_models/             Saved SB3 models + config.txt from results/'s
                            era. Read-only, unverified (see results/).
plots/                      Figures generated from the unverified results/.
                            Read-only, kept for history.
plots_verified/             Figures generated from results_verified/ only —
                            the trustworthy figures. See generate_plots.py.
generate_plots.py           Builds Figures 1-7 entirely from
                            results_verified/ CSVs (no hardcoded numbers).
                            Writes to plots_verified/. Also writes
                            results_verified/PLOT_NUMBERS_OLD_VS_VERIFIED.csv,
                            comparing every number this script used to
                            hardcode against the verified value.

scripts/
  build_data_manifest.py    SHA-256 manifest of every raw Colab file
                            ("check" or "write" mode).
  build_results_verified.py Rebuilds results_verified/phase{1,2,3}/ from the
                            raw Colab zips per the mapping table above.
  reevaluate_all.py         Re-evaluates all 39 models using evaluate.py's
                            own protocol; fills in missing/ambiguous eval
                            sets, cross-checks existing ones.
  run_baselines_verified.py Fixed-setpoint + random baselines for Phase 1
                            and all 6 Phase 3 world x drift combinations.
  build_eval_summary_all.py Aggregates every run's eval.csv into
                            results_verified/EVAL_SUMMARY_PER_RUN.csv and
                            EVAL_SUMMARY_ALL.csv (grouped by
                            algo/world/reward_mode/drift, averaged over
                            seeds).
  ppo_a2c_diagnostic.py     Evaluates every PPO/A2C (and DQN, for
                            comparison) model under both
                            deterministic=True and deterministic=False.
  run_naming.py             Parses/builds both the old 4-part and new
                            5-part run-name conventions.
  prove_naming_equivalence.py
                            Proves the new 5-part naming changes only the
                            output file name, not training behaviour.
  recompute_plot_numbers.py Original (pre-results_verified) audit tool:
                            recomputes the old Figure 2/3/5/6 numbers from
                            results/ CSVs. Superseded by
                            generate_plots.py's own comparison table, kept
                            for history.

tests/
  snapshot_baseline.py      Freezes a behavioural snapshot of the frozen
                            fixed-setpoint policy across every (world
                            preset, reward mode, drift condition, seed)
                            combination into baseline_snapshot.json.
  baseline_snapshot.json    The frozen snapshot itself (recorded before any
                            audit edits). Do not regenerate casually — it is
                            the ground truth every later change is checked
                            against.
  test_regression.py        Re-runs the same sweep and asserts bit-exact
                            equality against the snapshot. MUST pass after
                            any change to envs/, agents/, or training/.
  test_env_properties.py    General correctness tests: Gymnasium
                            check_env, seed-determinism, observation-space
                            bounds, and full-episode runs for every reward
                            mode x world preset x drift condition.

AUDIT.md                    Full audit report (on the old results/ data):
                            findings, severity, fixed vs report-only, and
                            proposed (unimplemented) v3 changes.
VERIFICATION.md             The verified-data rebuild: raw-data integrity,
                            completeness, reproduction-of-headline-numbers,
                            test results, and the old-vs-verified plot
                            table. Read this, not AUDIT.md, for the
                            trustworthy numbers.
data_manifest.json          SHA-256 + size of every file under
                            "Colab training results/", recorded before any
                            edit. Re-check anytime with
                            `scripts/build_data_manifest.py check`.
requirements.txt            Exact pinned versions for running training/
                            evaluation/plotting.
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

## Commands by phase

**Baselines** (no training; establish floor/ceiling):
```
python agents/random_agent_v2.py --world hard --reward_mode multi_objective
python agents/fixed_setpoint_agent_v2.py --world hard --reward_mode multi_objective
python agents/complexity_check.py          # validates easy/medium/hard ladder
```
Both baseline agents accept `--no_drifts` and `--render`.

**Phase 1 — algorithm comparison** (DQN / PPO / A2C, hard world,
multi_objective, with drifts, seeds 42/43/44):
```
python training/train_dqn.py --world hard --reward_mode multi_objective --seed 42
python training/train_ppo.py --world hard --reward_mode multi_objective --seed 42
python training/train_a2c.py --world hard --reward_mode multi_objective --seed 42
```

**Phase 2 — reward mode comparison** (DQN, hard world, with drifts, each of
the six reward modes, seeds 42/43/44):
```
python training/train_dqn.py --world hard --reward_mode biomass_only --seed 42
python training/train_dqn.py --world hard --reward_mode turbidity --seed 42
python training/train_dqn.py --world hard --reward_mode turbidity_delta --seed 42
python training/train_dqn.py --world hard --reward_mode efficiency --seed 42
python training/train_dqn.py --world hard --reward_mode sustainability --seed 42
```

**Phase 3 — world difficulty / drift condition** (DQN, multi_objective,
easy/medium/hard x with/without drifts):
```
python training/train_dqn.py --world easy   --reward_mode multi_objective --seed 42
python training/train_dqn.py --world medium --reward_mode multi_objective --seed 42
python training/train_dqn.py --world hard   --reward_mode multi_objective --seed 42 --no_drifts
```

**Evaluation** (any trained model, 20 episodes on seeds 100-119):
```
python training/evaluate.py --model_path trained_models/dqn_hard_multi_objective_42/model \
    --algo dqn --world hard --reward_mode multi_objective --seed 42
```
`--seed` here is only the *training* seed, used to name the output files —
evaluation itself always uses seeds 100-119. Pass `--no_drifts` if the model
was trained with it, so evaluation uses the matching environment. Pass
`--new_naming` to use the collision-free 5-part output name (below).

**Plots** (from project root, reads **results_verified/**, writes
**plots_verified/**):
```
python generate_plots.py
```

**Tests**:
```
python tests/test_regression.py                 # must pass after any edit
python -m pytest tests/test_env_properties.py -q
```

## Reproducibility note

For any run done **before** this verification pass (everything in
`results/` / `trained_models/`), run names follow
`{algo}_{world}_{reward_mode}_{seed}`, with no drift-condition component —
a with-drifts and a no-drifts run of the same algo/world/reward_mode/seed
silently overwrote each other's results. See AUDIT.md finding #12 and
`results/README_LEGACY.txt`.

For any **future** run, `train_dqn.py` / `train_ppo.py` / `train_a2c.py` /
`evaluate.py` accept an opt-in `--new_naming` flag that instead builds the
collision-free 5-part name used throughout `results_verified/`:
`{algo}_{world}_{reward_mode}_{drift}_{seed}`, where `drift` is `drifts` or
`nodrifts`. The flag changes **only** the output file/folder name — see
`scripts/prove_naming_equivalence.py`, which trains the same config with
and without it and diffs the resulting learning curves byte-for-byte.
`scripts/run_naming.py` parses both the old 4-part and new 5-part names.
