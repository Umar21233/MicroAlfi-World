# Microalfi World — RL on a Cellular-Automaton Bioreactor

A reinforcement-learning project simulating algae cultivation as a 10x10
cellular-automaton (CA) grid world. A single RL agent sits **outside** the
grid and controls global bioreactor conditions (light, CO2, nutrients, pH,
toxin filtering, radiation shielding, humidity, growth pulses); colony
behaviour (spreading, mutation, toxin buildup, collapse) **emerges** from
local cell-to-cell rules the agent never touches directly.

This repository continues a completed internship project into a final-year
project. The internship's experiments are **done** — their results live in
`results/` and `trained_models/`, and a paper is being written from them.
**Nothing in `envs/alphae_world_env_v2.py` may change behaviourally**: every
constant, CA rule, reward formula, and reset/seed sequence must keep
producing exactly the numbers already on disk. See `AUDIT.md` for the full
audit of this codebase and a list of proposed (not yet implemented) v3
improvements that would require a new, separate env file.

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
                            internship's training/evaluation runs. Read-only.
trained_models/             Saved SB3 models + config.txt. Read-only.
plots/                      Generated paper figures. Read-only.
generate_plots.py           Builds Figures 1-7 from results/ CSVs. NOTE:
                            Figures 2, 3, 5 and 6 currently use hardcoded
                            numbers rather than reading results/ directly —
                            see AUDIT.md Step 2f for a full hardcoded-vs-
                            recomputed comparison table.

scripts/
  recompute_plot_numbers.py Read-only audit tool: recomputes the Figure
                            2/3/5/6 numbers from results/ CSVs (and, for the
                            two untrained baselines, by re-running them
                            fresh) so they can be checked against the
                            hardcoded values in generate_plots.py.

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

AUDIT.md                    Full audit report: findings, severity, fixed vs
                            report-only, and proposed (unimplemented) v3
                            changes.
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
was trained with it, so evaluation uses the matching environment.

**Plots** (from project root, reads results/, writes plots/):
```
python generate_plots.py
```

**Tests**:
```
python tests/test_regression.py                 # must pass after any edit
python -m pytest tests/test_env_properties.py -q
```

## Reproducibility note

Run names follow `{algo}_{world}_{reward_mode}_{seed}`; CSVs follow
`{run_name}_{type}.csv`. This naming does **not** encode the drift
condition (`no_drifts`), so a with-drifts and a no-drifts run of the same
algo/world/reward_mode/seed would silently overwrite each other's results —
see AUDIT.md for where this already limits what Phase 3's no-drift figures
can be verified against.
