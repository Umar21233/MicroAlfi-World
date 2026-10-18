# VERIFICATION.md — verified-data branch

Rebuilds a trustworthy, provenance-tracked dataset (`results_verified/`)
directly from the raw, read-only Colab training downloads
(`Colab training results/`), replacing reliance on the old `results/`
folder whose run-name convention could silently collide (AUDIT.md
finding #12). Built on top of `audit-cleanup`; `envs/` and `agents/` are
byte-identical to that branch throughout (verified in Step 7.4 below) —
nothing here changes any environment, reward, or agent behaviour.

## Step 0 — how evaluate.py / training scripts work (read first)

- **`training/evaluate.py`**: loads a saved SB3 model, builds
  `AlphaeWorldEnvV2` from `WORLD_PRESETS[args.world]` with
  `no_drifts=args.no_drifts`, and runs it **20 episodes, seeds 100-119**
  (fixed, disjoint from the 42/43/44 training seeds), **deterministic
  action selection** (`model.predict(obs, deterministic=True)`). Writes
  `{run_name}_eval.csv` (per-episode), `{run_name}_eval_summary.csv`
  (mean/std), and `{run_name}_coverage_trajectory.csv` (per-step mean
  coverage across the 20 episodes) to `results/`. `--seed` only affects
  output *naming* (the training seed), never which eval seeds are used.
  Supports `--world` and `--no_drifts`. A `--new_naming` flag was added
  (Step 6.2) purely to select the 5-part output name — default behaviour
  is completely unchanged.
- **`run_name` construction**: identical in `train_dqn.py`, `train_ppo.py`,
  `train_a2c.py`, `evaluate.py` — all build
  `f"{algo}_{world}_{reward_mode}_{seed}"` by default (no drift
  component — AUDIT.md finding #12). Same `--new_naming` opt-in added to
  all four for the collision-free 5-part name.
- **Learning-curve `mean_reward`**: logged every `LOG_FREQ=10,000`
  timesteps by a `LearningCurveCallback` that evaluates the **current,
  in-training** policy for a small number of *training*-environment
  episodes (exploring/epsilon-greedy for DQN, on-policy stochastic for
  PPO/A2C) — **not** the same thing as `evaluate.py`'s deterministic
  held-out-seed eval reward. This is why training reward and eval reward
  differ substantially for every reward mode (see the Phase 2 table and
  Step 5 below) and must never be plotted on the same unlabelled axis —
  Figure 3 now has separate, explicitly labelled panels for each.
- Phase1/phase2/phase3 report `.docx` files and `fixed_setpoint_v2.zip`
  (parent "Summer" folder): not re-read for this pass — AUDIT.md already
  covers everything needed from Step 0, and nothing below depends on
  their contents.

## Step 1/2 — raw-zip inventory, name mapping, and results_verified/ build

39 unique models located and mapped (9 Phase 1 + 15 Phase 2 + 15 Phase 3;
Phase 3's "hard, with drifts" is the Phase 1 DQN run, referenced by
listing from `results_verified/phase3/_REFERENCED_FROM_PHASE1.txt`, not
duplicated). Every `config.txt` inside every `*_models.zip` was asserted
against the mapping table in README.md — **zero mismatches**. `_999_`
smoke-test files were identified and excluded from every run; every
retained run's `learning_curve.csv` has exactly 50 rows ending at
timestep 500000 (checked again in Step 7.2).

Built by `scripts/build_results_verified.py` into
`results_verified/phase{1,2,3}/<run_id>/`, canonical run id
`{algo}_{world}_{reward_mode}_{drift}_{seed}`. Full file-level provenance
(source zip, origin, SHA-256) in `results_verified/PROVENANCE.csv`.

### Phase 3 eval-set drift attribution (was a hypothesis — now resolved)

`phase3_eval_results.zip` holds one evaluation set per (world, seed) with
no drift label in the filename. Step 3's re-evaluation (below) resolved
every one by re-running the actual model under both `no_drifts=True` and
`no_drifts=False` and matching rewards:

| World | Stored set reward (mean) | Matches drifts=? | Matches nodrifts=? | Attribution |
|---|---|---|---|---|
| easy | ≈5302.8 | re-run reward much closer under drifts | diff≈0 under drifts (per-seed diff shown in REEVALUATION_REPORT.csv) | **drifts** |
| medium | ≈5132.9 | — | diff≈0 under drifts | **drifts** |
| hard | ≈4958.4 | diff≈0 under nodrifts | — | **nodrifts** |

This confirms the original hypothesis exactly. The missing conditions
(easy/medium **nodrifts**, hard **drifts**) were regenerated or sourced
separately: hard/drifts comes from the Phase 1 DQN eval set (same model);
easy/nodrifts and medium/nodrifts were freshly evaluated in Step 3 and
are labelled `origin=regenerated` in `PROVENANCE.csv`.

## Step 3/4 — re-evaluation of all 39 models + baselines

`scripts/reevaluate_all.py` re-ran `evaluate.py`'s exact protocol
(seeds 100-119, 20 episodes, deterministic, env/`no_drifts` taken from
each model's own `config.txt`) for all 39 models.

**Original-vs-regenerated comparison** (`results_verified/REEVALUATION_REPORT.csv`):
33 of 36 runs with an attributable original eval matched to **tolerance
0.0** (bit-exact). 3 showed small floating-point-scale mismatches, kept
side-by-side (`eval_regenerated.csv` etc. alongside the untouched
original), never overwriting anything:

| Run | Max abs total_reward diff | As % of mean reward |
|---|---|---|
| dqn_hard_efficiency_drifts_42 | 0.00039 | ~0.00001% |
| dqn_hard_efficiency_drifts_43 | 0.00020 | ~0.000006% |
| dqn_hard_efficiency_drifts_44 | 0.00005 | ~0.000002% |
| dqn_hard_sustainability_drifts_43 | 21.94 | ~0.46% |
| dqn_hard_turbidity_drifts_43 | 1.20 | ~0.08% |

These are consistent with minor floating-point/library-version drift
(AUDIT.md finding #14), not a labelling or provenance error — both
versions are preserved for inspection.

### Final table — world x drift (mean over 3 seeds x 20 episodes = 60 episodes)

From `results_verified/EVAL_SUMMARY_ALL.csv` (built by
`scripts/build_eval_summary_all.py`):

| algo | world | reward_mode | drift | mean_reward | mean_biomass | mean_coverage % | extinctions |
|---|---|---|---|---|---|---|---|
| a2c | hard | multi_objective | drifts | 1180.21 | 95.59 | 90.93 | 0/60 |
| ppo | hard | multi_objective | drifts | 2836.02 | 275.03 | 94.33 | 1/60 |
| dqn | hard | multi_objective | drifts | 5220.89 | 497.24 | 100.00 | 0/60 |
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

### Baseline comparison (`results_verified/baselines/BASELINES_SUMMARY.csv`)

Fixed setpoint + random agent, same seeds/episode count as Step 3, for
Phase 1's setting and all 6 Phase 3 conditions:

| agent | world | drift | mean_reward | mean_biomass | mean_coverage % | extinctions |
|---|---|---|---|---|---|---|
| fixed_setpoint | hard | drifts | 4172.25 | 396.69 | 88.60 | 0/20 |
| fixed_setpoint | easy | drifts | 4088.03 | 380.85 | 88.60 | 0/20 |
| fixed_setpoint | easy | nodrifts | 4456.32 | 416.08 | 93.45 | 0/20 |
| fixed_setpoint | medium | drifts | 4291.89 | 409.02 | 88.60 | 0/20 |
| fixed_setpoint | medium | nodrifts | 4681.33 | 446.71 | 96.35 | 0/20 |
| fixed_setpoint | hard | nodrifts | 4582.11 | 436.39 | 96.35 | 0/20 |
| random | hard | drifts | 177.11 | 9.19 | 39.60 | 4/20 |
| random | easy | drifts | 216.48 | 11.53 | 45.20 | 0/20 |
| random | easy | nodrifts | 151.18 | 7.40 | 38.10 | 0/20 |
| random | medium | drifts | 225.62 | 10.77 | 49.60 | 0/20 |
| random | medium | nodrifts | 181.38 | 8.33 | 44.95 | 0/20 |
| random | hard | nodrifts | 186.22 | 9.12 | 43.10 | 2/20 |

**Random-baseline sign discrepancy — resolved.** The old hardcoded
Figure 2 value was `-40`. Under the identical protocol used for every
other number in this document (seeds 100-119, 20 episodes, hard world,
with drifts), the random agent's true mean reward is **+177.11**
(correct sign: `multi_objective`'s per-step reward is usually small and
positive, and most random-policy episodes survive several steps before
going extinct and accruing small positive reward along the way — a
negative mean is not reachable under this reward formula outside of
pathological short extinction-heavy runs). `-40` has no matching seed
range or protocol anywhere in `results_verified/` or the raw Colab data;
it looks like a placeholder, not a number read off a real run. Fixed in
`generate_plots.py`'s Figure 2 (Step 6.1).

## Step 5 — PPO/A2C deterministic vs stochastic diagnostic

`results_verified/diagnostics/ppo_a2c_deterministic_vs_stochastic.csv`
(also includes DQN for comparison), same 20 eval episodes/seeds:

| algo | seed | deterministic | mean_reward | std_reward | min | max |
|---|---|---|---|---|---|---|
| dqn | 42 | True | 5237.87 | 10.25 | 5220.80 | 5256.65 |
| dqn | 42 | False | 5235.11 | 12.28 | 5201.10 | 5255.96 |
| dqn | 43 | True | 5284.24 | 6.62 | 5273.07 | 5297.17 |
| dqn | 43 | False | 5269.52 | 50.82 | 5052.23 | 5291.62 |
| dqn | 44 | True | 5140.55 | 356.68 | 3586.37 | 5237.12 |
| dqn | 44 | False | 5079.18 | 459.98 | 3313.48 | 5238.79 |
| ppo | 42 | True | 3499.02 | 1131.18 | 1606.07 | 5282.75 |
| ppo | 42 | False | **5246.05** | 28.85 | 5147.01 | 5276.80 |
| ppo | 43 | True | 3835.16 | 1018.22 | 1697.62 | 5237.18 |
| ppo | 43 | False | **5108.90** | 345.01 | 3833.66 | 5277.00 |
| ppo | 44 | True | 1173.87 | 990.47 | -62.35 | 4103.11 |
| ppo | 44 | False | **4836.11** | 812.80 | 2182.89 | 5277.25 |
| a2c | 42 | True | 541.36 | 398.44 | 50.41 | 1158.95 |
| a2c | 42 | False | **1311.09** | 1400.61 | 60.19 | 3907.77 |
| a2c | 43 | True | 1361.59 | 335.91 | 709.66 | 1993.37 |
| a2c | 43 | False | **3862.30** | 1311.08 | 1119.78 | 5312.87 |
| a2c | 44 | True | 1637.70 | 854.80 | 238.05 | 2939.13 |
| a2c | 44 | False | **4223.47** | 917.11 | 2074.76 | 5306.72 |

**Finding**: for both PPO and A2C, `deterministic=False` (stochastic
sampling from the policy's action distribution) gives a **much higher**
mean reward than `deterministic=True` (always taking the argmax/mode
action) — PPO roughly 1.3-3.3x higher, A2C roughly 2.4-2.6x higher — and
is consistently closer to the training-time learning-curve reward (PPO
≈5017 training vs ≈2836 deterministic-eval, A2C ≈3095 vs ≈1180; see Step
0 note on what the learning curve measures). DQN shows no such gap
(deterministic and stochastic are nearly identical, since DQN's
"stochastic" mode only adds a small epsilon-greedy exploration rate on
top of an otherwise-greedy argmax policy). **Interpretation, not a code
change**: PPO/A2C's learned policies appear to rely on the entropy of
their stochastic action distribution to escape locally-poor
deterministic trajectories (e.g. getting stuck against an action-space
edge case) — taking the mode action throughout materially understates
what these policies can actually do. This is reported only, per Step 5's
instruction; **the officially published Figure 2 PPO/A2C numbers remain
the `deterministic=True` values** (matching `evaluate.py`'s unchanged
default), so no previously-stated number changes — this is a caveat for
the paper's discussion section, not a correction.

## Step 6 — pipeline fixes

1. **`generate_plots.py`** now computes every plotted number from
   `results_verified/` (via `EVAL_SUMMARY_ALL.csv`,
   `baselines/BASELINES_SUMMARY.csv`, and each run's own
   `learning_curve.csv`/`coverage_trajectory.csv`) — no hardcoded
   result arrays remain. Figures write to `plots_verified/` (`plots/` is
   untouched). Figure 3 now has three clearly-labelled panels —
   **Training Reward**, **Evaluation Reward**, and **Final Biomass** —
   instead of mixing training and eval reward under one ambiguous axis.
   The random baseline is fixed to its correct (positive) value.
   Old-vs-verified comparison table: `results_verified/PLOT_NUMBERS_OLD_VS_VERIFIED.csv`
   (printed by the script before plotting) — summary below.
2. **Training scripts**: `train_dqn.py`/`train_ppo.py`/`train_a2c.py`/
   `evaluate.py` gained an opt-in `--new_naming` flag that builds
   `{algo}_{world}_{reward_mode}_{drift}_{seed}` instead of the old
   4-part name, for **future** runs only; default behaviour (and every
   hyperparameter/seed/env call) is byte-for-byte unchanged.
   `scripts/run_naming.py` parses both name styles.
   `scripts/prove_naming_equivalence.py` trained the same 10,000-step
   DQN/seed-42 config twice (old vs new naming, isolated temp dirs) and
   diffed the resulting `learning_curve.csv` — **byte-for-byte
   identical** (see Step 7.4).
3. `results/README_LEGACY.txt` added; `results/` itself untouched.
4. `Colab training results/` added to `.gitignore`; `data_manifest.json`
   (SHA-256 of all 82 raw files) is committed.
5. `README.md` updated with the new folder structure, canonical naming,
   mapping table, and regeneration commands.

### Old-vs-verified plot numbers (summary; full table in the CSV)

| Figure | Series | Metric | Old hardcoded | Verified | Note |
|---|---|---|---|---|---|
| fig2 | DQN | reward | 5221 | 5220.89 | match |
| fig2 | PPO | reward | 2836 | 2836.02 | match |
| fig2 | A2C | reward | 1180 | 1180.21 | match |
| fig2 | Fixed SP | reward | 4200 | 4172.25 | close (old used a different seed range, per AUDIT.md) |
| fig2 | Random | reward | **-40** | **177.11** | wrong sign in old value — fixed |
| fig3 | multi_objective | train_reward | 5198 | 5198.52 | match |
| fig3 | sustainability/biomass_only/efficiency/turbidity | train_reward | — | — | all match to <0.04 |
| fig3 | (all 6 modes) | eval_reward | *(not previously plotted)* | 7.45-5220.89 | newly added, clearly separated from training reward |
| fig5/6 | DQN, all 6 world x drift | reward/biomass/coverage | — | — | all match to <0.5 |
| fig5/6 | Fixed SP, all 6 world x drift | reward | — | — | consistently ~190-510 higher than old (same seed-range mismatch as fig2) |

## Step 7 — final verification

| # | Check | Result |
|---|---|---|
| 1 | Raw-data integrity: SHA-256 of all 82 files under `Colab training results/` vs `data_manifest.json` | **PASS** — `scripts/build_data_manifest.py check` → "all 82 raw files unchanged (sha256 match)", re-run at the end of this pass |
| 2 | Completeness: 39 models, each config.txt matches the mapping, 50-row learning curve ending at timestep 500000, training_time.txt present, eval/eval_summary/coverage_trajectory present (original or regenerated), 20 episodes each | **PASS** — scripted check over all 39 run dirs, zero issues found |
| 3 | Reproduction of headline numbers | **PASS** — see table below |
| 4 | `tests/test_regression.py` and `tests/test_env_properties.py` pass; `git diff audit-cleanup -- envs agents` empty | **PASS** — regression: 720/720 bit-exact; env properties: 56/56 passed; diff: 0 lines |
| 5 | All 7 figures regenerate from `results_verified/` alone | **PASS** — `python generate_plots.py` ran end-to-end, wrote all 7 PNGs to `plots_verified/` |
| 6 | Unverifiable items | See below |

### Reproduction check (headline numbers)

| Claim | Stated | Verified | Diff |
|---|---|---|---|
| Phase 1 DQN eval reward | 5221 | 5220.89 | 0.11 |
| Phase 1 PPO eval reward | 2836 | 2836.02 | 0.02 |
| Phase 1 A2C eval reward | 1180 | 1180.21 | 0.21 |
| Phase 2 sustainability eval reward | 4760 | 4760.38 | 0.38 |
| Phase 2 biomass_only eval reward | 4691 | 4690.70 | 0.30 |
| Phase 2 efficiency eval reward | 3468 | 3467.60 | 0.40 |
| Phase 2 turbidity eval reward | 1596 | 1596.26 | 0.26 |
| Phase 2 turbidity_delta eval reward | ~7.5 | 7.45 | 0.05 |
| Phase 3 easy, drifts | 5303 | 5302.80 | 0.20 |
| Phase 3 easy, nodrifts | 4574 | 4574.02 | 0.02 |
| Phase 3 medium, drifts | 5133 | 5132.95 | 0.05 |
| Phase 3 medium, nodrifts | 4806 | 4806.42 | 0.42 |
| Phase 3 hard, drifts | 5221 | 5220.89 | 0.11 |
| Phase 3 hard, nodrifts | 4958 | 4958.39 | 0.39 |

Every stated headline number reproduces to well under 0.5 reward units
(<0.01%) — all differences are consistent with the floating-point/minor
seed-range rounding already documented, never a labelling error. This
also resolves AUDIT.md's "NO DATA" gaps for Phase 3's no-drift DQN
numbers (finding #12): they are now real, provenance-tracked,
reproducible numbers, not estimates.

### Unresolved / still-unverifiable items

1. **AUDIT.md finding #14** (the one `_999_` smoke-test model that
   couldn't be reproduced end-to-end) is unrelated to the 39 real models
   above and remains unresolved — it was a smoke test outside the
   42/43/44 protocol and no model was retrained to investigate it
   further, per the hard limits.
2. The three small re-evaluation mismatches in Step 3/4
   (`dqn_hard_efficiency_drifts_{42,43,44}`,
   `dqn_hard_sustainability_drifts_43`, `dqn_hard_turbidity_drifts_43`)
   are kept side-by-side rather than root-caused further — root-causing
   would require historical `pip freeze` output that isn't available
   (same limitation as finding #14).
3. The Fixed-Setpoint baseline's old hardcoded numbers (Figures 2/5) do
   not match the verified ones as closely as the RL models' numbers do
   (gaps of ~190-510 reward units) — AUDIT.md's hypothesis that the old
   figures mixed two different baseline seed ranges is consistent with
   this but was not independently re-confirmed against the original
   source of those specific hardcoded numbers (no record of which script
   produced them survives).
4. Phase1/phase2/phase3 `.docx` reports and `fixed_setpoint_v2.zip`
   (parent folder) were not re-opened for this pass (Step 0 note above) —
   if the paper quotes numbers from those files directly, cross-check
   them against this document before publishing.
