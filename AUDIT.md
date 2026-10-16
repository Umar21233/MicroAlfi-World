# AUDIT.md — Microalfi World audit report

Audit performed on the `audit-cleanup` branch, against the baseline captured
in commit `a976e59` (initial commit, before any edit). Every row below marked
**Fixed** was verified not to change a single value in
`tests/baseline_snapshot.json` (720 frozen fixed-setpoint episodes across
every world preset x reward mode x drift condition) — see
`tests/test_regression.py`, which passed bit-exact after every commit.

## Findings

| # | Finding | File : line | Severity | Status | Why |
|---|---------|--------------|----------|--------|-----|
| 1 | `reset()` docstring said starting biomass is 0.20–0.50; code (and the adjacent inline comment) actually use 0.15–0.35 | `envs/alphae_world_env_v2.py:284` | Low | **Fixed** | Pure doc/comment edit, no code touched. Verified via regression test. |
| 2 | Spreading-rule comment said `SPREAD_MIN_BIOMASS` is 0.15; the constant is 0.10 | `envs/alphae_world_env_v2.py:603` | Low | **Fixed** | Pure comment edit. |
| 3 | `renderer.py` header docstring said "AlgaeWorldEnv v2"; the class is `AlphaeWorldEnvV2` | `envs/renderer.py:2` | Low | **Fixed** | Pure doc edit. |
| 4 | Two unpacked-but-unused local variables (`temp`, `toxin`) in `_ca_step` | `envs/alphae_world_env_v2.py:498` | Trivial | **Fixed** | Renamed to `_temp`/`_toxin`; pure unpacking with zero reads anywhere in the function, so removal cannot change any computed value. Verified via regression test. |
| 5 | Unused imports `DEFAULT_WORLD_CONFIG`, `TOXIN_SLOW_THRESH` | `agents/fixed_setpoint_agent_v2.py:34,38` | Trivial | **Fixed** | Neither name was referenced anywhere else in the file. |
| 6 | `requirements.txt` used loose `>=` bounds, was missing `torch` entirely (a direct SB3 dependency), and said `pygame` when the environment actually has `pygame-ce` (a different PyPI package with the same import name) installed | `requirements.txt` | Medium | **Fixed** | Pinned to the exact versions in `venv/` via `pip freeze`. This locks in reproducibility **going forward**; it does not retroactively explain finding #14 below. |
| 7 | **Temperature** (`global_state[1]` / `obs[1]`) drifts every step and is observed, but has **zero effect on growth** anywhere in `_ca_step`, and no action controls it | `envs/alphae_world_env_v2.py:27,99-101,498-590` | Medium | Report-only (documented) | Surprising but intentional/frozen — it is a pure non-actionable, non-influential background signal in every one of the 129 result files. Fixing it (e.g. making it modulate growth) would change every single stored result. Comment added at the constant definition; **no code changed**. |
| 8 | **`local_toxin` is structurally asymmetric**: it only ever *decreases* via action 7 (toxin filter). `TOXIN_DECAY` only lowers the *reported global* toxin observation (`obs[5]`, recomputed each step as `mean(local_toxin) − TOXIN_DECAY`) — it never touches the per-cell `local_toxin` array itself. Growth, photoinhibition, and spreading all only ever *add* to `local_toxin` | `envs/alphae_world_env_v2.py:131-136,704-705` | Medium | Report-only (documented) | A real and non-obvious asymmetry between the observed/reward-facing toxin signal and the underlying per-cell state. Frozen — fixing it would change every reward-mode result that has `toxin` enabled (i.e. most of them). Comment added; **no code changed**. |
| 9 | **`turbidity_delta` first-step artefact**: `prev_coverage` is reset to `0.0` in `reset()`, so step 1 of *every* episode under `turbidity_delta` computes `(coverage − 0) * 10 − cost*0.5`, i.e. exactly the plain `turbidity` reward, for that one step only | `envs/alphae_world_env_v2.py:274,322,424-425` | Low | Report-only (documented) | Affects 1 of 200 steps/episode — small but real, and could confuse anyone reading `turbidity_delta` numbers expecting a pure delta throughout. Comment added; **no code changed**. |
| 10 | **`_get_obs()` bug**: `mean_toxin` and `mean_age` are computed only `if n_living > 1`, falling back to `0.0` otherwise. Since `n_living = max(1, actual_count)`, this silently reports `0.0` for mean toxin/age whenever **exactly one** cell is alive — even though that cell's real toxin/age is almost certainly non-zero | `envs/alphae_world_env_v2.py:711-734` (esp. 722–728) | Medium | Report-only (**this is a genuine logic bug**) | Correct condition should be `n_living >= 1` (equivalently `living_mask.any()`). Frozen: this wrong-but-consistent observation is what every trained model actually saw and learned against in the single-survivor edge case near extinction. Fixing it would change observations (hence agent behaviour and every downstream result) in any episode passing through that state. Proposed for v3 (see below); **no code changed**. |
| 11 | Biomass is hard-capped at 5.0/cell = **500.0 total** across the 10x10 grid | `envs/alphae_world_env_v2.py:342-345` | Informational | Report-only (documented) | Not a bug — working as designed — but several DQN runs land at total biomass ≈ 497–499 (see recomputed Fig2/3/5/6 table below), i.e. they are saturating the hard ceiling, not converging to a free equilibrium. Worth stating explicitly in the paper. Comment added; **no code changed**. |
| 12 | **Run-name / CSV-naming convention has no `no_drifts` marker**: `run_name = f"{algo}_{world}_{reward_mode}_{seed}"` is identical regardless of `no_drifts`. A with-drifts and a no-drifts run of the same algo/world/reward_mode/seed write to the **same** `trained_models/{run_name}/` directory and the **same** `results/{run_name}_*.csv` files | `training/train_dqn.py`, `train_ppo.py`, `train_a2c.py`, `training/evaluate.py` (run_name construction) | **High** | Report-only | This is part of the frozen "run naming... in the training and evaluation scripts," so it cannot be changed here. It directly explains why Figure 5/6's no-drift numbers for seeds 42/43/44 cannot be recomputed or verified at all (see table below) — if a no-drift run of e.g. `dqn_hard_multi_objective_42` was ever done, its results are either (a) never written under a distinguishable name, or (b) silently overwritten by whichever of the with-drift/no-drift runs executed last. Proposed fix for v3: append `_nodrifts` to `run_name` when `no_drifts=True`. |
| 13 | **Figure 2/3/5/6 hardcoded-vs-recomputed mismatches**, most notably DQN's evaluation reward (hardcoded `5221`, recomputed mean-of-3-seeds `4958.39`; no individual seed 42/43/44 matches `5221` either) while PPO (`2836` vs `2836.02`) and A2C (`1180` vs `1180.21`) match almost exactly | `generate_plots.py:97-275` | **High** (for the DQN-specific numbers) | Report-only, per explicit instruction | See the full table under "Step 2f" below. **Not changed** — you asked to see the table and decide, not have the hardcoded numbers silently replaced. |
| 14 | **Pipeline does not reproduce**: re-running `training/evaluate.py` today against the one model+eval pair old enough to test end-to-end (`trained_models/dqn_easy_multi_objective_999/model.zip`, config says `no_drifts=True`) does **not** reproduce the stored `results/dqn_easy_multi_objective_999_eval.csv`, under either `no_drifts=True` or `no_drifts=False` | `training/evaluate.py` + `results/dqn_easy_multi_objective_999_eval*.csv` + `trained_models/dqn_easy_multi_objective_999/model.zip` | **High** | Report-only | Supporting evidence: `model.zip` was last modified 2026-07-23; its own `learning_curve.csv`/`eval*.csv` were last written 2026-08-28 — **over a month later**, which is itself anomalous (for the main seed-42/43/44 runs, `learning_curve.csv`/`training_time.txt` and the `*_eval*.csv` files are one day apart, the normal "train, then evaluate next" pattern). Most likely cause: package-version drift between whenever those CSVs were last (re)written and today's environment (`requirements.txt` had only loose `>=` bounds until finding #6 above was fixed) — RNG- or library-version-sensitive behaviour shifting the trajectory. **Root cause not pinned down further** — doing so would require historical `pip freeze` output that isn't available. This caveats any claim of exact reproducibility in the paper; it does **not** mean the stored results are wrong, only that they cannot currently be independently re-derived byte-for-byte from the one saved model available to test. |
| 15 | **Hyperparameter fairness caveats** (not a bug) — see the hyperparameter table below | `training/train_dqn.py`, `train_ppo.py`, `train_a2c.py` | Low/Medium | Report-only | No tuning/sweep script exists anywhere in the repo — every algorithm ran a single, untuned configuration. A2C's `n_steps` (2048) and `learning_rate` (0.0003) were both manually moved away from A2C's own SB3 defaults (5 and 7e-4 respectively) to match PPO/DQN, which is a defensible "equal budget" choice but is a deliberate asymmetry worth stating in the paper's limitations, not an incidental equal comparison. |
| 16 | Unused/duplicate files: the v1 environment and its baseline agents | `envs/alphae_world_env.py`, `agents/random_agent.py`, `agents/fixed_setpoint_agent.py` | Low | Report-only (listed, not deleted) | Not imported by any current training/evaluation/plotting script — superseded by the `_v2` env/agents. Per instructions, not deleted; listed here as obsolete candidates for a future cleanup once the paper is finished. (No separate "v1 renderer" file exists — there is only one `renderer.py`, already v2-only.) |
| 17 | Stray seeds `999` and `5001`, with far smaller `timesteps` (5,000–25,000 vs the usual 500,000), appear in `results/`/`trained_models/` for `a2c`/`dqn`/`ppo` `easy_multi_objective` | `results/*_999_*`, `results/*_5001_*`, `trained_models/*_999/` | Low | Report-only | Outside the stated 42/43/44 training-seed protocol; these are evidently early pipeline smoke-tests (confirmed by the tiny timestep counts), not part of the Phase 1/2/3 experiment matrix. Worth explicitly excluding from any paper table that aggregates "all seeds" for `easy_multi_objective`, and seed `999` is the specific case that surfaces finding #14. |
| 18 | No hardcoded absolute paths (e.g. `D:\`, `/content`) found anywhere in the codebase | — | — | Checked, nothing to fix | Every script already resolves paths via `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (`PROJECT_ROOT`) or an equivalent relative pattern. Listed here only because Step 2e explicitly asked for a report even if empty. |
| 19 | Reward-mode names, CLI `choices=[...]` lists, and CSV/run-name conventions are consistent across `envs/alphae_world_env_v2.py`, `train_dqn.py`, `train_ppo.py`, `train_a2c.py`, `evaluate.py`, `random_agent_v2.py`, `fixed_setpoint_agent_v2.py`, and `generate_plots.py`'s globs. The one place a naming collision could occur (`turbidity` vs `turbidity_delta` substring match in a glob) is already correctly guarded (`generate_plots.py:176-177` explicitly excludes `turbidity_delta` matches from the plain-`turbidity` series) | — | — | Checked, nothing to fix | Listed for completeness (Step 2a/2b). |
| 20 | Training seeds (42/43/44) and evaluation seeds (100-119, deterministic) never overlap and are used consistently | `training/*.py`, `training/evaluate.py` | — | Checked, nothing to fix | Listed for completeness (Step 2c). The only seeds outside this protocol are the smoke-test seeds in finding #17. |

## Step 2d — hyperparameter table (DQN vs PPO vs A2C)

All three scripts are structurally identical (same `LearningCurveCallback`,
same `run_name`/output convention) and differ only in these values. No
hyperparameter search was performed for any of the three — every value below
is a single CLI default, never swept.

| Hyperparameter | DQN | PPO | A2C | Notes |
|---|---|---|---|---|
| `learning_rate` | 0.0003 (CLI default) | 0.0003 (CLI default) | 0.0003 (CLI default) | Equal across all three. Matches SB3's own default for PPO; **overrides** SB3's A2C default (7e-4) to match. |
| `gamma` | 0.99 (hardcoded) | 0.99 (hardcoded) | 0.99 (hardcoded) | Equal, not exposed via CLI for any of the three. |
| `batch_size` | 64 (CLI default) | — (SB3 default 64, not exposed) | — (A2C has no `batch_size`) | Only DQN exposes this as a CLI flag; PPO's is left at SB3's own default. |
| `n_steps` | — (N/A, off-policy) | 2048 (CLI default) | 2048 (CLI default) | Matches SB3's own default for PPO; **overrides** SB3's A2C default (5) to match PPO's rollout length — a deliberate "equal rollout length" choice, not A2C's natural regime. |
| `total_timesteps` | 500,000 (CLI default) | 500,000 (CLI default) | 500,000 (CLI default) | Equal environment-interaction budget for all three. |
| `seed` | per-run (42/43/44) | per-run (42/43/44) | per-run (42/43/44) | Equal protocol. |
| Everything else | SB3 defaults, untouched (`buffer_size=1e6`, `learning_starts=50000`, `train_freq=4`, `target_update_interval=10000`, `exploration_fraction=0.1`, …) | SB3 defaults, untouched (`n_epochs=10`, `clip_range=0.2`, `gae_lambda=0.95`, `ent_coef=0.0`, `vf_coef=0.5`, …) | SB3 defaults, untouched (`ent_coef=0.0`, `vf_coef=0.5`, `rms_prop_eps=1e-5`, …) | None of these were tuned for any algorithm. |

**Fairness assessment**: the comparison is fair in terms of *equal compute
budget* (same `total_timesteps`, `learning_rate`, `gamma`) but is **not** a
"each algorithm's best achievable performance" comparison, since (a) no
hyperparameter search was run for any of the three, and (b) DQN's off-policy
replay reuses each transition many times per gradient step while
PPO/A2C's on-policy rollouts use each transition roughly once — so
"500,000 timesteps" is an equal *environment-interaction* budget, not an
equal *wall-clock* or *gradient-update* budget. State this as a limitation
in the paper rather than as "a fair head-to-head."

## Step 2f — hardcoded vs recomputed plot numbers

Produced by `scripts/recompute_plot_numbers.py` (read-only; re-runs the two
*untrained* baselines fresh on seeds 100-119, everything else is read
straight from `results/*_eval_summary.csv` / `*_learning_curve.csv`).
**Hardcoded values in `generate_plots.py` were left unchanged** — this table
is for you to decide what to do with.

### Figure 2 — Phase 1 evaluation (hard, multi_objective, with drifts)

| Metric | Hardcoded | Recomputed (mean of seeds 42/43/44, or fresh run) | Match? |
|---|---|---|---|
| DQN reward | 5221 | **4958.39** (per-seed: 42→4860.33, 43→4749.52, 44→5265.32 — none equal 5221) | **No** — off by ~5%, and no individual seed matches either |
| DQN biomass | 497 | 468.86 | Close-ish, not exact |
| DQN coverage % | 100 | 99.90 | Close |
| PPO reward | 2836 | 2836.02 | **Yes** (effectively exact) |
| PPO biomass | 275 | 275.03 | **Yes** |
| PPO coverage % | 94.3 | 94.33 | **Yes** |
| A2C reward | 1180 | 1180.21 | **Yes** |
| A2C biomass | 96 | 95.59 | **Yes** |
| A2C coverage % | 91.0 | 90.93 | **Yes** |
| Fixed SP reward | 4200 | 4172.25 (seeds 100-119) / **3948.89** (seeds 42-61) | Neither matches exactly — see note |
| Fixed SP biomass | 365 | 396.69 (100-119) / 375.37 (42-61) | Neither matches exactly |
| Fixed SP coverage % | 85 | 88.60 (100-119) / **85.55** (42-61) | **seeds 42-61 match closely** |
| Random reward | -40 | 198.34 (100-119) / 188.37 (42-61) | **No match under either seed protocol — wrong sign** |
| Random biomass | 10 | 9.63 / 9.68 | Close under either protocol |
| Random coverage % | 45 | 47.35 / 42.45 | Close under either protocol |

**Reading this**: PPO and A2C's Figure 2 numbers are essentially exact
matches to what's on disk today — only DQN's reward/biomass are off by a
noticeable amount, consistent with finding #14 (DQN pipeline reproducibility
issue). The Fixed-Setpoint and Random baseline numbers line up much better
against **seeds 42-61** (the protocol `agents/complexity_check.py` uses) than
against seeds 100-119 (the protocol `evaluate.py` uses for the RL models) —
i.e. Figure 2 is very likely comparing RL models evaluated on one seed range
against baselines evaluated on a *different* seed range. The Random agent's
hardcoded reward (`-40`) does not match either protocol and is the wrong
sign compared to both (a random policy nets a small **positive** mean reward
under `multi_objective` in practice, since many episodes survive several
steps before going extinct and accrue small positive per-step reward) —
this value looks like a hand-written placeholder rather than a number read
off a real run.

### Figure 3 — Phase 2 reward modes (DQN, hard, with drifts)

| Reward mode | Hardcoded training reward | Recomputed training reward (learning_curve, last point, mean of 3 seeds) | Recomputed **eval** reward (eval_summary, mean of 3 seeds) | Hardcoded biomass | Recomputed biomass | Hardcoded coverage % | Recomputed coverage % |
|---|---|---|---|---|---|---|---|
| multi_objective | 5198 | 4744.00 | 4958.39 | 497 | 468.86 | 100 | 99.90 |
| sustainability | 4719 | **4719.40** (match) | 4760.38 | 460 | 459.85 | 98.7 | 98.72 |
| biomass_only | 4638 | **4637.95** (match) | 4690.70 | 473 | 473.03 | 100 | 100.00 |
| efficiency | 2019 | **2019.03** (match) | **3467.60** | 26 | 25.98 | 46.8 | 46.82 |
| turbidity | 1592 | **1592.04** (match) | 1596.26 | 43 | 42.79 | 86.9 | 86.88 |
| turbidity_delta | (excluded, "different scale") | 7.61 | 7.45 | 54 | 54.01 | 89.6 | 89.57 |

**This confirms your suspicion.** Figure 3's left panel is explicitly titled
"Mean **Training** Reward," and four of five hardcoded values (`sustainability`,
`biomass_only`, `efficiency`, `turbidity`) match the **training**-reward
number (last point of `learning_curve.csv`) almost exactly. `multi_objective`
is the one exception — its hardcoded value (5198) is notably far from both
the recomputed training reward (4744.00) *and* the recomputed eval reward
(4958.39), and is suspiciously close to the same stale-looking `5221`/`5198`
neighbourhood as the DQN numbers in Figure 2/5 (see finding #13/#14) — i.e.
the `multi_objective` number across Figures 2, 3, and 5 looks like it all
traces back to one earlier (and since-superseded) DQN training/eval pass for
the hard-world multi_objective config, not the data currently in `results/`.
Separately: **training reward and evaluation reward are genuinely different
numbers for every reward mode** (e.g. `efficiency`: 2019 training vs 3467.60
eval — a 72% difference) because training reward is measured with an
epsilon-greedy/exploring policy mid-training while eval reward is measured
fully greedily on held-out seeds; mixing the two without labelling which is
which (as Figure 3 currently risks, since only the left panel's title
specifies "Training") would materially mislead a reader.

### Figure 5/6 — Phase 3 world preset x drift condition (DQN)

| Metric | Hardcoded (with drifts) | Recomputed (with drifts, mean of 3 seeds) | Hardcoded (no drifts) | Recomputed (no drifts) |
|---|---|---|---|---|
| DQN easy reward | 5303 | **5302.80** (match) | 4574 | **NO DATA** |
| DQN easy biomass | 497 | 497.39 | 425 | **NO DATA** |
| DQN easy coverage % | 100 | 100.00 | 99.4 | **NO DATA** |
| DQN medium reward | 5133 | **5132.95** (match) | 4806 | **NO DATA** |
| DQN medium biomass | 487 | 487.34 | 459 | **NO DATA** |
| DQN medium coverage % | 100 | 100.00 | 100 | **NO DATA** |
| DQN hard reward | 5221 | 4958.39 (no match — finding #13) | 4958 | **NO DATA** |
| DQN hard biomass | 497 | 468.86 | 469 | **NO DATA** |
| DQN hard coverage % | 100 | 99.90 | 99.9 | **NO DATA** |
| Fixed SP easy reward | 3898 | 4088.03 (seeds 100-119) | 4023 | 4456.32 (seeds 100-119, freshly run — agent is frozen/deterministic so this IS reproducible) |
| Fixed SP medium reward | 4075 | 4291.89 | 4229 | 4681.33 |
| Fixed SP hard reward | 3949 | 4172.25 | 4075 | 4582.11 |

**No stored eval CSV exists anywhere for a `no_drifts=True` run of DQN on
seeds 42/43/44 for any of easy/medium/hard** — see finding #12. Every
"no drifts" DQN number in Figures 5 and 6 (`4574/4806/4958` reward,
`425/459/469` biomass, `99.4/100/99.9` coverage) is **unverifiable from
anything currently in `results/`**. It is possible these numbers come from
real no-drift training runs that were done once and then silently
overwritten (finding #12), or they could be estimates — there is no way to
tell which from what's on disk today. The Fixed-Setpoint baseline *is*
re-runnable under `no_drifts=True` right now (it is a frozen, deterministic
policy) and its freshly-computed numbers do not match the hardcoded ones
particularly closely either (e.g. hard: hardcoded 4075 vs freshly-run
4582.11), reinforcing that these figures' numbers were not computed from the
current codebase+seed protocol.

## Step 2g — pipeline reproducibility check

See finding #14 above for the full result. Summary: loaded
`trained_models/dqn_easy_multi_objective_999/model.zip` (the only saved
model old enough, and with a matching stored `*_eval.csv`, to test
end-to-end), re-ran `training/evaluate.py` into a temp directory on seeds
100-119 under both `no_drifts=True` and `no_drifts=False`, and diffed
against the stored `results/dqn_easy_multi_objective_999_eval.csv`. **Neither
reproduces the stored file** (stored file shows 5 extinctions out of 20
episodes and rewards up to 2559; both fresh runs show 0 extinctions and a
much narrower, lower reward range). Timestamps on disk show the model and
its results CSVs are over a month apart in modification time, which is
anomalous compared to every main-experiment run (where training and
evaluation CSVs are one day apart). **No model was retrained to investigate
this further**, per the hard limits.

## Proposed v3 changes (NOT implemented)

These would require a genuinely new environment file/version, since fixing
any of them changes what a given seed computes and would invalidate every
existing result in `results/`/`trained_models/`.

1. Make temperature actually influence growth (and/or add an action that
   controls it), so `obs[1]` stops being a pure decoy.
2. Make `TOXIN_DECAY` apply to `self.local_toxin` directly (per-cell), not
   just the recomputed global observation — so local toxin can fall back on
   its own, not only via the toxin-filter action.
3. Fix the `turbidity_delta` first-step artefact, e.g. by seeding
   `prev_coverage` from the actual post-seeding coverage at `reset()` time
   instead of `0.0`.
4. Fix the `_get_obs()` single-survivor bug: change `n_living > 1` to
   `n_living >= 1` (equivalently, `living_mask.any()`) for `mean_toxin` and
   `mean_age`.
5. Add a `no_drifts` (and ideally also a short config hash) component to
   `run_name` so with-drift and no-drift runs of the same
   algo/world/reward_mode/seed can never collide or silently overwrite each
   other.
6. Run an actual hyperparameter search (or at minimum document a
   deliberately-matched config with justification) for DQN, PPO, and A2C
   rather than one untuned configuration each.
7. Pin exact library versions *before* running any experiment (now done for
   future work via `requirements.txt`), and record the installed package
   versions alongside each `config.txt`, so a reproducibility gap like
   finding #14 cannot recur silently.
