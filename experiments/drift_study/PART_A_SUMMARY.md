# Drift study -- Part A: cross-evaluation summary

Open question: why does DQN score higher WITH passive drifts than WITHOUT
(easy 5303 vs 4574, medium 5133 vs 4806, hard 5221 vs 4958 -- the headline
numbers already in `results_verified/EVAL_SUMMARY_ALL.csv`)? This is a
cross-evaluation study, not a retraining study: all 18 models already in
`results_verified/` are evaluated in both environments, nothing is
retrained. See `cross_eval.py` and `mechanism_diagnostics.py` for the
exact protocol (20 episodes, seeds 100-119, deterministic -- unchanged
from `training/evaluate.py`).

## Sanity check

All 6 diagonal cells (trained_with == evaluated_in) reproduce
`results_verified/EVAL_SUMMARY_ALL.csv` exactly: diff = 0.0000 reward in
every case (well inside the 0.5 tolerance). `CROSS_EVAL.csv` and
`CROSS_EVAL_SUMMARY.csv` can be trusted as an extension of the existing
verified dataset, not a separate re-derivation of it.

## 2x2 tables: mean reward, trained_with (rows) x evaluated_in (columns)

Mean over 3 seeds x 20 episodes = 60 episodes per cell. Fixed-setpoint
baseline (same protocol, from `results_verified/baselines/BASELINES_SUMMARY.csv`)
given for reference, not retrained.

### Easy world

| trained_with \ evaluated_in | drifts | nodrifts |
|---|---|---|
| drifts | **5302.80** | 4815.19 |
| nodrifts | 3525.05 | **4574.02** |

Fixed setpoint (reference): drifts = 4088.03, nodrifts = 4456.32

### Medium world

| trained_with \ evaluated_in | drifts | nodrifts |
|---|---|---|
| drifts | **5132.95** | 5091.08 |
| nodrifts | 3970.94 | **4806.42** |

Fixed setpoint (reference): drifts = 4291.89, nodrifts = 4681.33

### Hard world

| trained_with \ evaluated_in | drifts | nodrifts |
|---|---|---|
| drifts | **5220.89** | 5141.32 |
| nodrifts | 4452.41 | **4958.39** |

Fixed setpoint (reference): drifts = 4172.25, nodrifts = 4582.11

(Bold = the diagonal, i.e. each model evaluated in the environment it was
trained in -- these are the numbers already published.)

## Plain reading, against the three keys given

**"Both policies score higher in the drift environment: environment
effect."** Not supported. Only the drift-trained policy scores higher in
the drifts column than the nodrifts column, and even then the gap is
small in medium (+41.9) and hard (+79.6), larger in easy (+487.6). The
nodrift-trained policy goes the *other* way in every world: it scores
lower in drifts than in nodrifts (easy -1049.0, medium -835.5, hard
-506.0). A pure "the drift environment happens to be easier for anyone"
explanation is ruled out by this data: it is not easier for the
nodrift-trained policy.

**"Drift-trained policy better in both environments: training effect."**
Supported, in all three worlds, with no exception. Comparing the two
policies within the same evaluation environment: in the drifts column,
drift-trained beats nodrift-trained by 1777.7 (easy), 1162.0 (medium),
768.5 (hard). In the nodrifts column -- the nodrift-trained policy's own
home environment -- drift-trained *still* wins, by 241.2 (easy), 284.7
(medium), 182.9 (hard). Training with drifts produces a DQN policy that
is uniformly better than training without drifts, regardless of which
environment it is then asked to act in. This is the clearest, most
consistent pattern in the data, and it directly answers the open
question's framing: the headline numbers are not "drifts make the task
easier", they are "training with drifts produces a better policy."

**"No-drift policy collapses under drifts: overfitting to static
conditions."** Partially supported, with an important qualification.
There is a real, consistent reward drop for the nodrift-trained policy
moved into the drifts environment (10 to 23 percent across the three
worlds, see above), and in easy and medium worlds that drop is severe
enough that the nodrift-trained DQN (3525.05, 3970.94) actually scores
*below* the simple, non-learning fixed-setpoint controller evaluated in
the same drifting environment (4088.03, 4291.89) -- a learned policy
losing to a hand-coded one is a meaningful signal of brittleness. But
"collapse" is too strong a word for what the data shows: coverage stays
at or above 98.9 percent and there are zero extinctions in every one of
the 36 (world x trained_with x evaluated_in) cells of this cross-evaluation
(each 3 seeds x 20 episodes). The reward drop is real; the colony does
not die. There is also substantial seed-to-seed variance within this
condition that a 3-seed mean hides: easy/nodrift-trained's three seeds
score 3001.85, 4249.14 and 3324.15 reward when moved into drifts -- not a
uniform failure, one seed degrades far more than the other two.

## Mechanism diagnostics (hard world, seed 42 only -- one model per
condition, see the caveat below)

From `MECHANISM_TIMESERIES.csv` and `MECHANISM_ACTION_DISTRIBUTION.csv`
(plots: `plots_verified/drift_study/fig_mechanism_timeseries.png`,
`fig_mechanism_action_distribution.png`):

- All four (policy, environment) combinations survive all 20 episodes to
  the full 200 steps -- zero extinctions here either, consistent with
  the coverage finding above.
- The drift-trained policy maintains a visibly higher mean nutrients
  level across the whole episode (0.917 in drifts, 0.963 in nodrifts)
  than the nodrift-trained policy (0.823 in drifts, 0.856 in nodrifts).
  Toxin is correspondingly a little higher for the drift-trained policy
  (0.163/0.178 vs 0.123/0.139), consistent with sustaining more active
  growth (toxin is a growth byproduct in this environment).
- The action distributions explain why: the drift-trained policy spends
  32 to 44 percent of its steps on action 5 ("add nutrients"), versus 21
  to 22 percent for the nodrift-trained policy, and almost never uses
  action 1 ("increase light": under 1 percent, versus 8 to 10 percent
  for the nodrift-trained policy) or action 4 ("decrease co2": 0 to 1.7
  percent, versus 8.8 to 10 percent). The drift-trained policy looks like
  it learned a more concentrated nutrient-and-toxin-management strategy;
  the nodrift-trained policy spreads more of its action budget across
  smaller light/CO2 adjustments.

**What this does and does not show.** This is a plausible behavioural
account of *how* the training-effect gap shows up in the policies'
actions, for one seed in one world. It is not evidence of *why* training
under drift produces this strategy (that would need the ablation study in
Part B, which was prepared but not run). It should not be generalised to
all seeds or worlds from a single-seed diagnostic -- the 2x2 tables above,
not this diagnostic, are the three-seed-averaged evidence; this diagnostic
is illustrative of one concrete instance of the pattern, nothing more.

## What the numbers do not show

- They do not show *why* drift-trained DQN maintains nutrients better --
  only one candidate mechanism (its action distribution), from one seed,
  in one world.
- They do not show that drifts are intrinsically "good for training" in
  general -- this is one environment, one algorithm (DQN), one reward
  mode (multi_objective), and three seeds per cell.
- They do not rule out that the training-effect gap is partly an
  artefact of DQN's own optimisation dynamics (e.g. richer gradient
  signal from having to correct degrading conditions every episode,
  versus a more static, less demanding training signal with no_drifts)
  rather than a property of the drift *environment* itself -- the
  ablation study in Part B is designed to start separating "drift makes
  training harder in a way that helps DQN learn" from "something about
  this specific drift combination happens to interact well with DQN",
  but that study was only prepared, not run.
- Seed-to-seed variance is non-trivial, particularly for the
  nodrift-trained policy in the easy world; three seeds per cell is not
  enough to rule out that some of the training-effect gap is noise,
  though the gap's consistent direction across all three worlds and both
  evaluation environments (6 out of 6 comparisons, see above) makes pure
  chance an unlikely sole explanation.
