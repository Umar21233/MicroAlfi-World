"""
evaluate_ablation.py -- Drift study Part B.3 (evaluation half). NOT RUN in
this session -- there is nothing to evaluate yet, since train_ablation.py
has not been run (see that file and the task's hard limits).

Evaluates one ablation model (trained by train_ablation.py, "only X drift
on") using the project's existing evaluation protocol unchanged
(training.evaluate.run_evaluation: 20 episodes, seeds 100-119,
deterministic=True), in the SAME single-drift-on environment it was
trained in. Produces exactly the same three-file CSV format as
training/evaluate.py (*_eval.csv, *_eval_summary.csv,
*_coverage_trajectory.csv), so results_verified/drift_study/ablation/ can
be read by the same downstream tooling (build_eval_summary_all.py-style
aggregation) as the rest of results_verified/.

Usage (once train_ablation.py has actually produced a model):
    python experiments/drift_study/evaluate_ablation.py --drift co2 --seed 42
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN  # noqa: E402

from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.evaluate import (  # noqa: E402
    run_evaluation, summarize_episodes, write_csv, write_coverage_trajectory_csv,
    print_summary, N_EPISODES, SEED_START,
)
from experiments.drift_study.ablation_env import AblationDriftEnv, DRIFT_NAMES, ALL_OFF  # noqa: E402
from experiments.drift_study.train_ablation import run_name_for  # noqa: E402

ABLATION_DIR = PROJECT_ROOT / "results_verified" / "drift_study" / "ablation"


def main():
    parser = argparse.ArgumentParser(description="Evaluate one ablation DQN model")
    parser.add_argument("--drift", type=str, choices=DRIFT_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_name = run_name_for(args.drift, args.seed)
    run_dir = ABLATION_DIR / run_name
    model_path = run_dir / "model"
    if not model_path.with_suffix(".zip").exists():
        print(f"No trained model found at {model_path}.zip -- run "
              f"train_ablation.py --drift {args.drift} --seed {args.seed} first.")
        sys.exit(1)

    drift_flags = dict(ALL_OFF)
    drift_flags[args.drift] = True

    env = AblationDriftEnv(
        drift_flags=drift_flags,
        world_config=WORLD_PRESETS["hard"],
        reward_mode="multi_objective",
    )
    model = DQN.load(str(model_path))

    episode_rows, coverage_by_step = run_evaluation(model, env, N_EPISODES, SEED_START)
    summary_rows = summarize_episodes(episode_rows)

    write_csv(episode_rows, str(run_dir / "eval.csv"))
    write_csv(summary_rows, str(run_dir / "eval_summary.csv"))
    write_coverage_trajectory_csv(coverage_by_step, str(run_dir / "coverage_trajectory.csv"))

    print_summary(episode_rows, run_name)
    print(f"  Saved: {run_dir / 'eval.csv'}")
    print(f"  Saved: {run_dir / 'eval_summary.csv'}")
    print(f"  Saved: {run_dir / 'coverage_trajectory.csv'}\n")

    env.close()


if __name__ == "__main__":
    main()
