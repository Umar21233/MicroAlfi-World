"""
evaluate.py — Evaluate a Trained Model on AlphaeWorldEnvV2
=============================================================
Loads a saved DQN/PPO/A2C model and evaluates it for 20 episodes using
seeds 100-119 (a disjoint block from the training seed, so evaluation
never reuses a scenario the model trained on).

--seed is the TRAINING seed, kept only so output filenames record which
model produced them (evaluation itself always uses seeds 100-119
regardless of --seed) — this is what lets a results CSV be traced back to
exactly which trained model + settings it came from.

Outputs (all under results/, named {algo}_{world}_{reward_mode}_{seed}_*):
    *_eval.csv                — one row per episode: reward, final
                                coverage/biomass, extinct flag, type_b
                                fraction, mean toxin, steps, and a count
                                of each of the 11 actions used.
    *_eval_summary.csv        — mean and std of every numeric column above,
                                across all 20 episodes.
    *_coverage_trajectory.csv — colony coverage at every step, averaged
                                (mean + std) across all 20 episodes.

Usage (from project root):
    python training/evaluate.py --model_path trained_models/dqn_hard_multi_objective_42/model --algo dqn --world hard --reward_mode multi_objective --seed 42
    python training/evaluate.py --model_path trained_models/ppo_easy_turbidity_7/model --algo ppo --world easy --reward_mode turbidity --seed 7 --no_drifts
"""

import sys
import os
import csv
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3 import DQN, PPO, A2C

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS
from scripts.run_naming import build_run_name

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

N_EPISODES = 20
SEED_START = 100   # evaluation seeds 100..119 inclusive — separate from training seed


def run_evaluation(model, env, n_episodes: int, seed_start: int, deterministic: bool = True):
    """
    Run n_episodes evaluation episodes with the trained model acting
    greedily by default (deterministic=True), collecting per-episode stats
    and the per-step coverage trajectory in a single pass.

    deterministic: passed straight through to model.predict(). Defaults to
        True, preserving every existing call site's behaviour unchanged
        (including the CLI below, which never passes this argument). Added
        so a diagnostic script can compare deterministic vs stochastic
        evaluation without altering the default evaluation protocol.

    Returns:
        episode_rows     : list of per-episode result dicts
        coverage_by_step : dict of {step: [coverage values across episodes]}
    """
    n_actions = env.action_space.n
    episode_rows     = []
    coverage_by_step = {}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        ep_reward     = 0.0
        action_counts = np.zeros(n_actions, dtype=int)
        terminated    = False
        done          = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            coverage_by_step.setdefault(info["step"], []).append(info["colony_coverage"])
            done = terminated or truncated

        row = {
            "episode":         ep + 1,
            "total_reward":    round(ep_reward, 4),
            "final_coverage":  round(float(obs[9]), 4),
            "final_biomass":   round(info["total_biomass"], 4),
            "extinct":         bool(terminated),
            "type_b_fraction": round(float(obs[11]), 4),
            "mean_toxin":      round(float(obs[10]), 4),
            "steps":           env.step_count,
        }
        for a in range(n_actions):
            row[f"action_{a}_count"] = int(action_counts[a])

        episode_rows.append(row)

    return episode_rows, coverage_by_step


def summarize_episodes(episode_rows: list) -> list:
    """Compute mean and std of every numeric column across all episodes."""
    numeric_keys = [k for k in episode_rows[0].keys() if k != "episode"]
    summary_rows = []
    for stat_name, stat_fn in (("mean", np.mean), ("std", np.std)):
        row = {"stat": stat_name}
        for key in numeric_keys:
            values = [float(r[key]) for r in episode_rows]
            row[key] = round(float(stat_fn(values)), 4)
        summary_rows.append(row)
    return summary_rows


def write_csv(rows: list, path: str):
    """Write a list of same-shaped dicts to a CSV with a header row."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_trajectory_csv(coverage_by_step: dict, path: str):
    """Write step, mean_coverage, std_coverage — one row per step reached."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_coverage", "std_coverage"])
        for step in sorted(coverage_by_step.keys()):
            values = coverage_by_step[step]
            writer.writerow([
                step,
                round(float(np.mean(values)), 4),
                round(float(np.std(values)), 4),
            ])


def print_summary(episode_rows: list, run_name: str):
    """Print a clean terminal summary of the evaluation."""
    rewards     = [r["total_reward"]    for r in episode_rows]
    coverages   = [r["final_coverage"]  for r in episode_rows]
    biomasses   = [r["final_biomass"]   for r in episode_rows]
    extinctions = sum(r["extinct"]      for r in episode_rows)
    type_b_fracs = [r["type_b_fraction"] for r in episode_rows]
    mean_toxins  = [r["mean_toxin"]      for r in episode_rows]

    print(f"\n{'='*65}")
    print(f"  Evaluation — {run_name}  ({len(episode_rows)} episodes, seeds "
          f"{SEED_START}-{SEED_START + len(episode_rows) - 1})")
    print(f"{'='*65}")
    print(f"  Mean reward:        {np.mean(rewards):>10.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean final coverage:{np.mean(coverages)*100:>9.1f}% "
          f"± {np.std(coverages)*100:.1f}%")
    print(f"  Mean final biomass: {np.mean(biomasses):>10.3f} ± {np.std(biomasses):.3f}")
    print(f"  Extinctions:        {extinctions}/{len(episode_rows)}")
    print(f"  Mean Type B frac.:  {np.mean(type_b_fracs)*100:>9.1f}%")
    print(f"  Mean toxin:         {np.mean(mean_toxins):>10.3f}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on AlphaeWorldEnvV2")
    parser.add_argument("--model_path", type=str, required=True,
                        help="path to the saved SB3 model (e.g. trained_models/dqn_hard_multi_objective_42/model)")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo", "a2c"])
    parser.add_argument("--world", type=str, default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--reward_mode", type=str, default="multi_objective",
                        choices=["multi_objective", "biomass_only", "turbidity",
                                 "turbidity_delta", "efficiency", "sustainability"])
    parser.add_argument("--seed", type=int, default=42,
                        help="the TRAINING seed — used only for output file naming, "
                             "evaluation itself always uses seeds 100-119")
    parser.add_argument("--no_drifts", action="store_true")
    parser.add_argument("--new_naming", action="store_true",
                        help="Use the collision-free 5-part run name "
                             "{algo}_{world}_{reward_mode}_{drift}_{seed} "
                             "instead of the old 4-part name (default: off, "
                             "so existing behaviour/filenames are unchanged). "
                             "See scripts/run_naming.py.")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_name = build_run_name(
        args.algo, args.world, args.reward_mode, args.seed,
        no_drifts=args.no_drifts,
        style="new" if args.new_naming else "old",
    )

    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[args.world],
        reward_mode=args.reward_mode,
        no_drifts=args.no_drifts,
    )

    AlgoClass = ALGO_CLASSES[args.algo]
    model = AlgoClass.load(args.model_path)

    episode_rows, coverage_by_step = run_evaluation(model, env, N_EPISODES, SEED_START)
    summary_rows = summarize_episodes(episode_rows)

    write_csv(episode_rows, os.path.join(results_dir, f"{run_name}_eval.csv"))
    write_csv(summary_rows, os.path.join(results_dir, f"{run_name}_eval_summary.csv"))
    write_coverage_trajectory_csv(
        coverage_by_step, os.path.join(results_dir, f"{run_name}_coverage_trajectory.csv")
    )

    print_summary(episode_rows, run_name)
    print(f"  Saved: results/{run_name}_eval.csv")
    print(f"  Saved: results/{run_name}_eval_summary.csv")
    print(f"  Saved: results/{run_name}_coverage_trajectory.csv\n")

    env.close()


if __name__ == "__main__":
    main()
