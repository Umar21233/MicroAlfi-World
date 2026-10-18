"""
train_ppo.py — Train PPO on AlphaeWorldEnvV2
==============================================
Trains a Stable-Baselines3 PPO agent (MlpPolicy) on the CA grid world,
using whichever world preset / reward mode / seed are passed on the CLI.

Identical in structure to train_dqn.py — see that file for the full
rationale. Everything about a run is recorded next to its outputs so
results can always be traced back to exactly what produced them:
  - the trained model + a config.txt (world, reward_mode, seed, timesteps,
    no_drifts) saved together in trained_models/ppo_{world}_{reward_mode}_{seed}/
  - wall-clock training time in results/ppo_{world}_{reward_mode}_{seed}_training_time.txt
  - a learning curve (mean reward every 10000 timesteps) in
    results/ppo_{world}_{reward_mode}_{seed}_learning_curve.csv

Usage (from project root):
    python training/train_ppo.py
    python training/train_ppo.py --world easy --reward_mode turbidity --seed 7
    python training/train_ppo.py --timesteps 100000 --no_drifts
"""

import sys
import os
import csv
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS
from scripts.run_naming import build_run_name
from training.hyperparameters import TUNED

ALGO_NAME = "ppo"
LOG_FREQ  = 10_000   # timesteps between learning-curve log points


# ── Learning curve callback ───────────────────────────────────────────────────

class LearningCurveCallback(BaseCallback):
    """
    Logs mean episode reward every `log_freq` timesteps during training and
    writes the full curve to a CSV once training ends.

    Mean reward is read from SB3's own rolling ep_info_buffer, which the
    base algorithm populates automatically from the "episode" key our env
    already puts in info on termination/truncation — no extra Monitor
    wrapper needed.
    """

    def __init__(self, log_freq: int, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_path = csv_path
        self.rows     = []   # list of (timesteps, mean_reward)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            else:
                mean_reward = float("nan")   # no completed episodes yet
            self.rows.append((self.num_timesteps, mean_reward))
            if self.verbose:
                print(f"  [{self.num_timesteps:>8}] mean_reward={mean_reward:.3f}")
        return True   # never stop training early

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward"])
            writer.writerows(self.rows)


# ── Training entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO on AlphaeWorldEnvV2")
    parser.add_argument("--world", type=str, default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--reward_mode", type=str, default="multi_objective",
                        choices=["multi_objective", "biomass_only", "turbidity",
                                 "turbidity_delta", "efficiency", "sustainability"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--no_drifts", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--new_naming", action="store_true",
                        help="Use the collision-free 5-part run name "
                             "{algo}_{world}_{reward_mode}_{drift}_{seed} "
                             "instead of the old 4-part name (default: off, "
                             "so existing behaviour/filenames are unchanged). "
                             "See scripts/run_naming.py.")
    parser.add_argument("--tuned", action="store_true",
                        help="Use the tuned hyperparameters confirmed from "
                             "the Colab-trained models (training/hyperparameters.py) "
                             "instead of this script's own CLI defaults. "
                             "Default: off, so existing behaviour is unchanged.")
    args = parser.parse_args()

    if args.tuned:
        tuned = TUNED[ALGO_NAME]
        args.learning_rate = tuned["learning_rate"]
        args.n_steps = tuned["n_steps"]

    run_name = build_run_name(
        ALGO_NAME, args.world, args.reward_mode, args.seed,
        no_drifts=args.no_drifts,
        style="new" if args.new_naming else "old",
    )
    model_dir   = os.path.join(PROJECT_ROOT, "trained_models", run_name)
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[args.world],
        reward_mode=args.reward_mode,
        no_drifts=args.no_drifts,
    )
    env.reset(seed=args.seed)

    model = PPO("MlpPolicy", env, seed=args.seed, gamma=0.99,
                learning_rate=args.learning_rate, n_steps=args.n_steps, verbose=1)

    callback = LearningCurveCallback(
        log_freq=LOG_FREQ,
        csv_path=os.path.join(results_dir, f"{run_name}_learning_curve.csv"),
        verbose=1,
    )

    print(f"\n{'='*65}")
    print(f"  Training PPO  |  world={args.world}  reward={args.reward_mode}  "
          f"seed={args.seed}  timesteps={args.timesteps}  no_drifts={args.no_drifts}")
    print(f"{'='*65}\n")

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    training_time = time.time() - start_time

    model.save(os.path.join(model_dir, "model"))

    # config.txt — exactly which settings produced this model, so any saved
    # model can always be traced back to how it was trained
    with open(os.path.join(model_dir, "config.txt"), "w") as f:
        f.write(f"algo={ALGO_NAME}\n")
        f.write(f"world={args.world}\n")
        f.write(f"reward_mode={args.reward_mode}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"timesteps={args.timesteps}\n")
        f.write(f"no_drifts={args.no_drifts}\n")

    with open(os.path.join(results_dir, f"{run_name}_training_time.txt"), "w") as f:
        f.write(f"{training_time:.2f}\n")

    print(f"\n{'='*65}")
    print(f"  Training complete in {training_time:.1f}s")
    print(f"  Model saved to:   {model_dir}")
    print(f"  Config saved to:  {os.path.join(model_dir, 'config.txt')}")
    print(f"  Learning curve:   results/{run_name}_learning_curve.csv")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
