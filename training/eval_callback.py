"""
eval_callback.py — Shared periodic held-out evaluation callback
====================================================================
PeriodicEvalCallback is used by train_dqn.py, train_ppo.py and train_a2c.py
behind their --eval_curve flag. Kept in one place so all three algorithms
log the eval curve the same way.
"""

import os
import csv

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.evaluate import run_evaluation, N_EPISODES, SEED_START


class PeriodicEvalCallback(BaseCallback):
    """
    Every `eval_freq` timesteps, pauses training and runs a full held-out
    evaluation (seeds 100-119, deterministic) of the CURRENT model on a
    separate eval env, using the same run_evaluation() as training/evaluate.py.

    This is a cleaner training-time signal than LearningCurveCallback's
    mean training reward, since it is not polluted by exploration noise.
    Steps taken by eval_env never count toward total training steps.

    The CSV is written after every evaluation (not just at the end) so a
    disconnect mid-training does not lose already-collected rows.
    """

    def __init__(self, eval_freq: int, eval_env, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env  = eval_env
        self.csv_path  = csv_path
        self.header_written = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            episode_rows, _ = run_evaluation(
                self.model, self.eval_env, N_EPISODES, SEED_START, deterministic=True
            )
            rewards      = [r["total_reward"]  for r in episode_rows]
            biomasses    = [r["final_biomass"] for r in episode_rows]
            action_costs = [r["action_cost"]   for r in episode_rows]
            extinctions  = sum(r["extinct"] for r in episode_rows)

            row = {
                "timesteps":        self.num_timesteps,
                "mean_reward":      round(float(np.mean(rewards)), 4),
                "std_reward":       round(float(np.std(rewards)), 4),
                "mean_biomass":     round(float(np.mean(biomasses)), 4),
                "extinction_rate":  round(extinctions / len(episode_rows), 4),
                "mean_action_cost": round(float(np.mean(action_costs)), 4),
            }
            self._append_row(row)
            if self.verbose:
                print(f"  [eval @ {self.num_timesteps:>8}] mean_reward={row['mean_reward']:.3f} "
                      f"extinction_rate={row['extinction_rate']:.2f}")
        return True   # never stop training early

    def _append_row(self, row: dict) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        write_header = not self.header_written and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.header_written = True
