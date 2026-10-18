"""
train_ablation.py -- Drift study Part B.3. NOT RUN in this session -- see
the hard limit in the task that created this file. Running this script
with no arguments only prints the command list and time estimate below;
it never trains anything itself unless given --drift and --seed, which is
left for Colab.

Trains one DQN model on AblationDriftEnv (ablation_env.py) with exactly
one of the six passive drifts switched on, everything else off -- the
"only X drift on" condition -- using the --tuned hyperparameters
(training/hyperparameters.py: learning_rate=0.001, batch_size=64,
gamma=0.99), 500,000 timesteps, hard world, multi_objective reward mode.
Run name: dqn_hard_multi_objective_only-{drift}_{seed}.

Six drifts x three seeds (42, 43, 44) = 18 runs. Outputs go to
results_verified/drift_study/ablation/<run_name>/ (model.zip, config.txt,
learning_curve.csv, training_time.txt) -- a new subfolder, nothing in
results_verified/ is overwritten.

Temperature has no effect on growth anywhere in _ca_step (see
envs/alphae_world_env_v2.py's own note on TEMP_DRIFT_RATE), so the
only-temperature condition is expected to behave like no-drifts-at-all
despite its own passive drift being switched on. It is still run, because
the point of running it is to confirm that expectation empirically rather
than assume it.

Usage:
    python experiments/drift_study/train_ablation.py
        -- prints the 18 Colab-ready commands and the time estimate; does
           not train anything.
    python experiments/drift_study/train_ablation.py --drift co2 --seed 42
        -- actually runs one training run (not invoked in this session).
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.hyperparameters import TUNED  # noqa: E402
from experiments.drift_study.ablation_env import DRIFT_NAMES, ALL_OFF  # noqa: E402

ALGO_NAME = "dqn"
LOG_FREQ = 10_000
SEEDS = [42, 43, 44]
ESTIMATED_MINUTES_PER_RUN = 17  # per the task's own Colab-CPU estimate


class LearningCurveCallback(BaseCallback):
    """Logs mean reward over a short rollout every LOG_FREQ timesteps,
    identical in spirit to training/train_dqn.py's own callback."""

    def __init__(self, log_freq: int, csv_path: str, env, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_path = csv_path
        self.eval_env = env
        self.rows = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            rewards = []
            for _ in range(3):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(int(action))
                    ep_reward += reward
                    done = terminated or truncated
                rewards.append(ep_reward)
            mean_reward = sum(rewards) / len(rewards)
            self.rows.append({"timesteps": self.num_timesteps, "mean_reward": mean_reward})
            if self.verbose:
                print(f"  [{self.num_timesteps:>7}] mean_reward={mean_reward:.3f}")
        return True

    def _on_training_end(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward"])
            writer.writeheader()
            writer.writerows(self.rows)


def run_name_for(drift: str, seed: int) -> str:
    return f"{ALGO_NAME}_hard_multi_objective_only-{drift}_{seed}"


def print_commands_and_estimate():
    print("Ablation training -- NOT started. Colab-ready command list "
          f"({len(DRIFT_NAMES)} drifts x {len(SEEDS)} seeds = "
          f"{len(DRIFT_NAMES) * len(SEEDS)} runs):\n")
    for drift in DRIFT_NAMES:
        for seed in SEEDS:
            print(f"  python experiments/drift_study/train_ablation.py "
                  f"--drift {drift} --seed {seed}")
    total_runs = len(DRIFT_NAMES) * len(SEEDS)
    total_minutes = total_runs * ESTIMATED_MINUTES_PER_RUN
    print(f"\nTime estimate: ~{ESTIMATED_MINUTES_PER_RUN} minutes per run on Colab CPU "
          f"x {total_runs} runs = ~{total_minutes} minutes "
          f"(~{total_minutes / 60:.1f} hours) if run sequentially; "
          f"run several Colab sessions in parallel to shorten this.")
    print("\nNote: the only-temperature condition is expected (not assumed) to "
          "behave like no-drifts-at-all, since temperature has no effect on "
          "growth anywhere in the environment -- kept in the matrix to confirm "
          "that empirically rather than skip it.")


def main():
    parser = argparse.ArgumentParser(description="Train DQN on AblationDriftEnv (one drift on)")
    parser.add_argument("--drift", type=str, choices=DRIFT_NAMES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    if args.drift is None or args.seed is None:
        print_commands_and_estimate()
        return

    # Local import so AblationDriftEnv is only instantiated for an actual run.
    from experiments.drift_study.ablation_env import AblationDriftEnv

    drift_flags = dict(ALL_OFF)
    drift_flags[args.drift] = True

    run_name = run_name_for(args.drift, args.seed)
    out_dir = PROJECT_ROOT / "results_verified" / "drift_study" / "ablation" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned = TUNED[ALGO_NAME]

    env = AblationDriftEnv(
        drift_flags=drift_flags,
        world_config=WORLD_PRESETS["hard"],
        reward_mode="multi_objective",
    )
    env.reset(seed=args.seed)

    eval_env = AblationDriftEnv(
        drift_flags=drift_flags,
        world_config=WORLD_PRESETS["hard"],
        reward_mode="multi_objective",
    )

    model = DQN("MlpPolicy", env, seed=args.seed, gamma=tuned["gamma"],
                learning_rate=tuned["learning_rate"], batch_size=tuned["batch_size"],
                verbose=1)

    callback = LearningCurveCallback(
        log_freq=LOG_FREQ,
        csv_path=str(out_dir / "learning_curve.csv"),
        env=eval_env,
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    elapsed = time.time() - start

    model.save(str(out_dir / "model"))
    with open(out_dir / "config.txt", "w") as f:
        f.write(f"algo={ALGO_NAME}\n")
        f.write("world=hard\n")
        f.write("reward_mode=multi_objective\n")
        f.write(f"ablation_drift={args.drift}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"timesteps={args.timesteps}\n")
        f.write(f"tuned_learning_rate={tuned['learning_rate']}\n")
        f.write(f"tuned_batch_size={tuned['batch_size']}\n")
        f.write(f"tuned_gamma={tuned['gamma']}\n")
    with open(out_dir / "training_time.txt", "w") as f:
        f.write(f"{elapsed:.2f} seconds\n")

    print(f"\nSaved {run_name} -> {out_dir}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
