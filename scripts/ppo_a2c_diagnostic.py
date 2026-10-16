"""
ppo_a2c_diagnostic.py -- Step 5 of the verified-data rebuild (REPORT ONLY).

Training-reward at the last learning-curve point is far above the stored
eval reward for PPO (~5017 vs 2836) and A2C (~3095 vs 1180). This script
evaluates every PPO and A2C model (plus DQN for comparison) on the same
protocol (seeds 100-119, 20 episodes) with BOTH deterministic=True and
deterministic=False, using the deterministic= argument added to
training/evaluate.py's run_evaluation() (default True, so evaluate.py's
own default behaviour is unchanged).

Does NOT alter evaluate.py's default and does NOT touch any stored/
published eval numbers -- purely diagnostic, written to
results_verified/diagnostics/ppo_a2c_deterministic_vs_stochastic.csv.

Usage:
    python scripts/ppo_a2c_diagnostic.py
"""

import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from stable_baselines3 import DQN, PPO, A2C  # noqa: E402

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.evaluate import run_evaluation, N_EPISODES, SEED_START  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results_verified" / "diagnostics"
RESULTS_VERIFIED = PROJECT_ROOT / "results_verified"
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


def read_config(run_dir: Path) -> dict:
    config = {}
    for line in (run_dir / "config.txt").read_text().strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    return config


def load_model_and_env(run_dir: Path, config: dict):
    algo = config["algo"]
    world = config["world"]
    reward_mode = config["reward_mode"]
    no_drifts = config["no_drifts"] == "True"
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world], reward_mode=reward_mode, no_drifts=no_drifts,
    )
    model_bytes = (run_dir / "model.zip").read_bytes()
    model = ALGO_CLASSES[algo].load(io.BytesIO(model_bytes))
    return model, env


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for phase_dir in sorted(RESULTS_VERIFIED.iterdir()):
        if not phase_dir.is_dir() or phase_dir.name not in ("phase1", "phase2"):
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "config.txt").exists():
                continue
            config = read_config(run_dir)
            if config["algo"] not in ("dqn", "ppo", "a2c"):
                continue
            if config["world"] != "hard" or config["reward_mode"] != "multi_objective":
                continue  # Phase 1's three algorithms, hard/multi_objective/drifts only

            for det in (True, False):
                model, env = load_model_and_env(run_dir, config)
                episode_rows, _ = run_evaluation(model, env, N_EPISODES, SEED_START, deterministic=det)
                env.close()
                rewards = [r["total_reward"] for r in episode_rows]
                rows.append({
                    "algo": config["algo"], "seed": config["seed"],
                    "deterministic": det,
                    "mean_reward": round(float(np.mean(rewards)), 3),
                    "std_reward": round(float(np.std(rewards)), 3),
                    "min_reward": round(float(np.min(rewards)), 3),
                    "max_reward": round(float(np.max(rewards)), 3),
                })
                print(f"{config['algo']} seed={config['seed']} deterministic={det}: "
                      f"mean={np.mean(rewards):.2f} std={np.std(rewards):.2f}")

    out_path = OUT_DIR / "ppo_a2c_deterministic_vs_stochastic.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "deterministic",
                                                "mean_reward", "std_reward",
                                                "min_reward", "max_reward"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")

    print("\n=== Summary: mean across 3 seeds, deterministic vs stochastic ===")
    for algo in ("dqn", "ppo", "a2c"):
        for det in (True, False):
            vals = [r["mean_reward"] for r in rows if r["algo"] == algo and r["deterministic"] == det]
            if vals:
                print(f"  {algo:4s} deterministic={det!s:5s}: mean_reward={np.mean(vals):8.2f}")


if __name__ == "__main__":
    main()
