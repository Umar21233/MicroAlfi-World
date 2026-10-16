"""
run_baselines_verified.py -- Step 4 of the verified-data rebuild.

Runs the frozen fixed-setpoint agent (agents/fixed_setpoint_agent_v2.py)
and the frozen random agent (agents/random_agent_v2.py) using the EXACT
same evaluation protocol as training/evaluate.py (20 episodes, seeds
100-119) for:
    - hard, drifts        (the Phase 1 setting)
    - easy/medium/hard x drifts/nodrifts   (the 6 Phase 3 configurations)

Neither baseline agent is "trained" -- the fixed-setpoint policy is a frozen
rule-based function and the random agent samples uniformly -- so re-running
them does not touch the "no retrain" rule.

Writes results_verified/baselines/<agent>_<world>_multi_objective_<drift>.csv
(episode-level) and a summary table to
results_verified/baselines/BASELINES_SUMMARY.csv.

Usage:
    python scripts/run_baselines_verified.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv  # noqa: E402
import numpy as np  # noqa: E402

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from agents.fixed_setpoint_agent_v2 import fixed_setpoint_policy_v2  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results_verified" / "baselines"
N_EPISODES = 20
SEED_START = 100   # identical to training/evaluate.py's protocol

CONFIGS = [
    ("hard", False),   # Phase 1 setting
    ("easy", False), ("easy", True),
    ("medium", False), ("medium", True),
    ("hard", True),
]


def run_baseline(agent: str, world: str, no_drifts: bool):
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world],
        reward_mode="multi_objective",
        no_drifts=no_drifts,
    )
    rows = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=SEED_START + ep)
        ep_reward = 0.0
        terminated = False
        done = False
        while not done:
            if agent == "fixed_setpoint":
                action = fixed_setpoint_policy_v2(obs)
            elif agent == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(agent)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rows.append({
            "episode": ep + 1,
            "total_reward": round(float(ep_reward), 4),
            "final_coverage": round(float(obs[9]), 4),
            "final_biomass": round(info["total_biomass"], 4),
            "extinct": bool(terminated),
        })
    env.close()
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for agent in ("fixed_setpoint", "random"):
        for world, no_drifts in CONFIGS:
            drift = "nodrifts" if no_drifts else "drifts"
            rows = run_baseline(agent, world, no_drifts)
            name = f"{agent}_{world}_multi_objective_{drift}"
            path = OUT_DIR / f"{name}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            rewards = [r["total_reward"] for r in rows]
            biomasses = [r["final_biomass"] for r in rows]
            coverages = [r["final_coverage"] for r in rows]
            extinctions = sum(r["extinct"] for r in rows)
            summary_rows.append({
                "agent": agent, "world": world, "drift": drift,
                "mean_reward": round(float(np.mean(rewards)), 2),
                "std_reward": round(float(np.std(rewards)), 2),
                "mean_biomass": round(float(np.mean(biomasses)), 2),
                "mean_coverage_pct": round(float(np.mean(coverages)) * 100, 2),
                "extinctions": f"{extinctions}/{N_EPISODES}",
            })
            print(f"{name}: mean_reward={np.mean(rewards):.2f} "
                  f"mean_biomass={np.mean(biomasses):.2f} "
                  f"mean_coverage={np.mean(coverages)*100:.1f}% "
                  f"extinctions={extinctions}/{N_EPISODES}")

    summary_path = OUT_DIR / "BASELINES_SUMMARY.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
