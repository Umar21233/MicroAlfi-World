"""
snapshot_baseline.py -- Audit safety net: freeze a behavioural snapshot of
AlphaeWorldEnvV2 + fixed_setpoint_policy_v2 BEFORE any audit edits are made.

Runs the frozen fixed-setpoint policy for seeds 42..61 (20 seeds) over every
combination of:
    - 3 world presets   (easy, medium, hard)
    - 6 reward modes    (multi_objective, biomass_only, turbidity,
                         turbidity_delta, efficiency, sustainability)
    - 2 drift conditions (no_drifts True / False)

= 36 configurations x 20 seeds = 720 episodes, each recording:
    - total_reward      (sum of per-step rewards for the episode)
    - final_coverage    (colony_coverage, i.e. obs[9], at the last step)
    - final_total_biomass (env.biomass.sum() at the last step)

Writes the result to tests/baseline_snapshot.json.

This script must be run exactly once, BEFORE any edit to envs/, agents/, or
training/ -- that is what makes it a trustworthy "before" snapshot.
tests/test_regression.py re-runs the identical loop and asserts exact
equality against this file after every subsequent change.

Usage (from project root):
    python tests/snapshot_baseline.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS
from agents.fixed_setpoint_agent_v2 import fixed_setpoint_policy_v2

WORLD_PRESETS_LIST = ["easy", "medium", "hard"]
REWARD_MODES = ["multi_objective", "biomass_only", "turbidity",
                "turbidity_delta", "efficiency", "sustainability"]
NO_DRIFTS_VALUES = [False, True]
SEEDS = list(range(42, 62))   # 42..61 inclusive

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "baseline_snapshot.json")


def run_one_episode(world_preset: str, reward_mode: str, no_drifts: bool, seed: int) -> dict:
    """Run a single episode of the frozen fixed-setpoint policy and record
    the three tracked quantities exactly."""
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world_preset],
        reward_mode=reward_mode,
        no_drifts=no_drifts,
    )
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = fixed_setpoint_policy_v2(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    final_coverage = float(obs[9])
    final_total_biomass = float(env.biomass.sum())
    env.close()
    return {
        "total_reward": float(total_reward),
        "final_coverage": final_coverage,
        "final_total_biomass": final_total_biomass,
    }


def build_snapshot() -> dict:
    snapshot = {}
    n_total = len(WORLD_PRESETS_LIST) * len(REWARD_MODES) * len(NO_DRIFTS_VALUES) * len(SEEDS)
    n_done = 0
    for world_preset in WORLD_PRESETS_LIST:
        for reward_mode in REWARD_MODES:
            for no_drifts in NO_DRIFTS_VALUES:
                key = f"{world_preset}|{reward_mode}|{no_drifts}"
                snapshot[key] = []
                for seed in SEEDS:
                    result = run_one_episode(world_preset, reward_mode, no_drifts, seed)
                    snapshot[key].append(result)
                    n_done += 1
                print(f"  [{n_done:>4}/{n_total}] {key}")
    return snapshot


if __name__ == "__main__":
    print(f"Running baseline snapshot: {len(WORLD_PRESETS_LIST)} presets x "
          f"{len(REWARD_MODES)} reward modes x {len(NO_DRIFTS_VALUES)} drift "
          f"conditions x {len(SEEDS)} seeds = "
          f"{len(WORLD_PRESETS_LIST)*len(REWARD_MODES)*len(NO_DRIFTS_VALUES)*len(SEEDS)} episodes...")
    snapshot = build_snapshot()
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"\nSnapshot written to {SNAPSHOT_PATH}")
