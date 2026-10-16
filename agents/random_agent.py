"""
random_agent.py — Baseline 1: Pure Random
==========================================
Runs a completely random agent on AlphaeWorldEnv and reports
episode statistics. This is the floor baseline — any agent
with even a basic strategy should score higher than this.

Why do we need a random baseline?
  It tells us the minimum expected reward in this environment.
  If an RL agent ever scores below this, training has gone wrong.
  It also sanity-checks that the environment runs without errors.

Usage (from project root):
    python agents/random_agent.py
"""

import sys
import os

# Add the project root to Python's path so we can import from envs/
# Without this, Python won't find the envs package when running this
# script directly from the agents/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.alphae_world_env import AlphaeWorldEnv


def run_random_episodes(n_episodes: int = 10, max_steps: int = 200, seed: int = 42):
    """
    Run a random agent for n_episodes episodes and print a summary.

    The random agent calls env.action_space.sample() at every step,
    which returns a uniformly random integer from 0 to 6. It never
    looks at the observation — it is completely state-blind.

    Args:
        n_episodes : number of full episodes to run
        max_steps  : maximum steps per episode (passed to env)
        seed       : base random seed — each episode uses seed + episode_index
                     so results are reproducible but each episode differs

    Returns:
        list of result dicts, one per episode
    """
    env = AlphaeWorldEnv(max_steps=max_steps)
    results = []

    print(f"\n{'='*55}")
    print(f"  AlphaeWorld — Random Agent Baseline ({n_episodes} episodes)")
    print(f"{'='*55}")

    for ep in range(n_episodes):
        # Reset the environment at the start of each episode.
        # seed + ep ensures each episode starts from a different (but
        # reproducible) random state — important for fair comparison
        # against the fixed setpoint agent later.
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # The entire "policy" of the random agent: sample uniformly.
            # env.action_space.sample() returns an int in {0, 1, 2, 3, 4, 5, 6}.
            # The observation `obs` is deliberately ignored here.
            action = env.action_space.sample()

            # Step the environment and collect feedback
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # terminated: biomass hit zero (culture crashed)
            # truncated:  episode hit max_steps time limit
            done = terminated or truncated

        # obs[5] is biomass — the last element of the state vector
        final_biomass = float(obs[5])

        results.append({
            "episode":       ep + 1,
            "total_reward":  round(ep_reward, 3),
            "steps":         env.step_count,
            "final_biomass": round(final_biomass, 4),
            "crashed":       bool(terminated),   # True if culture crashed
        })

        print(f"  Ep {ep+1:>2}: reward={ep_reward:>8.3f}  "
              f"biomass={final_biomass:.4f}  "
              f"steps={env.step_count}  "
              f"{'💀 CRASH' if terminated else '✓'}")

    env.close()

    # ── Summary statistics ───────────────────────────────────────────────────
    # These numbers become the "floor" in your FYP results table.
    # Any trained RL agent should comfortably exceed mean_reward and
    # mean_biomass while having fewer crashes.
    rewards   = [r["total_reward"]  for r in results]
    biomasses = [r["final_biomass"] for r in results]
    crashes   = sum(r["crashed"] for r in results)

    print(f"\n{'-'*55}")
    print(f"  Mean reward:       {np.mean(rewards):>8.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean final biomass:{np.mean(biomasses):>8.4f} ± {np.std(biomasses):.4f}")
    print(f"  Crashes:           {crashes}/{n_episodes}")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    run_random_episodes(n_episodes=10)