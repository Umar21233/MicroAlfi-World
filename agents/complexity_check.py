"""
complexity_check.py — World Preset Complexity Validation
===========================================================
Confirms that the three world presets (easy, medium, hard) actually form
a meaningful difficulty ladder. Uses multi_objective reward mode
consistently across ALL presets — reward numbers are only comparable across
worlds if they're computed the same way, so reward_mode is fixed rather
than left as a CLI option here.

For each preset, runs both baseline agents — the random agent from
agents/random_agent_v2.py and the fixed setpoint controller from
agents/fixed_setpoint_agent_v2.py — for 20 episodes each (seeds 42-61) and
reports:

    Agent comparison (mean across episodes, for both agents):
        final total biomass — obs[8] * 100 (mean per-cell biomass scaled by
                               the 100-cell grid == total biomass summed
                               over the grid, i.e. env.biomass.sum())
        final colony coverage — obs[9] * 100, as a percentage

    Complexity metrics (derived from the above):
        biomass gap             — fixed setpoint mean biomass minus random
                                  mean biomass. Bigger gap = domain knowledge
                                  matters more for raw yield.
        coverage gap             — fixed setpoint mean coverage minus random
                                  mean coverage. Bigger gap = domain knowledge
                                  matters more for colony survival/spread.
        performance gap          — fixed setpoint mean reward minus random
                                  mean reward. Kept alongside the biomass/
                                  coverage gaps since it's still a useful
                                  cross-check, even though biomass/coverage
                                  are more directly interpretable.
        fixed survival rate      — fraction of fixed setpoint episodes that
                                  did NOT go extinct. Should fall as presets
                                  get harder if the presets are doing their job.
        random extinction rate   — fraction of random-agent episodes that
                                  went extinct. Should rise as presets get harder.
        coverage traj. variance  — variance of colony_coverage, pooled across
                                  every step of every random-agent episode.
                                  Measures how erratic/unstable coverage is
                                  under uncontrolled (random) actions. The
                                  random agent is used for this metric rather
                                  than the fixed setpoint agent because the
                                  fixed setpoint controller actively dampens
                                  instability, which would mask the world's
                                  underlying complexity.

Runs its own episode loops (rather than calling run_random_v2 /
run_fixed_setpoint_v2 directly) so it can collect per-step coverage
trajectories alongside the per-episode stats in a single pass, and to
keep output to two clean summary tables instead of per-episode logs.

No rendering — this script is for numeric validation only.

Usage:
    python agents/complexity_check.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS
from agents.fixed_setpoint_agent_v2 import fixed_setpoint_policy_v2

N_EPISODES  = 20
SEED_START  = 42                   # seeds 42..61 inclusive (20 seeds)
REWARD_MODE = "multi_objective"    # fixed across all presets for comparability
MAX_STEPS   = 200


def _run_episodes(world_preset: str, use_fixed_setpoint: bool) -> dict:
    """
    Run N_EPISODES episodes of either the fixed setpoint controller
    (use_fixed_setpoint=True) or the random agent (False) on a given
    world preset, using REWARD_MODE throughout.

    Args:
        world_preset       : "easy", "medium", or "hard"
        use_fixed_setpoint : True for the fixed setpoint policy, False for
                              uniform random actions

    Returns:
        dict with "rewards", "extinctions", "final_biomass", "final_coverage"
        (per-episode arrays) and "coverage" (flat array of colony_coverage
        pooled across every step of every episode).
    """
    env = AlphaeWorldEnvV2(
        max_steps=MAX_STEPS,
        reward_mode=REWARD_MODE,
        world_config=WORLD_PRESETS[world_preset],
    )

    rewards          = []
    extinctions      = []
    final_biomasses  = []   # obs[8] * 100 at episode end (== total grid biomass)
    final_coverages  = []   # obs[9] * 100 at episode end (coverage as a percentage)
    coverage_history = []   # flat list of per-step coverage across all episodes

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=SEED_START + ep)
        ep_reward = 0.0
        terminated = False
        done = False

        while not done:
            if use_fixed_setpoint:
                action = fixed_setpoint_policy_v2(obs)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            coverage_history.append(info["colony_coverage"])
            done = terminated or truncated

        rewards.append(ep_reward)
        extinctions.append(terminated)
        final_biomasses.append(float(obs[8]) * 100)
        final_coverages.append(float(obs[9]) * 100)

    env.close()
    return {
        "rewards":         np.array(rewards),
        "extinctions":     np.array(extinctions),
        "final_biomass":   np.array(final_biomasses),
        "final_coverage":  np.array(final_coverages),
        "coverage":        np.array(coverage_history),
    }


def run_complexity_check() -> dict:
    """Run both agents on all three presets and print a summary table."""
    results = {}

    print(f"\n{'='*78}")
    print(f"  Microalfi World v2 — Complexity Check  "
          f"(reward_mode={REWARD_MODE}, {N_EPISODES} episodes/agent/preset)")
    print(f"{'='*78}")

    for preset in ("easy", "medium", "hard"):
        print(f"  Running preset '{preset}' ...")
        random_data = _run_episodes(preset, use_fixed_setpoint=False)
        fixed_data  = _run_episodes(preset, use_fixed_setpoint=True)

        fixed_mean_biomass  = float(fixed_data["final_biomass"].mean())
        random_mean_biomass = float(random_data["final_biomass"].mean())
        fixed_mean_coverage  = float(fixed_data["final_coverage"].mean())
        random_mean_coverage = float(random_data["final_coverage"].mean())

        performance_gap   = float(fixed_data["rewards"].mean() - random_data["rewards"].mean())
        fixed_survival    = float(1.0 - fixed_data["extinctions"].mean())
        random_extinction = float(random_data["extinctions"].mean())
        coverage_variance = float(random_data["coverage"].var())

        results[preset] = {
            "fixed_mean_biomass":  fixed_mean_biomass,
            "random_mean_biomass": random_mean_biomass,
            "fixed_mean_coverage":  fixed_mean_coverage,
            "random_mean_coverage": random_mean_coverage,
            "biomass_gap":       fixed_mean_biomass - random_mean_biomass,
            "coverage_gap":      fixed_mean_coverage - random_mean_coverage,
            "performance_gap":   performance_gap,
            "fixed_survival":    fixed_survival,
            "random_extinction": random_extinction,
            "coverage_variance": coverage_variance,
        }

    _print_agent_comparison_table(results)
    _print_complexity_table(results)
    return results


def _print_agent_comparison_table(results: dict):
    """Print mean final biomass/coverage for both agents, per preset."""
    print(f"\n{'='*78}")
    print(f"  AGENT COMPARISON — Mean Final Biomass & Coverage")
    print(f"{'='*78}")
    print(f"  {'Preset':<8}{'Fixed Biomass':>16}{'Random Biomass':>17}"
          f"{'Fixed Coverage':>17}{'Random Coverage':>18}")
    print(f"  {'-'*76}")
    for preset in ("easy", "medium", "hard"):
        r = results[preset]
        print(f"  {preset:<8}"
              f"{r['fixed_mean_biomass']:>16.2f}"
              f"{r['random_mean_biomass']:>17.2f}"
              f"{r['fixed_mean_coverage']:>16.1f}%"
              f"{r['random_mean_coverage']:>17.1f}%")
    print(f"{'='*78}\n")


def _print_complexity_table(results: dict):
    """Print the complexity metrics (gaps, survival/extinction, variance)."""
    print(f"{'='*94}")
    print(f"  SUMMARY — World Preset Complexity")
    print(f"{'='*94}")
    print(f"  {'Preset':<8}{'Biomass Gap':>13}{'Coverage Gap':>14}{'Perf. Gap':>12}"
          f"{'Fixed Survival':>16}{'Random Extinct.':>17}{'Coverage Var.':>15}")
    print(f"  {'-'*90}")
    for preset in ("easy", "medium", "hard"):
        r = results[preset]
        print(f"  {preset:<8}"
              f"{r['biomass_gap']:>13.2f}"
              f"{r['coverage_gap']:>13.1f}%"
              f"{r['performance_gap']:>12.3f}"
              f"{r['fixed_survival']*100:>15.1f}%"
              f"{r['random_extinction']*100:>16.1f}%"
              f"{r['coverage_variance']:>15.5f}")
    print(f"{'='*94}\n")


if __name__ == "__main__":
    run_complexity_check()
