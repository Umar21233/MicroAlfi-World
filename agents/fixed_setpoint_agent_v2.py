"""
fixed_setpoint_agent_v2.py — Baseline 2 for Grid World
========================================================
Priority-ordered rule-based controller for AlphaeWorldEnvV2.
Extended to handle all 8 global conditions and 11 actions.
Supports reward_mode and world_config arguments for fair comparison
against the random agent across all experimental configurations.

Priority order (highest to lowest):
  1. Toxin critical         → activate toxin filter
  2. Colony coverage low    → trigger growth pulse
  3. Nutrients critical     → refill
  4. pH too acidic          → buffer
  5. Radiation too high     → shield
  6. Light out of range     → nudge
  7. CO2 out of range       → nudge
  8. Humidity out of range  → nudge
  9. Everything acceptable  → do nothing

Usage:
    python agents/fixed_setpoint_agent_v2.py
    python agents/fixed_setpoint_agent_v2.py --render
    python agents/fixed_setpoint_agent_v2.py --reward_mode turbidity
    python agents/fixed_setpoint_agent_v2.py --world easy
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.alphae_world_env_v2 import (
    AlphaeWorldEnvV2,
    LIGHT_SWEET_LOW, LIGHT_SWEET_HIGH,
    CO2_SWEET_LOW,   CO2_SWEET_HIGH,
    PH_OPTIMAL, HUMIDITY_OPTIMAL,
)
from agents.random_agent_v2 import WORLD_PRESETS

# ── Setpoint targets — centres of sweet spots ────────────────────────────────
LIGHT_TARGET    = (LIGHT_SWEET_LOW + LIGHT_SWEET_HIGH) / 2    # 0.65
CO2_TARGET      = (CO2_SWEET_LOW   + CO2_SWEET_HIGH)   / 2    # 0.525
HUMIDITY_TARGET = HUMIDITY_OPTIMAL                              # 0.55

# ── Intervention thresholds ───────────────────────────────────────────────────
NUTRIENT_REFILL         = 0.25
PH_BUFFER_THRESH        = PH_OPTIMAL - 0.10
TOXIN_FILTER_THRESH     = 0.18
RADIATION_SHIELD_THRESH = 0.55
COVERAGE_PULSE_THRESH   = 0.30
TOLERANCE               = 0.06


def fixed_setpoint_policy_v2(obs: np.ndarray) -> int:
    """
    Priority-ordered rule-based controller for AlphaeWorldEnvV2.

    Reads the 13-dimensional observation and returns the most urgent action.
    Rules are checked strictly in priority order — once a condition is met,
    lower-priority rules are not evaluated that step.

    Observation index map:
      [0] light  [1] temp  [2] co2   [3] nutrients  [4] ph
      [5] toxin  [6] rad   [7] humid [8] mean_bio   [9] coverage
      [10] mean_toxin  [11] type_b_frac  [12] mean_age

    Args:
        obs : numpy array of shape (13,)

    Returns:
        int in {0, 1, ..., 10}
    """
    light, temp, co2  = obs[0], obs[1], obs[2]
    nutrients, ph     = obs[3], obs[4]
    toxin, radiation, humidity = obs[5], obs[6], obs[7]
    coverage          = obs[9]

    if toxin > TOXIN_FILTER_THRESH:         return 7    # toxin filter
    if coverage < COVERAGE_PULSE_THRESH:    return 10   # growth pulse
    if nutrients < NUTRIENT_REFILL:         return 5    # add nutrients
    if ph < PH_BUFFER_THRESH:               return 6    # buffer pH
    if radiation > RADIATION_SHIELD_THRESH: return 8    # radiation shield

    if light < LIGHT_TARGET - TOLERANCE:    return 1    # increase light
    if light > LIGHT_TARGET + TOLERANCE:    return 2    # decrease light
    if co2   < CO2_TARGET   - TOLERANCE:    return 3    # increase co2
    if co2   > CO2_TARGET   + TOLERANCE:    return 4    # decrease co2
    if humidity < HUMIDITY_TARGET - TOLERANCE: return 9 # add humidity

    return 0   # do nothing — all conditions acceptable


def run_fixed_setpoint_v2(
    n_episodes:   int  = 5,
    max_steps:    int  = 200,
    seed:         int  = 42,
    render:       bool = False,
    reward_mode:  str  = "multi_objective",
    world_preset: str  = "hard",
    no_drifts:    bool = False,
):
    """
    Run the fixed setpoint controller on AlphaeWorldEnvV2.

    Uses the same seeds and configurations as run_random_v2 so results
    are directly comparable. This is the ceiling baseline — RL agents
    must exceed this to demonstrate genuine learning.

    Args:
        n_episodes   : number of episodes
        max_steps    : steps per episode
        seed         : base random seed
        render       : open pygame window if True
        reward_mode  : reward function to use
        world_preset : "easy", "medium", or "hard"
        no_drifts    : if True, disable passive global drifts (except radiation)
    """
    world_config = WORLD_PRESETS[world_preset]
    env = AlphaeWorldEnvV2(
        max_steps=max_steps,
        reward_mode=reward_mode,
        world_config=world_config,
        no_drifts=no_drifts,
    )
    results = []

    renderer = None
    if render:
        from envs.renderer import AlphaeRenderer
        renderer = AlphaeRenderer(cell_size=50, fps=10)

    print(f"\n{'='*65}")
    print(f"  AlphaeWorld v2 — Fixed Setpoint  |  "
          f"reward={reward_mode}  world={world_preset}")
    print(f"  Light: {LIGHT_TARGET:.2f}  CO2: {CO2_TARGET:.2f}  "
          f"Humidity: {HUMIDITY_TARGET:.2f}")
    print(f"{'='*65}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done      = False

        while not done:
            action = fixed_setpoint_policy_v2(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if renderer:
                renderer.render(env)

        final_coverage = float(obs[9])
        type_b_pct     = float(obs[11]) * 100

        results.append({
            "episode":      ep + 1,
            "total_reward": round(ep_reward, 3),
            "steps":        env.step_count,
            "coverage":     round(final_coverage, 4),
            "type_b_pct":   round(type_b_pct, 2),
            "extinct":      bool(terminated),
        })

        print(f"  Ep {ep+1:>2}: reward={ep_reward:>9.3f}  "
              f"coverage={final_coverage*100:>5.1f}%  "
              f"TypeB={type_b_pct:>4.1f}%  "
              f"steps={env.step_count}  "
              f"{'EXTINCT' if terminated else 'OK'}")

    if renderer:
        renderer.close()
    env.close()

    rewards     = [r["total_reward"] for r in results]
    coverages   = [r["coverage"]     for r in results]
    extinctions = sum(r["extinct"]   for r in results)

    print(f"\n{'-'*65}")
    print(f"  Mean reward:   {np.mean(rewards):>10.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean coverage: {np.mean(coverages)*100:>9.1f}% "
          f"± {np.std(coverages)*100:.1f}%")
    print(f"  Extinctions:   {extinctions}/{n_episodes}")
    print(f"{'='*65}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--episodes",     type=int, default=5)
    parser.add_argument("--reward_mode",  type=str, default="multi_objective",
                        choices=["multi_objective","biomass_only",
                                 "turbidity","turbidity_delta","efficiency","sustainability"])
    parser.add_argument("--world",        type=str, default="hard",
                        choices=["easy","medium","hard"])
    parser.add_argument("--no_drifts",    action="store_true")
    args = parser.parse_args()
    run_fixed_setpoint_v2(
        n_episodes=args.episodes,
        render=args.render,
        reward_mode=args.reward_mode,
        world_preset=args.world,
        no_drifts=args.no_drifts,
    )