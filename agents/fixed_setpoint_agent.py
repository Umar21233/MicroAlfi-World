"""
fixed_setpoint_agent.py — Baseline 2: Hand-coded Expert
=========================================================
A rule-based controller that uses domain knowledge (the sweet spots)
to manually maintain good growing conditions. No learning involved —
the rules are hardcoded by us, the designers.

This is the ceiling baseline. A trained RL agent should eventually
beat this score because it can discover more nuanced timing and
trade-offs than our hand-coded rules allow.

Why do we need this baseline?
  The random agent shows the floor. The fixed setpoint shows the ceiling
  a non-learning controller can achieve. If RL scores between these two,
  it's learning something but not generalising well. If it exceeds the
  fixed setpoint, it has genuinely discovered a better policy than we
  could write by hand.

Usage (from project root):
    python agents/fixed_setpoint_agent.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.alphae_world_env import (
    AlphaeWorldEnv,
    LIGHT_SWEET_LOW, LIGHT_SWEET_HIGH,
    CO2_SWEET_LOW,   CO2_SWEET_HIGH,
    PH_OPTIMAL,
)


# ── Setpoint targets ─────────────────────────────────────────────────────────
# We aim for the exact centre of each sweet spot, which is where the growth
# multiplier is maximised. A tolerance band of ±0.05 around each target
# prevents the controller from making unnecessary interventions when conditions
# are already acceptable (saving action cost).

LIGHT_TARGET    = (LIGHT_SWEET_LOW + LIGHT_SWEET_HIGH) / 2   # 0.65
CO2_TARGET      = (CO2_SWEET_LOW   + CO2_SWEET_HIGH)   / 2   # 0.525

# Refill nutrients when they drop below this fraction.
# Set to 0.25 (not 0.0) so we act before growth is severely impacted.
# If we waited until nutrients = 0, several steps of near-zero growth
# would already have been lost.
NUTRIENT_REFILL = 0.25


def fixed_setpoint_policy(obs: np.ndarray) -> int:
    """
    Priority-ordered rule-based controller.

    Reads the current state and returns the single action that addresses
    the most urgent problem. Rules are checked in priority order — if a
    higher-priority condition is met, lower-priority rules are never reached.

    Priority order (highest to lowest):
      1. Nutrients critical → refill immediately (growth stops without nutrients)
      2. pH too acidic      → buffer (silent killer — drifts slowly but hurts reward)
      3. Light out of range → nudge toward target
      4. CO2 out of range   → nudge toward target
      5. Everything fine    → do nothing (free action, saves cost)

    Why this priority order?
      Nutrients are first because depleted nutrients = zero growth, no matter
      how good everything else is. pH is second because it degrades silently
      over time and the penalty accumulates every step. Light and CO2 are
      last because the sweet spots are wide enough that small deviations don't
      immediately tank the reward.

    Args:
        obs : numpy array of shape (6,) — [light, temp, co2, nutrients, ph, biomass]

    Returns:
        int in {0, 1, 2, 3, 4, 5, 6} — action index
    """
    # Unpack the observation vector into named variables for readability.
    # temp and biomass are read but not acted on — this controller only
    # manages the four variables it has direct actions for.
    light, temp, co2, nutrients, ph, biomass = obs

    # ── Priority 1: nutrients critical ──────────────────────────────────────
    # If nutrients fall below the refill threshold, add them immediately.
    # This takes priority over everything else because without nutrients,
    # all other optimisations are pointless — growth rate collapses.
    if nutrients < NUTRIENT_REFILL:
        return 5   # add nutrients

    # ── Priority 2: pH too acidic ────────────────────────────────────────────
    # Growth acidifies the tank over time. If pH drops more than 0.10 below
    # optimal, the pH penalty in the reward function is eating into our score
    # every step. Buffer now rather than waiting for it to get worse.
    if ph < PH_OPTIMAL - 0.10:
        return 6   # adjust pH (buffer upward)

    # ── Priority 3: light out of acceptable range ────────────────────────────
    # We allow a ±0.05 tolerance around LIGHT_TARGET (0.65).
    # This means we only intervene when light < 0.60 or light > 0.70.
    # The tolerance prevents the agent from oscillating — without it, light
    # would be nudged up then down then up every step once it hits the target.
    if light < LIGHT_TARGET - 0.05:
        return 1   # increase light
    if light > LIGHT_TARGET + 0.05:
        return 2   # decrease light

    # ── Priority 4: CO2 out of acceptable range ──────────────────────────────
    # Same logic as light, but CO2 also depletes naturally as the organism
    # grows (the world dynamics draw it down). So in later steps, the
    # controller will increasingly need to top it up with action 3.
    if co2 < CO2_TARGET - 0.05:
        return 3   # increase co2
    if co2 > CO2_TARGET + 0.05:
        return 4   # decrease co2

    # ── Default: do nothing ──────────────────────────────────────────────────
    # If all conditions are within acceptable range, take no action.
    # action 0 costs 0.00, so this is always free. A well-tuned controller
    # should be doing nothing the majority of the time once it has nudged
    # conditions into the sweet spot — that's the hallmark of an efficient policy.
    return 0


def run_fixed_setpoint(n_episodes: int = 10, max_steps: int = 200, seed: int = 42):
    """
    Run the fixed setpoint controller for n_episodes and print a summary.

    Uses the same seeds as run_random_episodes() in random_agent.py so
    the two baselines face identical starting conditions and can be
    fairly compared.

    Args:
        n_episodes : number of full episodes to run
        max_steps  : maximum steps per episode (passed to env)
        seed       : base random seed (episode i uses seed + i)

    Returns:
        list of result dicts, one per episode
    """
    env = AlphaeWorldEnv(max_steps=max_steps)
    results = []

    print(f"\n{'='*55}")
    print(f"  AlphaeWorld — Fixed Setpoint Baseline ({n_episodes} episodes)")
    print(f"  Light target: {LIGHT_TARGET:.2f}  |  CO2 target: {CO2_TARGET:.2f}")
    print(f"{'='*55}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # Query the rule-based policy with the current observation
            action = fixed_setpoint_policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        final_biomass = float(obs[5])
        results.append({
            "episode":       ep + 1,
            "total_reward":  round(ep_reward, 3),
            "steps":         env.step_count,
            "final_biomass": round(final_biomass, 4),
            "crashed":       bool(terminated),
        })

        print(f"  Ep {ep+1:>2}: reward={ep_reward:>8.3f}  "
              f"biomass={final_biomass:.4f}  "
              f"steps={env.step_count}  "
              f"{'💀 CRASH' if terminated else '✓'}")

    env.close()

    # ── Summary statistics ───────────────────────────────────────────────────
    # These numbers are the "ceiling" your RL agents must exceed to demonstrate
    # they have genuinely learned something better than a hand-coded rule.
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
    run_fixed_setpoint(n_episodes=10)