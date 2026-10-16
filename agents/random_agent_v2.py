"""
random_agent_v2.py — Baseline 1 for Grid World
================================================
Runs a random agent on AlphaeWorldEnvV2. Supports reward_mode and
world_config arguments so the same baseline can be evaluated across
different world and reward configurations.

Usage:
    python agents/random_agent_v2.py
    python agents/random_agent_v2.py --render
    python agents/random_agent_v2.py --reward_mode turbidity
    python agents/random_agent_v2.py --world easy
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.alphae_world_env_v2 import AlphaeWorldEnvV2, DEFAULT_WORLD_CONFIG

# Preset world configurations for easy comparison
WORLD_PRESETS = {
    "easy":    {"spreading": True,  "mutation": False, "toxin": False,
                "colony_support": False, "photoinhibition": False},
    "medium":  {"spreading": True,  "mutation": False, "toxin": True,
                "colony_support": True,  "photoinhibition": False},
    "hard":    DEFAULT_WORLD_CONFIG,   # all rules active
}


def run_random_v2(
    n_episodes:  int  = 5,
    max_steps:   int  = 200,
    seed:        int  = 42,
    render:      bool = False,
    reward_mode: str  = "multi_objective",
    world_preset: str = "hard",
    no_drifts:   bool = False,
):
    """
    Run a random agent on AlphaeWorldEnvV2.

    The random agent samples uniformly from all 11 actions regardless of
    observation — it is completely state-blind. Its scores establish the
    floor baseline for each world/reward configuration.

    Args:
        n_episodes   : number of episodes
        max_steps    : steps per episode
        seed         : base random seed
        render       : open pygame window if True
        reward_mode  : reward function to use
        world_preset : "easy", "medium", or "hard"
        no_drifts    : if True, disable passive global drifts (except
                       radiation), matching run_fixed_setpoint_v2 so both
                       baselines can be run under either drift condition
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
    print(f"  AlphaeWorld v2 — Random Agent  |  "
          f"reward={reward_mode}  world={world_preset}")
    print(f"  Grid: 10×10  |  Actions: 11  |  Obs: 13 dims")
    print(f"{'='*65}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done      = False

        while not done:
            action = env.action_space.sample()
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
    run_random_v2(
        n_episodes=args.episodes,
        render=args.render,
        reward_mode=args.reward_mode,
        world_preset=args.world,
        no_drifts=args.no_drifts,
    )