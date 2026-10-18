"""
test_ablation_equivalence.py -- Drift study Part B.2.

Proves AblationDriftEnv (ablation_env.py) changes nothing except which
individual passive drifts are active:
    - all six flags True  must be bit-exact equal to AlphaeWorldEnvV2(no_drifts=False)
    - all six flags False must be bit-exact equal to AlphaeWorldEnvV2(no_drifts=True)

Runs 5 episodes of random actions (same seeds, same action sequence) on
each pair and asserts every recorded global_state, biomass, local_toxin,
reward and observation is exactly equal (not merely close).

Usage (from project root):
    venv\\Scripts\\python.exe experiments\\drift_study\\test_ablation_equivalence.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from experiments.drift_study.ablation_env import AblationDriftEnv, ALL_ON, ALL_OFF  # noqa: E402

N_EPISODES = 5
MAX_STEPS = 200
SEED_START = 100  # same evaluation seed block used throughout this project


def run_episodes(env, seed_start, n_episodes, action_seed):
    """Run n_episodes with a fixed, reproducible random action sequence and
    return the full trajectory of (obs, reward, terminated, truncated,
    global_state, biomass, local_toxin) for every step of every episode."""
    action_rng = np.random.RandomState(action_seed)
    trace = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        trace.append(("reset", obs.copy()))
        done = False
        while not done:
            action = int(action_rng.randint(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append((
                "step", action, obs.copy(), float(reward), terminated, truncated,
                env.global_state.copy(), env.biomass.copy(), env.local_toxin.copy(),
            ))
            done = terminated or truncated
    return trace


def traces_equal(trace_a, trace_b) -> bool:
    if len(trace_a) != len(trace_b):
        print(f"  Length mismatch: {len(trace_a)} vs {len(trace_b)}")
        return False
    for i, (a, b) in enumerate(zip(trace_a, trace_b)):
        if a[0] != b[0]:
            print(f"  Step {i}: record type mismatch {a[0]} vs {b[0]}")
            return False
        if a[0] == "reset":
            if not np.array_equal(a[1], b[1]):
                print(f"  Step {i} (reset): obs mismatch")
                return False
        else:
            _, action_a, obs_a, reward_a, term_a, trunc_a, g_a, bio_a, tox_a = a
            _, action_b, obs_b, reward_b, term_b, trunc_b, g_b, bio_b, tox_b = b
            if action_a != action_b:
                print(f"  Step {i}: action mismatch {action_a} vs {action_b}")
                return False
            if not np.array_equal(obs_a, obs_b):
                print(f"  Step {i}: obs mismatch")
                return False
            if reward_a != reward_b:
                print(f"  Step {i}: reward mismatch {reward_a} vs {reward_b}")
                return False
            if term_a != term_b or trunc_a != trunc_b:
                print(f"  Step {i}: done-flag mismatch")
                return False
            if not np.array_equal(g_a, g_b):
                print(f"  Step {i}: global_state mismatch\n    {g_a}\n    {g_b}")
                return False
            if not np.array_equal(bio_a, bio_b):
                print(f"  Step {i}: biomass array mismatch")
                return False
            if not np.array_equal(tox_a, tox_b):
                print(f"  Step {i}: local_toxin array mismatch")
                return False
    return True


def check(label, ablation_flags, no_drifts, world="hard", action_seed=12345):
    env_original = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world], reward_mode="multi_objective", no_drifts=no_drifts,
    )
    env_ablation = AblationDriftEnv(
        drift_flags=ablation_flags,
        world_config=WORLD_PRESETS[world], reward_mode="multi_objective", no_drifts=no_drifts,
    )
    trace_original = run_episodes(env_original, SEED_START, N_EPISODES, action_seed)
    trace_ablation = run_episodes(env_ablation, SEED_START, N_EPISODES, action_seed)
    env_original.close()
    env_ablation.close()

    equal = traces_equal(trace_original, trace_ablation)
    n_steps = sum(1 for r in trace_original if r[0] == "step")
    print(f"{label}: {'PASS' if equal else 'FAIL'} "
          f"({N_EPISODES} episodes, {n_steps} total steps compared)")
    return equal


def main():
    ok_on = check("All six drifts ON  vs no_drifts=False", ALL_ON, no_drifts=False)
    ok_off = check("All six drifts OFF vs no_drifts=True ", ALL_OFF, no_drifts=True)

    if ok_on and ok_off:
        print("\nPASS: AblationDriftEnv is bit-exact equivalent to AlphaeWorldEnvV2 "
              "at both of its original extremes.")
    else:
        print("\nFAILURE: ablation env does not reproduce the original environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
