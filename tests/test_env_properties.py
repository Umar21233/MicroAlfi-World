"""
test_env_properties.py -- General correctness tests for AlphaeWorldEnvV2.

Unlike test_regression.py (which pins exact values from a frozen snapshot),
these tests check structural properties that should hold regardless of the
exact numbers: the env satisfies the Gymnasium API, is seed-deterministic,
stays within its declared observation bounds, and runs to completion for
every reward mode / world preset / drift-condition combination.

Run with:
    python -m pytest tests/test_env_properties.py -q
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS

REWARD_MODES = ["multi_objective", "biomass_only", "turbidity",
                "turbidity_delta", "efficiency", "sustainability"]
WORLD_PRESET_NAMES = ["easy", "medium", "hard"]


def test_check_env_passes():
    """Gymnasium's own environment-contract checker must pass."""
    from gymnasium.utils.env_checker import check_env
    env = AlphaeWorldEnvV2(world_config=WORLD_PRESETS["hard"])
    check_env(env.unwrapped if hasattr(env, "unwrapped") else env, skip_render_check=True)
    env.close()


def test_same_seed_gives_identical_trajectory():
    """Resetting and stepping with the same seed and the same action
    sequence must reproduce bit-identical observations and rewards."""
    def run(seed, actions):
        env = AlphaeWorldEnvV2(world_config=WORLD_PRESETS["hard"])
        obs, info = env.reset(seed=seed)
        trace = [obs.copy()]
        rewards = []
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            trace.append(obs.copy())
            rewards.append(reward)
        env.close()
        return trace, rewards

    rng = np.random.default_rng(0)
    actions = [int(a) for a in rng.integers(0, 11, size=50)]

    trace1, rewards1 = run(123, actions)
    trace2, rewards2 = run(123, actions)

    assert len(trace1) == len(trace2)
    for o1, o2 in zip(trace1, trace2):
        np.testing.assert_array_equal(o1, o2)
    assert rewards1 == rewards2


@pytest.mark.parametrize("world_preset", WORLD_PRESET_NAMES)
@pytest.mark.parametrize("reward_mode", REWARD_MODES)
def test_observations_within_bounds(world_preset, reward_mode):
    """Every observation returned must lie within observation_space bounds."""
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world_preset],
        reward_mode=reward_mode,
    )
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs), f"reset() obs out of bounds: {obs}"
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"step() obs out of bounds: {obs}"
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("world_preset", WORLD_PRESET_NAMES)
@pytest.mark.parametrize("reward_mode", REWARD_MODES)
@pytest.mark.parametrize("no_drifts", [False, True])
def test_full_episode_runs_without_error(world_preset, reward_mode, no_drifts):
    """Every (preset, reward_mode, no_drifts) combination must be able to
    run a full 200-step episode (or terminate early via extinction) without
    raising, and must respect max_steps as a hard cap."""
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world_preset],
        reward_mode=reward_mode,
        no_drifts=no_drifts,
        max_steps=200,
    )
    obs, info = env.reset(seed=11)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps <= 200, "episode exceeded max_steps without truncating"
        assert np.isfinite(reward), f"non-finite reward: {reward}"
    env.close()


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-q"]))
