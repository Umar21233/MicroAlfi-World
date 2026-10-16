"""
test_regression.py -- Audit safety net: re-run the exact same fixed-setpoint
sweep as tests/snapshot_baseline.py and assert EXACT equality against
tests/baseline_snapshot.json.

This is the trip-wire for the audit's ONE RULE: if any edit to envs/,
agents/, or training/ changes a single bit of what the environment,
agents, rewards, or evaluation compute, this test fails. If it ever fails,
the change that caused it must be reverted.

Run after every commit made during the audit:
    python -m pytest tests/test_regression.py -q
or directly:
    python tests/test_regression.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tests.snapshot_baseline import (
    build_snapshot, SNAPSHOT_PATH, WORLD_PRESETS_LIST, REWARD_MODES,
    NO_DRIFTS_VALUES, SEEDS,
)


def load_expected_snapshot() -> dict:
    assert os.path.exists(SNAPSHOT_PATH), (
        f"Baseline snapshot not found at {SNAPSHOT_PATH}. "
        "Run tests/snapshot_baseline.py once (BEFORE making any edits) to create it."
    )
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


def test_fixed_setpoint_exact_regression():
    """
    Re-run every (world_preset, reward_mode, no_drifts, seed) combination
    from the frozen snapshot and assert each of the three tracked quantities
    (total_reward, final_coverage, final_total_biomass) is EXACTLY equal
    (bit-for-bit float equality -- no tolerance) to the snapshot value.
    """
    expected = load_expected_snapshot()
    actual = build_snapshot()

    assert set(expected.keys()) == set(actual.keys()), (
        "Snapshot key set changed -- world presets / reward modes / drift "
        "conditions no longer match the frozen baseline sweep."
    )

    mismatches = []
    for key in expected:
        exp_episodes = expected[key]
        act_episodes = actual[key]
        assert len(exp_episodes) == len(act_episodes) == len(SEEDS), (
            f"{key}: episode count changed "
            f"(expected {len(exp_episodes)}, got {len(act_episodes)})"
        )
        for i, (exp_ep, act_ep) in enumerate(zip(exp_episodes, act_episodes)):
            seed = SEEDS[i]
            for field in ("total_reward", "final_coverage", "final_total_biomass"):
                if exp_ep[field] != act_ep[field]:
                    mismatches.append(
                        f"{key} seed={seed} {field}: "
                        f"expected={exp_ep[field]!r} actual={act_ep[field]!r}"
                    )

    if mismatches:
        preview = "\n".join(mismatches[:20])
        more = f"\n... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        raise AssertionError(
            f"REGRESSION: {len(mismatches)} value(s) changed from the frozen "
            f"baseline snapshot. The ONE RULE has been violated -- revert the "
            f"change that caused this.\n{preview}{more}"
        )


if __name__ == "__main__":
    test_fixed_setpoint_exact_regression()
    print("PASS: all values exactly match tests/baseline_snapshot.json")
