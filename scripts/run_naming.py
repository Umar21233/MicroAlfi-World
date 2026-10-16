"""
run_naming.py -- Canonical run-id helper.

OLD naming (still used by training/*.py and training/evaluate.py's default
behaviour, and by every file in results/ and the raw Colab zips):
    {algo}_{world}_{reward_mode}_{seed}
This has no drift-condition component, so a with-drifts and a no-drifts run
of the same algo/world/reward_mode/seed collide (see AUDIT.md finding #12).

NEW naming (for FUTURE runs only -- see training/train_dqn.py etc. for the
opt-in flag that builds this instead of the old 4-part name):
    {algo}_{world}_{reward_mode}_{drift}_{seed}
where drift is "drifts" (no_drifts=False) or "nodrifts" (no_drifts=True).

This module parses BOTH. It does not change any training/evaluation logic;
it is a pure string-parsing helper.
"""

import re

REWARD_MODES = ["multi_objective", "biomass_only", "turbidity",
                "turbidity_delta", "efficiency", "sustainability"]
WORLDS = ["easy", "medium", "hard"]
ALGOS = ["dqn", "ppo", "a2c"]
DRIFTS = ["drifts", "nodrifts"]

# Longest-reward-mode-first so "turbidity_delta" matches before "turbidity"
_REWARD_MODES_BY_LEN = sorted(REWARD_MODES, key=len, reverse=True)


def drift_token(no_drifts: bool) -> str:
    """bool -> 'drifts' / 'nodrifts' token used in the new naming scheme."""
    return "nodrifts" if no_drifts else "drifts"


def no_drifts_from_token(token: str) -> bool:
    if token == "drifts":
        return False
    if token == "nodrifts":
        return True
    raise ValueError(f"Unknown drift token: {token!r} (expected 'drifts' or 'nodrifts')")


def build_run_name(algo: str, world: str, reward_mode: str, seed, *,
                    no_drifts: bool = None, drift: str = None,
                    style: str = "old") -> str:
    """
    Build a run name in either style.
        style="old" (default, UNCHANGED existing behaviour):
            {algo}_{world}_{reward_mode}_{seed}
        style="new":
            {algo}_{world}_{reward_mode}_{drift}_{seed}
            requires either no_drifts (bool) or drift ("drifts"/"nodrifts")
    """
    if style == "old":
        return f"{algo}_{world}_{reward_mode}_{seed}"
    elif style == "new":
        if drift is None:
            if no_drifts is None:
                raise ValueError("style='new' requires no_drifts or drift")
            drift = drift_token(no_drifts)
        if drift not in DRIFTS:
            raise ValueError(f"drift must be one of {DRIFTS}, got {drift!r}")
        return f"{algo}_{world}_{reward_mode}_{drift}_{seed}"
    else:
        raise ValueError(f"style must be 'old' or 'new', got {style!r}")


def parse_run_name(name: str) -> dict:
    """
    Parse EITHER the old 4-part name ({algo}_{world}_{reward_mode}_{seed})
    or the new 5-part name ({algo}_{world}_{reward_mode}_{drift}_{seed}).

    Returns a dict: {algo, world, reward_mode, seed, drift, no_drifts, style}
    where drift/no_drifts are None if the name has no drift component
    (old style -- drift condition is then simply unknown from the name alone).

    Raises ValueError if the name cannot be parsed against any known
    algo/world/reward_mode vocabulary.
    """
    for algo in ALGOS:
        prefix = algo + "_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for world in WORLDS:
            wprefix = world + "_"
            if not rest.startswith(wprefix):
                continue
            rest2 = rest[len(wprefix):]
            for reward_mode in _REWARD_MODES_BY_LEN:
                rprefix = reward_mode + "_"
                if not rest2.startswith(rprefix):
                    continue
                tail = rest2[len(rprefix):]
                # tail is either "{seed}" (old) or "{drift}_{seed}" (new)
                m_new = re.fullmatch(r"(drifts|nodrifts)_(\d+)", tail)
                if m_new:
                    drift, seed = m_new.group(1), int(m_new.group(2))
                    return {
                        "algo": algo, "world": world, "reward_mode": reward_mode,
                        "seed": seed, "drift": drift,
                        "no_drifts": no_drifts_from_token(drift),
                        "style": "new",
                    }
                m_old = re.fullmatch(r"(\d+)", tail)
                if m_old:
                    seed = int(m_old.group(1))
                    return {
                        "algo": algo, "world": world, "reward_mode": reward_mode,
                        "seed": seed, "drift": None, "no_drifts": None,
                        "style": "old",
                    }
    raise ValueError(f"Could not parse run name: {name!r}")


def canonical_run_id(algo: str, world: str, reward_mode: str, seed, no_drifts: bool) -> str:
    """The canonical 5-part run id used throughout results_verified/."""
    return build_run_name(algo, world, reward_mode, seed, no_drifts=no_drifts, style="new")


if __name__ == "__main__":
    # Tiny self-test
    tests_old = [
        "dqn_hard_multi_objective_42",
        "ppo_easy_turbidity_delta_999",
        "a2c_medium_sustainability_7",
    ]
    tests_new = [
        "dqn_hard_multi_objective_drifts_42",
        "dqn_easy_multi_objective_nodrifts_43",
        "ppo_hard_turbidity_delta_drifts_5001",
    ]
    for n in tests_old + tests_new:
        print(n, "->", parse_run_name(n))
    assert canonical_run_id("dqn", "easy", "multi_objective", 43, True) == \
        "dqn_easy_multi_objective_nodrifts_43"
    print("OK")
