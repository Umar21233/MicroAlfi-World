"""
prove_naming_equivalence.py -- Step 6.2 proof: the --new_naming flag added
to training/train_dqn.py changes ONLY the output file/folder name, never
the training logic, hyperparameters, or RNG sequence.

Runs a short (10,000 timestep) DQN training on seed 42 TWICE in-process:
once with the old naming (default) and once with --new_naming, each
redirected to its own temporary folder by monkeypatching the module's
PROJECT_ROOT global (train_dqn.py's own code is never modified or run
with its real default paths, so results/ and trained_models/ are never
touched). Then diffs the two runs' learning_curve.csv byte-for-byte.

Usage:
    python scripts/prove_naming_equivalence.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import training.train_dqn as train_dqn  # noqa: E402


def run_isolated(extra_args, tmp_root: Path):
    """Run train_dqn.main() with PROJECT_ROOT monkeypatched to tmp_root."""
    old_project_root = train_dqn.PROJECT_ROOT
    old_argv = sys.argv
    train_dqn.PROJECT_ROOT = str(tmp_root)
    sys.argv = ["train_dqn.py", "--timesteps", "10000", "--seed", "42"] + extra_args
    try:
        train_dqn.main()
    finally:
        train_dqn.PROJECT_ROOT = old_project_root
        sys.argv = old_argv


def main():
    base = Path(tempfile.mkdtemp(prefix="naming_proof_"))
    old_dir = base / "old_naming"
    new_dir = base / "new_naming"
    old_dir.mkdir()
    new_dir.mkdir()

    print(f"Running OLD-naming 10,000-step DQN training (seed 42) -> {old_dir}")
    run_isolated([], old_dir)

    print(f"Running NEW-naming 10,000-step DQN training (seed 42) -> {new_dir}")
    run_isolated(["--new_naming"], new_dir)

    old_lc = old_dir / "results" / "dqn_hard_multi_objective_42_learning_curve.csv"
    new_lc = new_dir / "results" / "dqn_hard_multi_objective_drifts_42_learning_curve.csv"

    print(f"\nOld-naming output:  {old_lc} (exists={old_lc.exists()})")
    print(f"New-naming output:  {new_lc} (exists={new_lc.exists()})")

    old_text = old_lc.read_text()
    new_text = new_lc.read_text()

    identical = (old_text == new_text)
    print(f"\nlearning_curve.csv contents byte-for-byte identical: {identical}")
    if not identical:
        print("OLD:\n" + old_text)
        print("NEW:\n" + new_text)
    else:
        print(old_text)

    shutil.rmtree(base, ignore_errors=True)
    print(f"\nCleaned up {base}")

    if not identical:
        print("\nFAILURE: naming change altered training behaviour.")
        sys.exit(1)
    print("\nPASS: --new_naming changes only the output name, not training behaviour.")


if __name__ == "__main__":
    main()
