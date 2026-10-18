"""
verify_tuned_hyperparameters.py -- Step 4 proof.

Confirms that passing --tuned to train_dqn.py / train_ppo.py / train_a2c.py
actually produces a saved model whose stored hyperparameters match
training/hyperparameters.TUNED (and therefore the values read from the
real Colab-trained models.zip files under "Colab training results/" --
see docs/VERIFICATION.md Step 4).

Trains each algorithm for only 10,000 timesteps (seed 42), each into its
own temporary directory (PROJECT_ROOT monkeypatched, exactly like
prove_naming_equivalence.py), then reads the `data` JSON back out of the
saved model.zip and checks it against training.hyperparameters.TUNED.
Never touches results/, trained_models/, or "Colab training results/".

Usage:
    python scripts/verify_tuned_hyperparameters.py
"""

import json
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import training.train_dqn as train_dqn  # noqa: E402
import training.train_ppo as train_ppo  # noqa: E402
import training.train_a2c as train_a2c  # noqa: E402
from training.hyperparameters import TUNED  # noqa: E402

MODULES = {"dqn": train_dqn, "ppo": train_ppo, "a2c": train_a2c}


def run_isolated(module, tmp_root: Path):
    old_project_root = module.PROJECT_ROOT
    old_argv = sys.argv
    module.PROJECT_ROOT = str(tmp_root)
    sys.argv = ["train.py", "--timesteps", "10000", "--seed", "42", "--tuned"]
    try:
        module.main()
    finally:
        module.PROJECT_ROOT = old_project_root
        sys.argv = old_argv


def main():
    base = Path(tempfile.mkdtemp(prefix="tuned_hp_proof_"))
    all_ok = True

    for algo, module in MODULES.items():
        tmp_root = base / algo
        tmp_root.mkdir()
        print(f"Training {algo.upper()} for 10,000 timesteps (seed 42, --tuned) -> {tmp_root}")
        run_isolated(module, tmp_root)

        model_dirs = list((tmp_root / "trained_models").glob(f"{algo}_*"))
        model_zip = model_dirs[0] / "model.zip"
        with zipfile.ZipFile(model_zip) as zf:
            with zf.open("data") as f:
                data = json.load(f)

        expected = TUNED[algo]
        print(f"  Expected (training/hyperparameters.TUNED['{algo}']): {expected}")
        actual = {k: data.get(k) for k in expected}
        print(f"  Actual (saved model.zip 'data'):                 {actual}")

        ok = all(data.get(k) == v for k, v in expected.items())
        print(f"  {'PASS' if ok else 'FAIL'}: {algo} tuned hyperparameters match\n")
        all_ok = all_ok and ok

    shutil.rmtree(base, ignore_errors=True)
    print(f"Cleaned up {base}")

    if not all_ok:
        print("\nFAILURE: at least one algorithm's --tuned run did not match training/hyperparameters.TUNED.")
        sys.exit(1)
    print("\nPASS: --tuned reproduces the Colab-trained models' hyperparameters for all three algorithms.")


if __name__ == "__main__":
    main()
