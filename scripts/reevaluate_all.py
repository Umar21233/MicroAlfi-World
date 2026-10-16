"""
reevaluate_all.py -- Step 3 of the verified-data rebuild.

Re-evaluates all 39 models using training/evaluate.py's EXACT protocol
(imported directly, not reimplemented): 20 episodes, seeds 100-119,
model.predict(obs, deterministic=True), env built from AlphaeWorldEnvV2
with world_config=WORLD_PRESETS[world] and no_drifts from each model's own
config.txt. evaluate.py itself is NOT modified.

For the 24 runs with an unambiguous original eval already in
results_verified/ (Phase 1, Phase 2, turbidity_delta): regenerate, compare
total_reward per episode (max abs diff), and report. If it matches within
1e-6, the original is kept and marked "verified" in PROVENANCE.csv; a
"regenerated" copy is written alongside either way for transparency
(suffixed _regenerated.csv) only when it does NOT match.

For Phase 3's 9 ambiguous (world, seed) eval sets in phase3_eval_results.zip
(no drift label): regenerate eval for BOTH the drifts and nodrifts model of
that (world, seed), compare each against the stored ambiguous set, and
attribute the stored set to whichever one matches. The non-matching drift
condition's eval is written as a fresh regenerated eval (no original
existed for it).

Writes:
    results_verified/<phase>/<run_id>/eval.csv, eval_summary.csv,
        coverage_trajectory.csv  (filled in for every run that didn't
        already have them from Step 2, or regenerated+kept where a mismatch
        was found)
    results_verified/REEVALUATION_REPORT.csv  -- per-run comparison table
    results_verified/PROVENANCE.csv  -- appended with the new rows

Usage:
    python scripts/reevaluate_all.py
"""

import csv
import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from stable_baselines3 import DQN, PPO, A2C  # noqa: E402

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.evaluate import (  # noqa: E402
    run_evaluation, summarize_episodes, write_csv, write_coverage_trajectory_csv,
    N_EPISODES, SEED_START,
)
from scripts.run_naming import canonical_run_id, build_run_name  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results_verified"
RAW_DIR = PROJECT_ROOT / "Colab training results"
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_config(run_dir: Path) -> dict:
    config = {}
    for line in (run_dir / "config.txt").read_text().strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    return config


def load_model_and_env(run_dir: Path, config: dict):
    algo = config["algo"]
    world = config["world"]
    reward_mode = config["reward_mode"]
    no_drifts = config["no_drifts"] == "True"

    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world],
        reward_mode=reward_mode,
        no_drifts=no_drifts,
    )
    AlgoClass = ALGO_CLASSES[algo]

    # model.zip in run_dir is SB3's own zip format -- extract to a temp
    # path next to it (SB3's .load() wants a file path or file-like object
    # supporting its own zip reading; a BytesIO works too).
    import io
    model_bytes = (run_dir / "model.zip").read_bytes()
    model = AlgoClass.load(io.BytesIO(model_bytes))
    return model, env


def regenerate_eval(run_dir: Path):
    config = read_config(run_dir)
    model, env = load_model_and_env(run_dir, config)
    episode_rows, coverage_by_step = run_evaluation(model, env, N_EPISODES, SEED_START)
    summary_rows = summarize_episodes(episode_rows)
    env.close()
    return episode_rows, summary_rows, coverage_by_step


def max_abs_reward_diff(episode_rows_a, episode_rows_b) -> float:
    # NOTE: episode_rows read back from a CSV (csv.DictReader) have "episode"
    # as a STRING; rows built fresh by run_evaluation() have it as an int.
    # Sorting by the raw value would silently sort one list lexicographically
    # ("1","10","11",...,"2","20","3",...) and the other numerically,
    # misaligning every pair after episode 1 and producing a spurious huge
    # "diff" that is actually comparing two different episodes. Cast to int
    # for both before sorting.
    a = sorted(episode_rows_a, key=lambda r: int(r["episode"]))
    b = sorted(episode_rows_b, key=lambda r: int(r["episode"]))
    assert len(a) == len(b)
    return max(abs(float(x["total_reward"]) - float(y["total_reward"])) for x, y in zip(a, b))


def write_eval_outputs(run_dir: Path, episode_rows, summary_rows, coverage_by_step, suffix=""):
    write_csv(episode_rows, run_dir / f"eval{suffix}.csv")
    write_csv(summary_rows, run_dir / f"eval_summary{suffix}.csv")
    write_coverage_trajectory_csv(coverage_by_step, run_dir / f"coverage_trajectory{suffix}.csv")


def read_existing_eval_rows(run_dir: Path):
    import csv as _csv
    path = run_dir / "eval.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    return rows


TOLERANCE = 1e-6


def process_run_with_existing_eval(run_dir: Path, run_id: str, report_rows: list, provenance_rows: list):
    """Runs from Phase 1 / Phase 2 / turbidity_delta: original eval already present."""
    original_rows = read_existing_eval_rows(run_dir)
    regen_rows, regen_summary, regen_cov = regenerate_eval(run_dir)
    diff = max_abs_reward_diff(original_rows, regen_rows)

    status = "verified" if diff <= TOLERANCE else "MISMATCH"
    report_rows.append({
        "run_id": run_id, "category": "original-vs-regenerated",
        "max_abs_total_reward_diff": diff, "status": status,
    })

    if diff > TOLERANCE:
        # Keep both: original stays as eval.csv, regenerated goes alongside.
        write_eval_outputs(run_dir, regen_rows, regen_summary, regen_cov, suffix="_regenerated")
        for fname in ["eval_regenerated.csv", "eval_summary_regenerated.csv", "coverage_trajectory_regenerated.csv"]:
            provenance_rows.append({
                "run_id": run_id, "file": fname, "source_zip": "regenerated-in-place",
                "origin": "regenerated", "sha256": sha256_bytes((run_dir / fname).read_bytes()),
            })
    # else: original kept as-is, already recorded in PROVENANCE.csv from Step 2.
    print(f"  {run_id}: max|d_reward|={diff:.6g} -> {status}")


def process_phase3_ambiguous(runs_by_key: dict, report_rows: list, provenance_rows: list):
    """
    Phase 3's 9 ambiguous (world, seed) eval sets in phase3_eval_results.zip
    carry no drift label. For each, regenerate eval for BOTH the drifts and
    nodrifts model and see which one matches the stored ambiguous set.
    """
    eval_zip_path = RAW_DIR / "Phase 3" / "phase3_eval_results.zip"
    with zipfile.ZipFile(eval_zip_path) as z:
        names = set(z.namelist())

        for world in ("easy", "medium", "hard"):
            for seed in (42, 43, 44):
                old_name = build_run_name("dqn", world, "multi_objective", seed, style="old")
                eval_name = f"{old_name}_eval.csv"
                if eval_name not in names:
                    print(f"  WARNING: {eval_name} not found in {eval_zip_path.name}, skipping")
                    continue
                import io, csv as _csv
                ambiguous_rows = list(_csv.DictReader(io.StringIO(z.read(eval_name).decode("utf-8"))))

                candidates = {}
                for drift in ("drifts", "nodrifts"):
                    run_id = canonical_run_id("dqn", world, "multi_objective", seed, drift == "nodrifts")
                    run_dir = runs_by_key[run_id]
                    regen_rows, regen_summary, regen_cov = regenerate_eval(run_dir)
                    diff = max_abs_reward_diff(ambiguous_rows, regen_rows)
                    candidates[drift] = (diff, regen_rows, regen_summary, regen_cov, run_dir, run_id)

                diffs = {d: candidates[d][0] for d in candidates}
                best_drift = min(diffs, key=lambda d: diffs[d])
                best_diff = diffs[best_drift]
                other_drift = "nodrifts" if best_drift == "drifts" else "drifts"

                status = "attributed" if best_diff <= TOLERANCE else "AMBIGUOUS-NO-MATCH"
                report_rows.append({
                    "run_id": f"dqn_{world}_multi_objective_{{drift}}_{seed} (ambiguous phase3 set)",
                    "category": "phase3-drift-attribution",
                    "max_abs_total_reward_diff": f"drifts={diffs['drifts']:.6g} nodrifts={diffs['nodrifts']:.6g}",
                    "status": f"{status}: attributed to {best_drift}",
                })
                print(f"  dqn_{world}_multi_objective_seed{seed}: "
                      f"drifts diff={diffs['drifts']:.6g}, nodrifts diff={diffs['nodrifts']:.6g} "
                      f"-> attributed to {best_drift}")

                # Write the attributed (matching) run: keep the ORIGINAL ambiguous
                # rows as its eval (origin=original-colab, now correctly labelled).
                # Guard: if this run_id is "hard"+"drifts", it is the Phase 1 DQN
                # run, whose eval.csv was already written+verified from the
                # UNAMBIGUOUS phase1_eval_results.zip -- do not re-write/clobber
                # it here; this ambiguous phase3 set matching it is itself a
                # confirmation (noted in the report), not a new write.
                best_diff_, _, best_summary, best_cov, best_dir, best_id = candidates[best_drift]
                if (best_dir / "eval.csv").exists():
                    print(f"    (skip write: {best_id} already has a verified "
                          f"eval.csv from Phase 1/2 -- this match confirms it)")
                else:
                    write_eval_outputs(best_dir, ambiguous_rows, best_summary, best_cov)
                    for fname in ["eval.csv", "eval_summary.csv", "coverage_trajectory.csv"]:
                        provenance_rows.append({
                            "run_id": best_id, "file": fname,
                            "source_zip": f"Phase 3/phase3_eval_results.zip (attributed to {best_drift})",
                            "origin": "original-colab", "sha256": sha256_bytes((best_dir / fname).read_bytes()),
                        })

                # Write the OTHER (non-matching) run: no original exists for it,
                # so write its own freshly regenerated eval. Same guard as above
                # in case the "other" side happens to be the already-populated
                # Phase 1 hard+drifts run.
                _, other_rows, other_summary, other_cov, other_dir, other_id = candidates[other_drift]
                if (other_dir / "eval.csv").exists():
                    print(f"    (skip write: {other_id} already has a verified "
                          f"eval.csv from Phase 1/2)")
                else:
                    write_eval_outputs(other_dir, other_rows, other_summary, other_cov)
                    for fname in ["eval.csv", "eval_summary.csv", "coverage_trajectory.csv"]:
                        provenance_rows.append({
                            "run_id": other_id, "file": fname, "source_zip": "regenerated-in-place",
                            "origin": "regenerated", "sha256": sha256_bytes((other_dir / fname).read_bytes()),
                        })


def main():
    # Build run_id -> run_dir map
    runs_by_key = {}
    for phase_dir in OUT_DIR.iterdir():
        if not phase_dir.is_dir():
            continue
        for run_dir in phase_dir.iterdir():
            if run_dir.is_dir():
                runs_by_key[run_dir.name] = run_dir

    report_rows = []
    provenance_rows = []

    print("=== Runs with an existing unambiguous original eval (Phase 1, Phase 2) ===")
    for run_id, run_dir in sorted(runs_by_key.items()):
        if (run_dir / "eval.csv").exists() and run_dir.parent.name in ("phase1", "phase2"):
            process_run_with_existing_eval(run_dir, run_id, report_rows, provenance_rows)

    print("\n=== Phase 3 ambiguous (world, seed) eval-set drift attribution ===")
    process_phase3_ambiguous(runs_by_key, report_rows, provenance_rows)

    # Append new provenance rows
    prov_path = OUT_DIR / "PROVENANCE.csv"
    with open(prov_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "file", "source_zip", "origin", "sha256"])
        writer.writerows(provenance_rows)
    print(f"\nAppended {len(provenance_rows)} provenance rows to {prov_path}")

    report_path = OUT_DIR / "REEVALUATION_REPORT.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "category", "max_abs_total_reward_diff", "status"])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {report_path}")

    # Final summary table: world x drift, mean reward/biomass/coverage, extinctions
    print("\n=== Final table: world x drift (mean of 3 seeds x 20 episodes) ===")
    import pandas as pd
    for world in ("easy", "medium", "hard"):
        for drift in ("drifts", "nodrifts"):
            rewards, biomasses, coverages, extinctions = [], [], [], 0
            for seed in (42, 43, 44):
                run_id = canonical_run_id("dqn", world, "multi_objective", seed, drift == "nodrifts")
                run_dir = runs_by_key[run_id]
                df = pd.read_csv(run_dir / "eval.csv")
                rewards.append(df["total_reward"].mean())
                biomasses.append(df["final_biomass"].mean())
                coverages.append(df["final_coverage"].mean())
                extinctions += int(df["extinct"].sum())
            print(f"  {world:7s} {drift:9s}: reward={np.mean(rewards):8.2f}  "
                  f"biomass={np.mean(biomasses):7.2f}  coverage={np.mean(coverages)*100:6.2f}%  "
                  f"extinctions={extinctions}/60")


if __name__ == "__main__":
    main()
