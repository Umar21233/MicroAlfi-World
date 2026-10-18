"""
build_eval_summary_all.py -- Read-only aggregation tool (Step 3 deliverable).

Walks every run directory under results_verified/phase{1,2,3}/, reads each
run's config.txt + eval.csv (20 episodes each), and writes two CSVs:

  results_verified/EVAL_SUMMARY_PER_RUN.csv
      One row per run_id (39 rows): algo, world, reward_mode, drift, seed,
      mean_reward, mean_biomass, mean_coverage_pct, extinctions (out of 20).

  results_verified/EVAL_SUMMARY_ALL.csv
      One row per (algo, world, reward_mode, drift) group, averaged over its
      3 seeds (60 episodes): mean_reward, mean_biomass, mean_coverage_pct,
      extinctions (out of 60), n_seeds.

Does not modify any run directory. Run from the project root:
    venv\\Scripts\\python.exe scripts\\build_eval_summary_all.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_naming import drift_token  # noqa: E402

VERIFIED = PROJECT_ROOT / "results_verified"


def read_config(path: Path) -> dict:
    cfg = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def main():
    rows = []
    for phase_dir in sorted(VERIFIED.glob("phase*")):
        if not phase_dir.is_dir():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            cfg_path = run_dir / "config.txt"
            eval_path = run_dir / "eval.csv"
            if not cfg_path.exists() or not eval_path.exists():
                continue
            cfg = read_config(cfg_path)
            no_drifts = cfg["no_drifts"] == "True"
            ev = pd.read_csv(eval_path)
            rows.append({
                "run_id": run_dir.name,
                "algo": cfg["algo"],
                "world": cfg["world"],
                "reward_mode": cfg["reward_mode"],
                "drift": drift_token(no_drifts),
                "seed": int(cfg["seed"]),
                "mean_reward": ev["total_reward"].mean(),
                "mean_biomass": ev["final_biomass"].mean(),
                "mean_coverage_pct": ev["final_coverage"].mean() * 100,
                "extinctions": int(ev["extinct"].astype(str).isin(["True", "1", "1.0"]).sum()),
                "n_episodes": len(ev),
            })

    per_run = pd.DataFrame(rows).sort_values(["algo", "world", "reward_mode", "drift", "seed"])
    per_run.to_csv(VERIFIED / "EVAL_SUMMARY_PER_RUN.csv", index=False)
    print(f"Wrote {VERIFIED / 'EVAL_SUMMARY_PER_RUN.csv'} ({len(per_run)} runs)")

    grouped = per_run.groupby(["algo", "world", "reward_mode", "drift"]).agg(
        mean_reward=("mean_reward", "mean"),
        mean_biomass=("mean_biomass", "mean"),
        mean_coverage_pct=("mean_coverage_pct", "mean"),
        extinctions=("extinctions", "sum"),
        n_seeds=("seed", "count"),
        total_episodes=("n_episodes", "sum"),
    ).reset_index()
    grouped.to_csv(VERIFIED / "EVAL_SUMMARY_ALL.csv", index=False)
    print(f"Wrote {VERIFIED / 'EVAL_SUMMARY_ALL.csv'} ({len(grouped)} groups)")

    print("\n=== Final table: world x drift (mean over seeds x 20 episodes) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
