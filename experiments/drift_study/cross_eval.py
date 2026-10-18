"""
cross_eval.py -- Drift study Part A.1-A.4.

For each of the 18 DQN multi_objective models (easy/medium/hard world x
drifts/nodrifts trained_with x seeds 42/43/44 -- the hard/drifts models
are the Phase 1 DQN models), loads the model from results_verified/ and
evaluates it in BOTH environments (no_drifts=False and True, same world
preset), using the project's existing evaluation protocol unchanged
(training.evaluate.run_evaluation: 20 episodes, seeds 100-119,
deterministic=True).

Never writes into results_verified/phase{1,2,3}/ or anywhere else under
results_verified/ except results_verified/drift_study/ (new).

Outputs:
    results_verified/drift_study/CROSS_EVAL.csv
        One row per (world, trained_with, evaluated_in, seed): 36 rows.
    results_verified/drift_study/CROSS_EVAL_SUMMARY.csv
        One row per (world, trained_with, evaluated_in), averaged over the
        3 seeds: 12 rows -- the 2x2 table per world, flattened.

Also prints the 2x2 table per world (plus the fixed-setpoint baseline in
both environments, read from results_verified/baselines/BASELINES_SUMMARY.csv
for reference) and the Step 4 sanity check against EVAL_SUMMARY_ALL.csv.

Usage (from project root):
    venv\\Scripts\\python.exe experiments\\drift_study\\cross_eval.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.evaluate import run_evaluation, N_EPISODES, SEED_START  # noqa: E402

VERIFIED = PROJECT_ROOT / "results_verified"
OUT_DIR = VERIFIED / "drift_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORLDS = ["easy", "medium", "hard"]
SEEDS = [42, 43, 44]

# (trained_with, no_drifts_trained, phase-dir-per-world)
TRAINED_WITH = [
    ("drifts", False),
    ("nodrifts", True),
]


def model_path(world: str, drift_token: str, seed: int) -> Path:
    phase = "phase1" if (world == "hard" and drift_token == "drifts") else "phase3"
    run_id = f"dqn_{world}_multi_objective_{drift_token}_{seed}"
    return VERIFIED / phase / run_id / "model.zip"


def load_model(path: Path) -> DQN:
    data = path.read_bytes()
    return DQN.load(io.BytesIO(data))


def evaluate_one(model: DQN, world: str, eval_no_drifts: bool) -> dict:
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world],
        reward_mode="multi_objective",
        no_drifts=eval_no_drifts,
    )
    episode_rows, _ = run_evaluation(model, env, N_EPISODES, SEED_START, deterministic=True)
    env.close()
    rewards = [r["total_reward"] for r in episode_rows]
    biomasses = [r["final_biomass"] for r in episode_rows]
    coverages = [r["final_coverage"] for r in episode_rows]
    extinctions = sum(r["extinct"] for r in episode_rows)
    return {
        "mean_reward": round(float(np.mean(rewards)), 4),
        "mean_biomass": round(float(np.mean(biomasses)), 4),
        "mean_coverage": round(float(np.mean(coverages)), 4),
        "extinctions": int(extinctions),
    }


def main():
    rows = []
    for world in WORLDS:
        for trained_with, _ in TRAINED_WITH:
            for seed in SEEDS:
                mp = model_path(world, trained_with, seed)
                print(f"Loading {mp.relative_to(PROJECT_ROOT)}")
                model = load_model(mp)
                for evaluated_in, eval_no_drifts in [("drifts", False), ("nodrifts", True)]:
                    stats = evaluate_one(model, world, eval_no_drifts)
                    row = {
                        "world": world,
                        "trained_with": trained_with,
                        "evaluated_in": evaluated_in,
                        "seed": seed,
                        **stats,
                    }
                    rows.append(row)
                    print(f"  seed={seed} trained={trained_with:9s} "
                          f"evaluated_in={evaluated_in:9s} "
                          f"reward={stats['mean_reward']:.2f} "
                          f"biomass={stats['mean_biomass']:.2f} "
                          f"coverage={stats['mean_coverage']*100:.1f}% "
                          f"extinct={stats['extinctions']}/20")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "CROSS_EVAL.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'CROSS_EVAL.csv'} ({len(df)} rows)")

    summary = df.groupby(["world", "trained_with", "evaluated_in"]).agg(
        mean_reward=("mean_reward", "mean"),
        mean_biomass=("mean_biomass", "mean"),
        mean_coverage=("mean_coverage", "mean"),
        extinctions=("extinctions", "sum"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "CROSS_EVAL_SUMMARY.csv", index=False)
    print(f"Wrote {OUT_DIR / 'CROSS_EVAL_SUMMARY.csv'} ({len(summary)} rows)")

    # -- Fixed-setpoint baseline for reference (already computed under the
    # identical protocol; not re-run here) ----------------------------------
    baselines = pd.read_csv(VERIFIED / "baselines" / "BASELINES_SUMMARY.csv")

    # -- 2x2 tables per world -------------------------------------------------
    print("\n" + "=" * 70)
    print("2x2 TABLES: trained_with (rows) x evaluated_in (columns), mean reward")
    print("=" * 70)
    for world in WORLDS:
        sub = summary[summary.world == world]
        pivot = sub.pivot(index="trained_with", columns="evaluated_in", values="mean_reward")
        pivot = pivot.reindex(index=["drifts", "nodrifts"], columns=["drifts", "nodrifts"])
        print(f"\n-- {world} --")
        print(pivot.to_string())
        fs = baselines[(baselines.agent == "fixed_setpoint") & (baselines.world == world)]
        fs_drifts = fs[fs.drift == "drifts"]["mean_reward"].values
        fs_nodrifts = fs[fs.drift == "nodrifts"]["mean_reward"].values
        print(f"   (reference) fixed_setpoint: drifts={fs_drifts[0]:.2f}  "
              f"nodrifts={fs_nodrifts[0]:.2f}")

    # -- Step 4 sanity check: diagonal vs EVAL_SUMMARY_ALL.csv ----------------
    print("\n" + "=" * 70)
    print("SANITY CHECK: diagonal cells vs results_verified/EVAL_SUMMARY_ALL.csv")
    print("(tolerance: 0.5 reward)")
    print("=" * 70)
    eval_all = pd.read_csv(VERIFIED / "EVAL_SUMMARY_ALL.csv")
    ok = True
    for world in WORLDS:
        for trained_with in ["drifts", "nodrifts"]:
            diag = summary[(summary.world == world) & (summary.trained_with == trained_with)
                            & (summary.evaluated_in == trained_with)]
            cross_val = float(diag["mean_reward"].iloc[0])
            ref = eval_all[(eval_all.algo == "dqn") & (eval_all.world == world)
                           & (eval_all.reward_mode == "multi_objective")
                           & (eval_all.drift == trained_with)]
            if ref.empty:
                print(f"  {world}/{trained_with}: NO REFERENCE ROW in EVAL_SUMMARY_ALL.csv")
                ok = False
                continue
            ref_val = float(ref["mean_reward"].iloc[0])
            diff = abs(cross_val - ref_val)
            status = "OK" if diff <= 0.5 else "MISMATCH"
            if status == "MISMATCH":
                ok = False
            print(f"  {world}/{trained_with}: cross_eval={cross_val:.4f}  "
                  f"verified={ref_val:.4f}  diff={diff:.4f}  [{status}]")

    if not ok:
        print("\nSANITY CHECK FAILED -- stopping, do not trust CROSS_EVAL.csv until resolved.")
        sys.exit(1)
    print("\nSANITY CHECK PASSED: all diagonal cells reproduce EVAL_SUMMARY_ALL.csv within tolerance.")


if __name__ == "__main__":
    main()
