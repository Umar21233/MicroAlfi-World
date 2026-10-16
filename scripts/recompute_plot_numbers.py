"""
recompute_plot_numbers.py -- Audit tool: recompute the hardcoded numbers used
in generate_plots.py (Figures 2, 3, 5, 6) directly from the CSVs in results/,
and (for the two baseline agents, which are deterministic/random and involve
no training) by re-running them fresh.

This script is READ-ONLY with respect to results/, trained_models/ and plots/:
it never writes into those directories. It is part of the audit paper trail,
not part of the frozen training/evaluation pipeline.

Usage (from project root):
    python scripts/recompute_plot_numbers.py
"""

import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2
from agents.random_agent_v2 import WORLD_PRESETS
from agents.fixed_setpoint_agent_v2 import fixed_setpoint_policy_v2

RESULTS = os.path.join(PROJECT_ROOT, "results")

EVAL_SEED_START = 100   # same protocol evaluate.py uses for RL models
N_EVAL_EPISODES = 20


def eval_summary_mean(run_name: str, col: str):
    """Read the 'mean' row of a stored *_eval_summary.csv for one run."""
    path = os.path.join(RESULTS, f"{run_name}_eval_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["stat"] == "mean"]
    if row.empty or col not in row.columns:
        return None
    return float(row[col].iloc[0])


def avg_over_seeds(algo, world, reward_mode, seeds, col):
    """Average an eval_summary column over several training seeds."""
    vals = []
    for s in seeds:
        v = eval_summary_mean(f"{algo}_{world}_{reward_mode}_{s}", col)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals)), len(vals)


def last_learning_curve_reward(run_name: str):
    """Last logged mean_reward in a *_learning_curve.csv (end-of-training reward)."""
    path = os.path.join(RESULTS, f"{run_name}_learning_curve.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df["mean_reward"].iloc[-1])


def avg_training_reward_over_seeds(algo, world, reward_mode, seeds):
    vals = []
    for s in seeds:
        v = last_learning_curve_reward(f"{algo}_{world}_{reward_mode}_{s}")
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals)), len(vals)


def run_baseline_fresh(policy, world_preset, reward_mode, no_drifts=False,
                        seed_start=EVAL_SEED_START, n_episodes=N_EVAL_EPISODES):
    """
    Run a frozen, deterministic/random baseline policy fresh using the same
    seed protocol evaluate.py uses for RL models (seeds 100..119). This is
    NOT a trained model -- it is either uniform-random action sampling or the
    fixed priority-ordered controller, so re-running it does not touch any
    "no retrain" boundary.

    policy: either "random" or "fixed"
    """
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS[world_preset],
        reward_mode=reward_mode,
        no_drifts=no_drifts,
    )
    rewards, coverages, biomasses = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        ep_reward = 0.0
        done = False
        while not done:
            if policy == "random":
                action = env.action_space.sample()
            else:
                action = fixed_setpoint_policy_v2(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        coverages.append(float(obs[9]))
        biomasses.append(info["total_biomass"])
    env.close()
    return {
        "reward":   float(np.mean(rewards)),
        "coverage": float(np.mean(coverages)) * 100,
        "biomass":  float(np.mean(biomasses)),
    }


def section(title):
    print(f"\n{'='*90}\n  {title}\n{'='*90}")


def row(label, hardcoded, recomputed, note=""):
    if recomputed is None:
        rec_str = "NO DATA"
    elif isinstance(recomputed, tuple):
        rec_str = f"{recomputed[0]:.2f} (n={recomputed[1]})"
    else:
        rec_str = f"{recomputed:.2f}"
    hc_str = f"{hardcoded}" if hardcoded is not None else "-"
    print(f"  {label:<38}{hc_str:>14}{rec_str:>22}   {note}")


if __name__ == "__main__":
    SEEDS = [42, 43, 44]

    # ── Figure 2: Phase 1 evaluation (hard world, multi_objective, with drifts) ──
    section("FIGURE 2 -- Phase 1 evaluation (hard, multi_objective, with drifts)")
    print(f"  {'metric':<38}{'hardcoded':>14}{'recomputed':>22}   note")
    print(f"  {'-'*86}")

    for algo, label in [("dqn", "DQN"), ("ppo", "PPO"), ("a2c", "A2C")]:
        r = avg_over_seeds(algo, "hard", "multi_objective", SEEDS, "total_reward")
        b = avg_over_seeds(algo, "hard", "multi_objective", SEEDS, "final_biomass")
        c = avg_over_seeds(algo, "hard", "multi_objective", SEEDS, "final_coverage")
        hc_idx = {"dqn": 0, "ppo": 1, "a2c": 2}[algo]
        hc_reward = [5221, 2836, 1180][hc_idx]
        hc_bio    = [497, 275, 96][hc_idx]
        hc_cov    = [100, 94.3, 91.0][hc_idx]
        row(f"{label} reward",   hc_reward, r)
        row(f"{label} biomass",  hc_bio,    b)
        row(f"{label} coverage(%)", hc_cov, (c[0]*100, c[1]) if c else None)

    fixed = run_baseline_fresh("fixed", "hard", "multi_objective")
    random_ = run_baseline_fresh("random", "hard", "multi_objective")
    row("Fixed SP reward",   4200, fixed["reward"],   "freshly run, seeds 100-119")
    row("Fixed SP biomass",  365,  fixed["biomass"],  "freshly run, seeds 100-119")
    row("Fixed SP coverage(%)", 85, fixed["coverage"], "freshly run, seeds 100-119")
    row("Random reward",     -40,  random_["reward"], "freshly run, seeds 100-119")
    row("Random biomass",    10,   random_["biomass"], "freshly run, seeds 100-119")
    row("Random coverage(%)", 45,  random_["coverage"], "freshly run, seeds 100-119")

    # ── Figure 3: Phase 2 reward mode comparison (DQN, hard, with drifts) ──
    section("FIGURE 3 -- Phase 2 reward modes (DQN, hard world, with drifts)")
    print(f"  {'metric':<38}{'hardcoded':>14}{'recomputed':>22}   note")
    print(f"  {'-'*86}")

    modes = ["multi_objective", "sustainability", "biomass_only", "efficiency",
              "turbidity", "turbidity_delta"]
    hc_bio_p2 = {"multi_objective": 497, "sustainability": 460, "biomass_only": 473,
                 "efficiency": 26, "turbidity": 43, "turbidity_delta": 54}
    hc_cov_p2 = {"multi_objective": 100, "sustainability": 98.7, "biomass_only": 100,
                 "efficiency": 46.8, "turbidity": 86.9, "turbidity_delta": 89.6}
    hc_train_reward = {"multi_objective": 5198, "sustainability": 4719,
                        "biomass_only": 4638, "efficiency": 2019, "turbidity": 1592}

    for mode in modes:
        train_r = avg_training_reward_over_seeds("dqn", "hard", mode, SEEDS)
        eval_r  = avg_over_seeds("dqn", "hard", mode, SEEDS, "total_reward")
        bio     = avg_over_seeds("dqn", "hard", mode, SEEDS, "final_biomass")
        cov     = avg_over_seeds("dqn", "hard", mode, SEEDS, "final_coverage")
        row(f"{mode} TRAINING reward", hc_train_reward.get(mode), train_r,
            "from *_learning_curve.csv (last logged point)")
        row(f"{mode} EVAL reward", None, eval_r,
            "from *_eval_summary.csv (not separately hardcoded)")
        row(f"{mode} biomass",  hc_bio_p2[mode], bio)
        row(f"{mode} coverage(%)", hc_cov_p2[mode], (cov[0]*100, cov[1]) if cov else None)

    # ── Figure 5 & 6: Phase 3 world x drift ──────────────────────────────────
    section("FIGURE 5/6 -- Phase 3 world preset x drift condition (DQN)")
    print(f"  {'metric':<38}{'hardcoded':>14}{'recomputed':>22}   note")
    print(f"  {'-'*86}")

    presets = ["easy", "medium", "hard"]
    hc_dqn_rew_drift   = {"easy": 5303, "medium": 5133, "hard": 5221}
    hc_dqn_rew_nodrift = {"easy": 4574, "medium": 4806, "hard": 4958}
    hc_fixed_rew_drift   = {"easy": 3898, "medium": 4075, "hard": 3949}
    hc_fixed_rew_nodrift = {"easy": 4023, "medium": 4229, "hard": 4075}
    hc_wd_bio = {"easy": 497, "medium": 487, "hard": 497}
    hc_nd_bio = {"easy": 425, "medium": 459, "hard": 469}
    hc_wd_cov = {"easy": 100, "medium": 100, "hard": 100}
    hc_nd_cov = {"easy": 99.4, "medium": 100, "hard": 99.9}

    for preset in presets:
        rew = avg_over_seeds("dqn", preset, "multi_objective", SEEDS, "total_reward")
        bio = avg_over_seeds("dqn", preset, "multi_objective", SEEDS, "final_biomass")
        cov = avg_over_seeds("dqn", preset, "multi_objective", SEEDS, "final_coverage")
        row(f"DQN {preset} reward (WITH drifts)",  hc_dqn_rew_drift[preset], rew,
            "matches stored run_name convention (no_drifts not in filename)")
        row(f"DQN {preset} biomass (WITH drifts)", hc_wd_bio[preset], bio)
        row(f"DQN {preset} coverage(%) (WITH drifts)", hc_wd_cov[preset],
            (cov[0]*100, cov[1]) if cov else None)
        row(f"DQN {preset} reward (NO drifts)", hc_dqn_rew_nodrift[preset], None,
            "NO stored eval CSV exists for a no_drifts seed-42/43/44 run of this "
            "preset -- run_name has no no_drifts marker, so it cannot be "
            "distinguished from the with-drifts run even if it existed on disk")
        row(f"DQN {preset} biomass (NO drifts)", hc_nd_bio[preset], None, "see above")
        row(f"DQN {preset} coverage(%) (NO drifts)", hc_nd_cov[preset], None, "see above")

        fixed_d  = run_baseline_fresh("fixed", preset, "multi_objective", no_drifts=False)
        fixed_nd = run_baseline_fresh("fixed", preset, "multi_objective", no_drifts=True)
        row(f"Fixed SP {preset} reward (WITH drifts)", hc_fixed_rew_drift[preset],
            fixed_d["reward"], "freshly run, seeds 100-119")
        row(f"Fixed SP {preset} reward (NO drifts)", hc_fixed_rew_nodrift[preset],
            fixed_nd["reward"], "freshly run, seeds 100-119 (fixed-setpoint agent "
            "IS frozen/deterministic, so no_drifts can be re-run safely)")

    print()
