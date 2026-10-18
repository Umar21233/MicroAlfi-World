"""
mechanism_diagnostics.py -- Drift study Part A.5.

Hard world only, one model per condition (seed 42): the drift-trained DQN
(results_verified/phase1/dqn_hard_multi_objective_drifts_42) and the
no-drift-trained DQN (results_verified/phase3/dqn_hard_multi_objective_nodrifts_42).
Each is evaluated in BOTH environments (drifts, nodrifts) using the
project's existing protocol (20 episodes, seeds 100-119, deterministic),
logging the per-step mean of light, co2, nutrients, ph, toxin, coverage,
and the overall action distribution (11 actions) -- 4 (policy, environment)
combinations in total.

Outputs:
    results_verified/drift_study/MECHANISM_TIMESERIES.csv
        columns: policy, environment, step, light, co2, nutrients, ph,
        toxin, coverage, n_episodes_reaching
    results_verified/drift_study/MECHANISM_ACTION_DISTRIBUTION.csv
        columns: policy, environment, action, fraction
    plots_verified/drift_study/fig_mechanism_timeseries.png
    plots_verified/drift_study/fig_mechanism_action_distribution.png

Usage (from project root):
    venv\\Scripts\\python.exe experiments\\drift_study\\mechanism_diagnostics.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.alphae_world_env_v2 import AlphaeWorldEnvV2  # noqa: E402
from agents.random_agent_v2 import WORLD_PRESETS  # noqa: E402
from training.evaluate import N_EPISODES, SEED_START  # noqa: E402

VERIFIED = PROJECT_ROOT / "results_verified"
OUT_DIR = VERIFIED / "drift_study"
PLOTS_DIR = PROJECT_ROOT / "plots_verified" / "drift_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = {
    "drift_trained":   VERIFIED / "phase1" / "dqn_hard_multi_objective_drifts_42" / "model.zip",
    "nodrift_trained": VERIFIED / "phase3" / "dqn_hard_multi_objective_nodrifts_42" / "model.zip",
}
ENVIRONMENTS = {
    "drifts":   False,  # no_drifts=False
    "nodrifts": True,
}
METRICS = ["light", "co2", "nutrients", "ph", "toxin", "coverage"]
N_ACTIONS = 11


def load_model(path: Path) -> DQN:
    return DQN.load(io.BytesIO(path.read_bytes()))


def run_diagnostic(model: DQN, eval_no_drifts: bool):
    env = AlphaeWorldEnvV2(
        world_config=WORLD_PRESETS["hard"],
        reward_mode="multi_objective",
        no_drifts=eval_no_drifts,
    )
    per_step = {}
    action_counts = np.zeros(N_ACTIONS, dtype=int)

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=SEED_START + ep)
        done = False
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            g = env.global_state
            per_step.setdefault(step_idx, []).append({
                "light": float(g[0]), "co2": float(g[2]), "nutrients": float(g[3]),
                "ph": float(g[4]), "toxin": float(g[5]),
                "coverage": info["colony_coverage"],
            })
            step_idx += 1
            done = terminated or truncated

    env.close()

    rows = []
    for step in sorted(per_step.keys()):
        vals = per_step[step]
        row = {"step": step, "n_episodes_reaching": len(vals)}
        for key in METRICS:
            row[key] = round(float(np.mean([v[key] for v in vals])), 4)
        rows.append(row)

    action_frac = (action_counts / action_counts.sum()).tolist()
    return rows, action_frac


def main():
    timeseries_rows = []
    action_rows = []

    for policy_name, model_path in POLICIES.items():
        print(f"Loading {model_path.relative_to(PROJECT_ROOT)}")
        model = load_model(model_path)
        for env_name, eval_no_drifts in ENVIRONMENTS.items():
            print(f"  Running {policy_name} in {env_name} environment ...")
            rows, action_frac = run_diagnostic(model, eval_no_drifts)
            for r in rows:
                timeseries_rows.append({"policy": policy_name, "environment": env_name, **r})
            for a, frac in enumerate(action_frac):
                action_rows.append({
                    "policy": policy_name, "environment": env_name,
                    "action": a, "fraction": round(frac, 4),
                })

    ts_df = pd.DataFrame(timeseries_rows)
    ts_df.to_csv(OUT_DIR / "MECHANISM_TIMESERIES.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'MECHANISM_TIMESERIES.csv'} ({len(ts_df)} rows)")

    act_df = pd.DataFrame(action_rows)
    act_df.to_csv(OUT_DIR / "MECHANISM_ACTION_DISTRIBUTION.csv", index=False)
    print(f"Wrote {OUT_DIR / 'MECHANISM_ACTION_DISTRIBUTION.csv'} ({len(act_df)} rows)")

    # -- Plot 1: per-step timeseries, one panel per metric -------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    colours = {
        ("drift_trained", "drifts"):     "#1F78B4",
        ("drift_trained", "nodrifts"):   "#A6CEE3",
        ("nodrift_trained", "drifts"):   "#E31A1C",
        ("nodrift_trained", "nodrifts"): "#FB9A99",
    }
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 3][idx % 3]
        for (policy_name, env_name), colour in colours.items():
            sub = ts_df[(ts_df.policy == policy_name) & (ts_df.environment == env_name)]
            ax.plot(sub["step"], sub[metric], color=colour, linewidth=1.5,
                     label=f"{policy_name}, env={env_name}")
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Drift study: per-step mean global state and coverage "
                 "(hard world, DQN seed 42, 20 eval episodes)", fontsize=11)
    plt.tight_layout()
    fig_path = PLOTS_DIR / "fig_mechanism_timeseries.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # -- Plot 2: action distribution bar chart -------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    x = np.arange(N_ACTIONS)
    combos = list(colours.keys())
    for i, (policy_name, env_name) in enumerate(combos):
        sub = act_df[(act_df.policy == policy_name) & (act_df.environment == env_name)]
        sub = sub.sort_values("action")
        ax.bar(x + (i - 1.5) * width, sub["fraction"], width,
               label=f"{policy_name}, env={env_name}", color=colours[(policy_name, env_name)])
    ax.set_xticks(x)
    ax.set_xlabel("action")
    ax.set_ylabel("fraction of steps")
    ax.set_title("Drift study: action distribution (hard world, DQN seed 42, 20 eval episodes)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = PLOTS_DIR / "fig_mechanism_action_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
