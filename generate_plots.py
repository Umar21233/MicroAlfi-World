"""
generate_plots.py -- Generate all paper figures from training and evaluation CSVs.
Run from the Internship folder with venv activated:
    python generate_plots.py
Saves all figures to plots/
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

RESULTS = "results"
PLOTS   = "plots"
os.makedirs(PLOTS, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "dqn":        "#1F78B4",
    "ppo":        "#33A02C",
    "a2c":        "#E31A1C",
    "fixed":      "#FF7F00",
    "random":     "#888888",
    "multi":      "#1F78B4",
    "biomass":    "#33A02C",
    "turbidity":  "#E31A1C",
    "turb_delta": "#FF7F00",
    "efficiency": "#6A3D9A",
    "sustain":    "#B15928",
    "nodrifts":   "#1F78B4",
    "drifts":     "#E31A1C",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def save(name):
    path = os.path.join(PLOTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def load_learning_curve(pattern):
    """Average learning curves across seeds matching glob pattern."""
    files = glob.glob(os.path.join(RESULTS, pattern))
    if not files:
        return None, None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "timesteps" in df.columns and "mean_reward" in df.columns:
                dfs.append(df.set_index("timesteps")["mean_reward"])
        except Exception:
            pass
    if not dfs:
        return None, None
    combined = pd.concat(dfs, axis=1)
    return combined.mean(axis=1), combined.std(axis=1)


# =============================================================================
# FIGURE 1 -- Phase 1: Learning curves (DQN, PPO, A2C)
# =============================================================================
print("Figure 1: Phase 1 learning curves")
fig, ax = plt.subplots(figsize=(8, 4.5))

for algo, colour, label in [
    ("ppo", C["ppo"], "PPO"),
    ("dqn", C["dqn"], "DQN"),
    ("a2c", C["a2c"], "A2C"),
]:
    mean, std = load_learning_curve(f"{algo}_hard_multi_objective_*_learning_curve.csv")
    if mean is not None:
        ax.plot(mean.index, mean.values, color=colour, label=label, linewidth=1.8)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=colour)

ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Phase 1: Learning Curves (hard world, multi_objective, with drifts)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("fig1_phase1_learning_curves.png")

# =============================================================================
# FIGURE 2 -- Phase 1: Evaluation bar chart
# =============================================================================
print("Figure 2: Phase 1 evaluation comparison")

algos  = ["DQN", "PPO", "A2C", "Fixed SP", "Random"]
reward = [5221,  2836,  1180,  4200,        -40]
bio    = [497,   275,   96,    365,          10]
cov    = [100,   94.3,  91.0,  85,           45]

x = np.arange(len(algos))
colours = [C["dqn"], C["ppo"], C["a2c"], C["fixed"], C["random"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].bar(x, reward, color=colours, width=0.55, edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(algos)
axes[0].set_ylabel("Mean Evaluation Reward"); axes[0].set_title("Evaluation Reward")
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(x, bio, color=colours, width=0.55, edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(algos)
axes[1].set_ylabel("Mean Final Biomass"); axes[1].set_title("Final Biomass")
axes[1].grid(True, axis="y", alpha=0.3)

axes[2].bar(x, cov, color=colours, width=0.55, edgecolor="white")
axes[2].set_xticks(x); axes[2].set_xticklabels(algos)
axes[2].set_ylabel("Mean Final Coverage (%)"); axes[2].set_title("Colony Coverage")
axes[2].set_ylim(0, 115)
axes[2].grid(True, axis="y", alpha=0.3)

fig.suptitle("Phase 1: Algorithm Comparison (hard world, multi_objective, with drifts)", fontsize=11)
plt.tight_layout()
save("fig2_phase1_evaluation.png")

# =============================================================================
# FIGURE 3 -- Phase 2: Reward mode comparison bars
# =============================================================================
print("Figure 3: Phase 2 reward modes")

modes_labels = ["multi_obj\n(P1)", "sustain", "biomass\nonly", "efficiency", "turbidity", "turb\ndelta"]
bio_p2  = [497, 460, 473, 26,  43,  54]
cov_p2  = [100, 98.7, 100, 46.8, 86.9, 89.6]
rewards_main = [5198, 4719, 4638, 2019, 1592]  # turbidity_delta excluded (different scale)
modes_main_labels = ["multi_obj\n(P1)", "sustain", "biomass\nonly", "efficiency", "turbidity"]
colours_p2 = [C["multi"], C["sustain"], C["biomass"], C["efficiency"], C["turbidity"], C["turb_delta"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

x_main = np.arange(len(modes_main_labels))
axes[0].bar(x_main, rewards_main, color=colours_p2[:5], width=0.55, edgecolor="white")
axes[0].set_xticks(x_main); axes[0].set_xticklabels(modes_main_labels)
axes[0].set_ylabel("Mean Training Reward")
axes[0].set_title("Training Reward by Reward Mode\n(turbidity_delta excluded — different scale)")
axes[0].grid(True, axis="y", alpha=0.3)

x2 = np.arange(len(modes_labels))
axes[1].bar(x2, bio_p2, color=colours_p2, width=0.55, edgecolor="white")
axes[1].set_xticks(x2); axes[1].set_xticklabels(modes_labels)
axes[1].set_ylabel("Mean Final Biomass")
axes[1].set_title("Final Biomass by Reward Mode (all six modes)")
axes[1].grid(True, axis="y", alpha=0.3)

fig.suptitle("Phase 2: Reward Mode Comparison (DQN, hard world, with drifts)", fontsize=11)
plt.tight_layout()
save("fig3_phase2_reward_modes.png")

# =============================================================================
# FIGURE 4 -- Phase 2: Coverage trajectories
# =============================================================================
print("Figure 4: Phase 2 coverage trajectories")

fig, ax = plt.subplots(figsize=(8, 4.5))

traj_configs = [
    ("dqn_hard_multi_objective_*_coverage_trajectory.csv",  C["multi"],     "multi_objective"),
    ("dqn_hard_biomass_only_*_coverage_trajectory.csv",     C["biomass"],   "biomass_only"),
    ("dqn_hard_sustainability_*_coverage_trajectory.csv",   C["sustain"],   "sustainability"),
    ("dqn_hard_turbidity_delta_*_coverage_trajectory.csv",  C["turb_delta"],"turbidity_delta"),
    ("dqn_hard_turbidity_*_coverage_trajectory.csv",        C["turbidity"], "turbidity"),
]

for pattern, colour, label in traj_configs:
    files = glob.glob(os.path.join(RESULTS, pattern))
    # Exclude turbidity_delta matches when searching plain turbidity
    if pattern == "dqn_hard_turbidity_*_coverage_trajectory.csv":
        files = [f for f in files if "turbidity_delta" not in f]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "mean_coverage" in df.columns:
                dfs.append(df["mean_coverage"].values)
        except Exception:
            pass
    if dfs:
        lengths = [len(d) for d in dfs]
        min_len = min(lengths)
        arr  = np.array([d[:min_len] for d in dfs])
        mean = arr.mean(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean * 100, color=colour, label=label, linewidth=1.8)

ax.set_xlabel("Episode Step")
ax.set_ylabel("Mean Colony Coverage (%)")
ax.set_title("Phase 2: Colony Coverage Trajectory by Reward Mode")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("fig4_phase2_coverage_trajectories.png")

# =============================================================================
# FIGURE 5 -- Phase 3: DQN vs Fixed Setpoint
# =============================================================================
print("Figure 5: Phase 3 DQN vs Fixed Setpoint")

configs   = ["easy\nno drift", "easy\nwith drift", "medium\nno drift",
             "medium\nwith drift", "hard\nno drift", "hard\nwith drift"]
dqn_rew   = [4574, 5303, 4806, 5133, 4958, 5221]
fixed_rew = [4023, 3898, 4229, 4075, 4075, 3949]
advantage = [d - f for d, f in zip(dqn_rew, fixed_rew)]

x     = np.arange(len(configs))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].bar(x - width/2, dqn_rew,   width, label="DQN",           color=C["dqn"],   edgecolor="white")
axes[0].bar(x + width/2, fixed_rew, width, label="Fixed Setpoint", color=C["fixed"], edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(configs, fontsize=8)
axes[0].set_ylabel("Mean Evaluation Reward")
axes[0].set_title("DQN vs Fixed Setpoint: Evaluation Reward")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

bar_colours = [C["nodrifts"], C["drifts"]] * 3
axes[1].bar(x, advantage, color=bar_colours, edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(configs, fontsize=8)
axes[1].set_ylabel("DQN Advantage (reward gap)")
axes[1].set_title("DQN Advantage over Fixed Setpoint")
axes[1].grid(True, axis="y", alpha=0.3)
nd_patch = mpatches.Patch(color=C["nodrifts"], label="No drifts")
wd_patch = mpatches.Patch(color=C["drifts"],   label="With drifts")
axes[1].legend(handles=[nd_patch, wd_patch])

fig.suptitle("Phase 3: World Difficulty and Drift Condition Comparison", fontsize=11)
plt.tight_layout()
save("fig5_phase3_dqn_vs_baseline.png")

# =============================================================================
# FIGURE 6 -- Phase 3: Biomass and coverage by world preset
# =============================================================================
print("Figure 6: Phase 3 difficulty and drift effect")

presets = ["easy", "medium", "hard"]
nd_bio  = [425,  459,  469]
wd_bio  = [497,  487,  497]
nd_cov  = [99.4, 100,  99.9]
wd_cov  = [100,  100,  100]

x     = np.arange(len(presets))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].bar(x - width/2, nd_bio, width, label="No drifts",   color=C["nodrifts"], edgecolor="white")
axes[0].bar(x + width/2, wd_bio, width, label="With drifts", color=C["drifts"],   edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(presets)
axes[0].set_ylabel("Mean Final Biomass")
axes[0].set_title("Final Biomass by World Preset")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(x - width/2, nd_cov, width, label="No drifts",   color=C["nodrifts"], edgecolor="white")
axes[1].bar(x + width/2, wd_cov, width, label="With drifts", color=C["drifts"],   edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(presets)
axes[1].set_ylabel("Mean Final Coverage (%)")
axes[1].set_title("Colony Coverage by World Preset")
axes[1].set_ylim(95, 101)
axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

fig.suptitle("Phase 3: DQN Performance Across World Presets and Drift Conditions", fontsize=11)
plt.tight_layout()
save("fig6_phase3_difficulty.png")

# =============================================================================
# FIGURE 7 -- Phase 1 per-algorithm learning curves (separate panels)
# =============================================================================
print("Figure 7: Per-algorithm learning curves with variance")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for idx, (algo, colour, label) in enumerate([
    ("dqn", C["dqn"], "DQN"),
    ("ppo", C["ppo"], "PPO"),
    ("a2c", C["a2c"], "A2C"),
]):
    mean, std = load_learning_curve(f"{algo}_hard_multi_objective_*_learning_curve.csv")
    if mean is not None:
        axes[idx].plot(mean.index, mean.values, color=colour, linewidth=1.8)
        axes[idx].fill_between(mean.index, mean - std, mean + std, alpha=0.25, color=colour)
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Mean Episode Reward")
        axes[idx].grid(True, alpha=0.3)
    else:
        axes[idx].set_title(f"{label} (no data)")
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Mean Episode Reward")

fig.suptitle("Phase 1: Per-Algorithm Learning Curves with Variance", fontsize=11)
plt.tight_layout()
save("fig7_phase1_per_algo_curves.png")

print("\nAll figures saved to plots/")
