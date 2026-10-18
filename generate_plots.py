"""
generate_plots.py -- Generate all paper figures from results_verified/ CSVs.
Run from the Internship folder with venv activated:
    python generate_plots.py

Every plotted quantity is computed here, at run time, from
results_verified/ (learning_curve.csv, eval.csv -> EVAL_SUMMARY_ALL.csv,
coverage_trajectory.csv, baselines/BASELINES_SUMMARY.csv). There are no
hardcoded result numbers in this file. See VERIFICATION.md for how
results_verified/ was built and verified against the raw Colab zips, and
results_verified/PLOT_NUMBERS_OLD_VS_VERIFIED.csv for a comparison against
the old hand-maintained values this script used to contain.

Figures are written to plots_verified/ (the old plots/, built from the
unverified results/ folder, is left untouched).
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VERIFIED = os.path.join(PROJECT_ROOT, "results_verified")
PLOTS = os.path.join(PROJECT_ROOT, "plots_verified")
os.makedirs(PLOTS, exist_ok=True)

EVAL_ALL = pd.read_csv(os.path.join(VERIFIED, "EVAL_SUMMARY_ALL.csv"))
BASELINES = pd.read_csv(os.path.join(VERIFIED, "baselines", "BASELINES_SUMMARY.csv"))

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


def run_dirs(run_id_pattern):
    """Every results_verified/phase*/<run_id_pattern> directory."""
    return sorted(glob.glob(os.path.join(VERIFIED, "phase*", run_id_pattern)))


def load_learning_curve(run_id_pattern):
    """Average learning curves (training reward) across seeds matching pattern."""
    dirs = run_dirs(run_id_pattern)
    dfs = []
    for d in dirs:
        f = os.path.join(d, "learning_curve.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if "timesteps" in df.columns and "mean_reward" in df.columns:
            dfs.append(df.set_index("timesteps")["mean_reward"])
    if not dfs:
        return None, None
    combined = pd.concat(dfs, axis=1)
    return combined.mean(axis=1), combined.std(axis=1)


def final_training_reward(run_id_pattern):
    mean, _ = load_learning_curve(run_id_pattern)
    return float(mean.iloc[-1]) if mean is not None else None


def eval_lookup(algo, world, reward_mode, drift):
    row = EVAL_ALL[(EVAL_ALL.algo == algo) & (EVAL_ALL.world == world) &
                    (EVAL_ALL.reward_mode == reward_mode) & (EVAL_ALL.drift == drift)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"reward": r.mean_reward, "biomass": r.mean_biomass, "coverage": r.mean_coverage_pct}


def baseline_lookup(agent, world, drift):
    row = BASELINES[(BASELINES.agent == agent) & (BASELINES.world == world) & (BASELINES.drift == drift)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"reward": r.mean_reward, "biomass": r.mean_biomass, "coverage": r.mean_coverage_pct}


# =============================================================================
# OLD-vs-VERIFIED comparison table (Step 6.1 deliverable)
# =============================================================================
# The values below are exactly what the pre-verification generate_plots.py
# hardcoded (see git history / AUDIT.md Step 2f) -- kept here ONLY as a fixed
# reference for this one comparison table, never used for plotting.
_OLD = {
    ("fig2", "dqn", "reward"): 5221, ("fig2", "dqn", "biomass"): 497, ("fig2", "dqn", "coverage"): 100,
    ("fig2", "ppo", "reward"): 2836, ("fig2", "ppo", "biomass"): 275, ("fig2", "ppo", "coverage"): 94.3,
    ("fig2", "a2c", "reward"): 1180, ("fig2", "a2c", "biomass"): 96, ("fig2", "a2c", "coverage"): 91.0,
    ("fig2", "fixed", "reward"): 4200, ("fig2", "fixed", "biomass"): 365, ("fig2", "fixed", "coverage"): 85,
    ("fig2", "random", "reward"): -40, ("fig2", "random", "biomass"): 10, ("fig2", "random", "coverage"): 45,
    ("fig3", "multi_objective", "train_reward"): 5198, ("fig3", "multi_objective", "biomass"): 497, ("fig3", "multi_objective", "coverage"): 100,
    ("fig3", "sustainability", "train_reward"): 4719, ("fig3", "sustainability", "biomass"): 460, ("fig3", "sustainability", "coverage"): 98.7,
    ("fig3", "biomass_only", "train_reward"): 4638, ("fig3", "biomass_only", "biomass"): 473, ("fig3", "biomass_only", "coverage"): 100,
    ("fig3", "efficiency", "train_reward"): 2019, ("fig3", "efficiency", "biomass"): 26, ("fig3", "efficiency", "coverage"): 46.8,
    ("fig3", "turbidity", "train_reward"): 1592, ("fig3", "turbidity", "biomass"): 43, ("fig3", "turbidity", "coverage"): 86.9,
    ("fig3", "turbidity_delta", "biomass"): 54, ("fig3", "turbidity_delta", "coverage"): 89.6,
    ("fig5", "easy", "nodrifts", "dqn"): 4574, ("fig5", "easy", "drifts", "dqn"): 5303,
    ("fig5", "medium", "nodrifts", "dqn"): 4806, ("fig5", "medium", "drifts", "dqn"): 5133,
    ("fig5", "hard", "nodrifts", "dqn"): 4958, ("fig5", "hard", "drifts", "dqn"): 5221,
    ("fig5", "easy", "nodrifts", "fixed"): 4023, ("fig5", "easy", "drifts", "fixed"): 3898,
    ("fig5", "medium", "nodrifts", "fixed"): 4229, ("fig5", "medium", "drifts", "fixed"): 4075,
    ("fig5", "hard", "nodrifts", "fixed"): 4075, ("fig5", "hard", "drifts", "fixed"): 3949,
    ("fig6", "easy", "nodrifts", "biomass"): 425, ("fig6", "easy", "drifts", "biomass"): 497,
    ("fig6", "medium", "nodrifts", "biomass"): 459, ("fig6", "medium", "drifts", "biomass"): 487,
    ("fig6", "hard", "nodrifts", "biomass"): 469, ("fig6", "hard", "drifts", "biomass"): 497,
    ("fig6", "easy", "nodrifts", "coverage"): 99.4, ("fig6", "easy", "drifts", "coverage"): 100,
    ("fig6", "medium", "nodrifts", "coverage"): 100, ("fig6", "medium", "drifts", "coverage"): 100,
    ("fig6", "hard", "nodrifts", "coverage"): 99.9, ("fig6", "hard", "drifts", "coverage"): 100,
}

_rows = []

print("Old (hardcoded) vs verified (results_verified/) plot numbers")
print("=" * 70)

# Fig 2
dqn2, ppo2, a2c2 = (eval_lookup("dqn", "hard", "multi_objective", "drifts"),
                    eval_lookup("ppo", "hard", "multi_objective", "drifts"),
                    eval_lookup("a2c", "hard", "multi_objective", "drifts"))
fixed2 = baseline_lookup("fixed_setpoint", "hard", "drifts")
random2 = baseline_lookup("random", "hard", "drifts")
for label, verified in [("dqn", dqn2), ("ppo", ppo2), ("a2c", a2c2), ("fixed", fixed2), ("random", random2)]:
    for metric, vkey in [("reward", "reward"), ("biomass", "biomass"), ("coverage", "coverage")]:
        old = _OLD.get(("fig2", label, metric))
        new = verified[vkey] if verified else None
        _rows.append({"figure": "fig2", "series": label, "metric": metric, "old_hardcoded": old, "verified": new})

# Fig 3 (DQN, hard, with drifts, per reward mode)
MODES = ["multi_objective", "sustainability", "biomass_only", "efficiency", "turbidity", "turbidity_delta"]
fig3_train = {m: final_training_reward(f"dqn_hard_{m}_drifts_*") for m in MODES}
fig3_eval = {m: eval_lookup("dqn", "hard", m, "drifts") for m in MODES}
for m in MODES:
    old_tr = _OLD.get(("fig3", m, "train_reward"))
    _rows.append({"figure": "fig3", "series": m, "metric": "train_reward", "old_hardcoded": old_tr, "verified": fig3_train[m]})
    ev = fig3_eval[m]
    for metric, vkey in [("biomass", "biomass"), ("coverage", "coverage")]:
        old = _OLD.get(("fig3", m, metric))
        _rows.append({"figure": "fig3", "series": m, "metric": metric, "old_hardcoded": old, "verified": ev[vkey] if ev else None})
    _rows.append({"figure": "fig3", "series": m, "metric": "eval_reward", "old_hardcoded": None, "verified": ev["reward"] if ev else None})

# Fig 5/6 (Phase 3, world x drift)
WORLDS = ["easy", "medium", "hard"]
DRIFTS = ["nodrifts", "drifts"]
for world in WORLDS:
    for drift in DRIFTS:
        dqn_e = eval_lookup("dqn", world, "multi_objective", drift)
        fixed_e = baseline_lookup("fixed_setpoint", world, drift)
        _rows.append({"figure": "fig5", "series": f"{world}/{drift}/dqn", "metric": "reward",
                      "old_hardcoded": _OLD.get(("fig5", world, drift, "dqn")), "verified": dqn_e["reward"] if dqn_e else None})
        _rows.append({"figure": "fig5", "series": f"{world}/{drift}/fixed", "metric": "reward",
                      "old_hardcoded": _OLD.get(("fig5", world, drift, "fixed")), "verified": fixed_e["reward"] if fixed_e else None})
        for metric, vkey in [("biomass", "biomass"), ("coverage", "coverage")]:
            _rows.append({"figure": "fig6", "series": f"{world}/{drift}", "metric": metric,
                          "old_hardcoded": _OLD.get(("fig6", world, drift, metric)), "verified": dqn_e[vkey] if dqn_e else None})

_table = pd.DataFrame(_rows)
_table["abs_diff"] = (_table["old_hardcoded"] - _table["verified"]).abs()
_table_path = os.path.join(VERIFIED, "PLOT_NUMBERS_OLD_VS_VERIFIED.csv")
_table.to_csv(_table_path, index=False)
with pd.option_context("display.width", 160, "display.max_rows", 200):
    print(_table.to_string(index=False))
print(f"\nSaved comparison table -> {_table_path}\n")

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
    mean, std = load_learning_curve(f"{algo}_hard_multi_objective_drifts_*")
    if mean is not None:
        ax.plot(mean.index, mean.values, color=colour, label=label, linewidth=1.8)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=colour)

ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Episode Reward (training)")
ax.set_title("Phase 1: Training Learning Curves (hard world, multi_objective, with drifts)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("fig1_phase1_learning_curves.png")

# =============================================================================
# FIGURE 2 -- Phase 1: Evaluation bar chart
# =============================================================================
print("Figure 2: Phase 1 evaluation comparison")

algos = ["DQN", "PPO", "A2C", "Fixed SP", "Random"]
verified2 = [dqn2, ppo2, a2c2, fixed2, random2]
reward = [v["reward"] for v in verified2]
bio = [v["biomass"] for v in verified2]
cov = [v["coverage"] for v in verified2]

x = np.arange(len(algos))
colours = [C["dqn"], C["ppo"], C["a2c"], C["fixed"], C["random"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].bar(x, reward, color=colours, width=0.55, edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(algos)
axes[0].set_ylabel("Mean Evaluation Reward (20 eval episodes, seeds 100-119)")
axes[0].set_title("Evaluation Reward")
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

fig.suptitle("Phase 1: Algorithm Comparison (hard world, multi_objective, with drifts) -- Evaluation", fontsize=11)
plt.tight_layout()
save("fig2_phase1_evaluation.png")

# =============================================================================
# FIGURE 3 -- Phase 2: Reward mode comparison (training reward, eval reward, biomass)
# =============================================================================
print("Figure 3: Phase 2 reward modes")

modes_labels = ["multi_obj\n(P1)", "sustain", "biomass\nonly", "efficiency", "turbidity", "turb\ndelta"]
modes_main_labels = ["multi_obj\n(P1)", "sustain", "biomass\nonly", "efficiency", "turbidity"]
MODES_MAIN = ["multi_objective", "sustainability", "biomass_only", "efficiency", "turbidity"]
colours_p2 = [C["multi"], C["sustain"], C["biomass"], C["efficiency"], C["turbidity"], C["turb_delta"]]

train_main = [fig3_train[m] for m in MODES_MAIN]
eval_main = [fig3_eval[m]["reward"] for m in MODES_MAIN]
bio_p2 = [fig3_eval[m]["biomass"] for m in MODES]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

x_main = np.arange(len(modes_main_labels))
axes[0].bar(x_main, train_main, color=colours_p2[:5], width=0.55, edgecolor="white")
axes[0].set_xticks(x_main); axes[0].set_xticklabels(modes_main_labels)
axes[0].set_ylabel("Mean Training Reward (learning curve, final point)")
axes[0].set_title("Training Reward by Reward Mode\n(turbidity_delta excluded -- different scale)")
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(x_main, eval_main, color=colours_p2[:5], width=0.55, edgecolor="white")
axes[1].set_xticks(x_main); axes[1].set_xticklabels(modes_main_labels)
axes[1].set_ylabel("Mean Evaluation Reward (20 eval episodes, deterministic)")
axes[1].set_title("Evaluation Reward by Reward Mode\n(turbidity_delta excluded -- different scale)")
axes[1].grid(True, axis="y", alpha=0.3)

x2 = np.arange(len(modes_labels))
axes[2].bar(x2, bio_p2, color=colours_p2, width=0.55, edgecolor="white")
axes[2].set_xticks(x2); axes[2].set_xticklabels(modes_labels)
axes[2].set_ylabel("Mean Final Biomass (evaluation)")
axes[2].set_title("Final Biomass by Reward Mode (all six modes)")
axes[2].grid(True, axis="y", alpha=0.3)

fig.suptitle("Phase 2: Reward Mode Comparison (DQN, hard world, with drifts)", fontsize=11)
plt.tight_layout()
save("fig3_phase2_reward_modes.png")

# =============================================================================
# FIGURE 4 -- Phase 2: Coverage trajectories
# =============================================================================
print("Figure 4: Phase 2 coverage trajectories")

fig, ax = plt.subplots(figsize=(8, 4.5))

traj_configs = [
    ("dqn_hard_multi_objective_drifts_*",  C["multi"],     "multi_objective"),
    ("dqn_hard_biomass_only_drifts_*",     C["biomass"],   "biomass_only"),
    ("dqn_hard_sustainability_drifts_*",   C["sustain"],   "sustainability"),
    ("dqn_hard_turbidity_delta_drifts_*",  C["turb_delta"],"turbidity_delta"),
    ("dqn_hard_turbidity_drifts_*",        C["turbidity"], "turbidity"),
]

for pattern, colour, label in traj_configs:
    dfs = []
    for d in run_dirs(pattern):
        f = os.path.join(d, "coverage_trajectory.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            if "mean_coverage" in df.columns:
                dfs.append(df["mean_coverage"].values)
    if dfs:
        min_len = min(len(d) for d in dfs)
        arr = np.array([d[:min_len] for d in dfs])
        mean = arr.mean(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean * 100, color=colour, label=label, linewidth=1.8)

ax.set_xlabel("Episode Step")
ax.set_ylabel("Mean Colony Coverage (%)")
ax.set_title("Phase 2: Colony Coverage Trajectory by Reward Mode (evaluation episodes)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("fig4_phase2_coverage_trajectories.png")

# =============================================================================
# FIGURE 5 -- Phase 3: DQN vs Fixed Setpoint
# =============================================================================
print("Figure 5: Phase 3 DQN vs Fixed Setpoint")

configs = ["easy\nno drift", "easy\nwith drift", "medium\nno drift",
           "medium\nwith drift", "hard\nno drift", "hard\nwith drift"]
pairs = [("easy", "nodrifts"), ("easy", "drifts"), ("medium", "nodrifts"),
         ("medium", "drifts"), ("hard", "nodrifts"), ("hard", "drifts")]
dqn_rew = [eval_lookup("dqn", w, "multi_objective", d)["reward"] for w, d in pairs]
fixed_rew = [baseline_lookup("fixed_setpoint", w, d)["reward"] for w, d in pairs]
advantage = [d - f for d, f in zip(dqn_rew, fixed_rew)]

x = np.arange(len(configs))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].bar(x - width/2, dqn_rew, width, label="DQN", color=C["dqn"], edgecolor="white")
axes[0].bar(x + width/2, fixed_rew, width, label="Fixed Setpoint", color=C["fixed"], edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(configs, fontsize=8)
axes[0].set_ylabel("Mean Evaluation Reward")
axes[0].set_title("DQN vs Fixed Setpoint: Evaluation Reward")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

bar_colours = [C["nodrifts"], C["drifts"]] * 3
axes[1].bar(x, advantage, color=bar_colours, edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(configs, fontsize=8)
axes[1].set_ylabel("DQN Advantage (reward gap, evaluation)")
axes[1].set_title("DQN Advantage over Fixed Setpoint")
axes[1].grid(True, axis="y", alpha=0.3)
nd_patch = mpatches.Patch(color=C["nodrifts"], label="No drifts")
wd_patch = mpatches.Patch(color=C["drifts"], label="With drifts")
axes[1].legend(handles=[nd_patch, wd_patch])

fig.suptitle("Phase 3: World Difficulty and Drift Condition Comparison -- Evaluation", fontsize=11)
plt.tight_layout()
save("fig5_phase3_dqn_vs_baseline.png")

# =============================================================================
# FIGURE 6 -- Phase 3: Biomass and coverage by world preset
# =============================================================================
print("Figure 6: Phase 3 difficulty and drift effect")

presets = ["easy", "medium", "hard"]
nd_bio = [eval_lookup("dqn", w, "multi_objective", "nodrifts")["biomass"] for w in presets]
wd_bio = [eval_lookup("dqn", w, "multi_objective", "drifts")["biomass"] for w in presets]
nd_cov = [eval_lookup("dqn", w, "multi_objective", "nodrifts")["coverage"] for w in presets]
wd_cov = [eval_lookup("dqn", w, "multi_objective", "drifts")["coverage"] for w in presets]

x = np.arange(len(presets))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].bar(x - width/2, nd_bio, width, label="No drifts", color=C["nodrifts"], edgecolor="white")
axes[0].bar(x + width/2, wd_bio, width, label="With drifts", color=C["drifts"], edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(presets)
axes[0].set_ylabel("Mean Final Biomass (evaluation)")
axes[0].set_title("Final Biomass by World Preset")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(x - width/2, nd_cov, width, label="No drifts", color=C["nodrifts"], edgecolor="white")
axes[1].bar(x + width/2, wd_cov, width, label="With drifts", color=C["drifts"], edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(presets)
axes[1].set_ylabel("Mean Final Coverage (%) (evaluation)")
axes[1].set_title("Colony Coverage by World Preset")
axes[1].set_ylim(95, 101)
axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

fig.suptitle("Phase 3: DQN Performance Across World Presets and Drift Conditions -- Evaluation", fontsize=11)
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
    mean, std = load_learning_curve(f"{algo}_hard_multi_objective_drifts_*")
    if mean is not None:
        axes[idx].plot(mean.index, mean.values, color=colour, linewidth=1.8)
        axes[idx].fill_between(mean.index, mean - std, mean + std, alpha=0.25, color=colour)
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Mean Episode Reward (training)")
        axes[idx].grid(True, alpha=0.3)
    else:
        axes[idx].set_title(f"{label} (no data)")
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Mean Episode Reward (training)")

fig.suptitle("Phase 1: Per-Algorithm Training Learning Curves with Variance", fontsize=11)
plt.tight_layout()
save("fig7_phase1_per_algo_curves.png")

print("\nAll figures saved to plots_verified/")
