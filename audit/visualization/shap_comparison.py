"""
Side-by-side SHAP summary plots for different demographic groups.

Generates beeswarm-style SHAP plots comparing feature importance
distributions between demographic subgroups (e.g., White vs Black patients).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from audit.config import (
    BIAS_OUTPUT_DIR, PLOTS_DIR, TASK_DISPLAY_NAME, MODEL_DISPLAY_NAME,
    ensure_output_dirs,
)

matplotlib.use("Agg")

DARK_STYLE = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#2d2d44",
    "axes.edgecolor": "#555577",
    "axes.labelcolor": "#ccccdd",
    "text.color": "#ccccdd",
    "xtick.color": "#aaaacc",
    "ytick.color": "#aaaacc",
    "grid.color": "#3d3d55",
    "legend.facecolor": "#2d2d44",
    "legend.edgecolor": "#555577",
}


def _apply_dark_theme():
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update(DARK_STYLE)
    plt.rcParams["font.size"] = 11


def plot_shap_comparison(top_n=15):
    """Generate side-by-side SHAP importance plots for demographic subgroups.

    For each task × model, creates horizontal bar plots showing mean |SHAP|
    values for the top features, with bars grouped by demographic subgroup.

    Args:
        top_n: Number of top features to display.

    Returns:
        List of paths to generated plot files.
    """
    ensure_output_dirs()
    _apply_dark_theme()
    plot_paths = []

    imp_path = BIAS_OUTPUT_DIR / "shap_subgroup_importance.csv"
    if not imp_path.exists():
        print("  SHAP importance data not found, skipping comparison plots")
        return plot_paths

    df = pd.read_csv(imp_path)

    # Get overall top features for ordering
    overall = df[df["demographic_col"] == "overall"]

    for task in df["task"].unique():
        for model in df["model"].unique():
            task_model_df = df[(df["task"] == task) & (df["model"] == model)]
            overall_tm = overall[(overall["task"] == task) & (overall["model"] == model)]

            if overall_tm.empty:
                continue

            top_features = (
                overall_tm.nlargest(top_n, "mean_abs_shap")["feature"].tolist()
            )

            for demo_col in ["ethnicity", "gender", "age", "insurance"]:
                demo_df = task_model_df[task_model_df["demographic_col"] == demo_col]
                if demo_df.empty:
                    continue

                groups = demo_df["group"].unique()
                if len(groups) < 2:
                    continue

                n_groups = len(groups)
                fig, axes = plt.subplots(1, n_groups,
                                         figsize=(6 * n_groups, max(6, top_n * 0.4)),
                                         sharey=True)
                if n_groups == 1:
                    axes = [axes]

                palette = sns.color_palette("Set2", n_groups)

                for gi, (group, ax) in enumerate(zip(sorted(groups), axes)):
                    group_df = demo_df[demo_df["group"] == group]
                    group_features = group_df.set_index("feature")

                    vals = []
                    for feat in reversed(top_features):
                        if feat in group_features.index:
                            vals.append(group_features.loc[feat, "mean_abs_shap"])
                        else:
                            vals.append(0)

                    y_pos = range(len(top_features))
                    ax.barh(y_pos, vals, color=palette[gi], edgecolor="white",
                            linewidth=0.5)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(list(reversed(top_features)), fontsize=9)
                    ax.set_xlabel("Mean |SHAP Value|")
                    ax.set_title(f"{group}", fontsize=12, fontweight="bold")
                    ax.grid(axis="x", alpha=0.3)

                fig.suptitle(
                    f"SHAP Feature Importance by {demo_col.title()}\n"
                    f"{TASK_DISPLAY_NAME.get(task, task)} - "
                    f"{MODEL_DISPLAY_NAME.get(model, model)}",
                    fontsize=13, fontweight="bold",
                )
                plt.tight_layout()

                fname = f"shap_comparison_{task}_{model}_{demo_col}"
                for fmt in ["png", "pdf"]:
                    path = PLOTS_DIR / f"{fname}.{fmt}"
                    fig.savefig(path, dpi=300, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    plot_paths.append(path)
                plt.close(fig)

    print(f"  Generated {len(plot_paths)} SHAP comparison plots")
    return plot_paths


if __name__ == "__main__":
    plot_shap_comparison()
