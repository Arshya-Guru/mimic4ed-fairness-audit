"""
Bar charts of AUROC/sensitivity/specificity per subgroup, grouped by task.

Uses a consistent dark theme across all plots. Saves as PNG (300dpi) and PDF.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

from audit.config import (
    BIAS_OUTPUT_DIR, PLOTS_DIR, TASKS, MODELS,
    TASK_DISPLAY_NAME, MODEL_DISPLAY_NAME, ensure_output_dirs,
)

matplotlib.use("Agg")

# Dark theme configuration
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

SUBGROUP_PALETTE = sns.color_palette("Set2", 12)


def _apply_dark_theme():
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update(DARK_STYLE)
    plt.rcParams["font.size"] = 11


def plot_subgroup_performance(metrics=None):
    """Generate grouped bar charts of performance metrics per subgroup.

    Creates one plot per task showing AUROC, sensitivity, and specificity
    across demographic subgroups for each model.

    Args:
        metrics: List of metrics to plot. Defaults to ['auroc', 'sensitivity', 'specificity'].

    Returns:
        List of paths to generated plot files.
    """
    ensure_output_dirs()
    _apply_dark_theme()
    metrics = metrics or ["auroc", "sensitivity", "specificity"]
    plot_paths = []

    perf_path = BIAS_OUTPUT_DIR / "subgroup_performance.csv"
    if not perf_path.exists():
        print("  Subgroup performance data not found, skipping plots")
        return plot_paths

    df = pd.read_csv(perf_path)

    for demo_col in df["demographic_col"].unique():
        if demo_col == "overall":
            continue

        for metric in metrics:
            metric_df = df[
                (df["demographic_col"] == demo_col) & (df["metric"] == metric)
            ]
            if metric_df.empty:
                continue

            for task in metric_df["task"].unique():
                task_df = metric_df[metric_df["task"] == task]
                if task_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))
                groups = task_df["group"].unique()
                model_names = task_df["model"].unique()
                x = range(len(groups))
                width = 0.8 / len(model_names)

                for i, model in enumerate(model_names):
                    model_df = task_df[task_df["model"] == model]
                    vals = []
                    ci_lo = []
                    ci_hi = []
                    for g in groups:
                        grow = model_df[model_df["group"] == g]
                        if len(grow) > 0:
                            vals.append(grow["value"].values[0])
                            ci_lo.append(grow["ci_lower"].values[0])
                            ci_hi.append(grow["ci_upper"].values[0])
                        else:
                            vals.append(0)
                            ci_lo.append(0)
                            ci_hi.append(0)

                    positions = [xi + i * width for xi in x]
                    yerr_lo = [v - cl for v, cl in zip(vals, ci_lo)]
                    yerr_hi = [ch - v for v, ch in zip(vals, ci_hi)]
                    ax.bar(
                        positions, vals, width,
                        label=MODEL_DISPLAY_NAME.get(model, model),
                        color=SUBGROUP_PALETTE[i % len(SUBGROUP_PALETTE)],
                        yerr=[yerr_lo, yerr_hi],
                        capsize=3, edgecolor="white", linewidth=0.5,
                    )

                ax.set_xlabel(demo_col.replace("_", " ").title())
                ax.set_ylabel(metric.upper())
                ax.set_title(
                    f"{metric.upper()} by {demo_col.title()} - "
                    f"{TASK_DISPLAY_NAME.get(task, task)}",
                    fontsize=13, fontweight="bold",
                )
                ax.set_xticks([xi + width * (len(model_names) - 1) / 2 for xi in x])
                ax.set_xticklabels(groups, rotation=30, ha="right")
                ax.legend(loc="lower right", fontsize=9)
                ax.grid(axis="y", alpha=0.3)

                if metric in ("auroc", "sensitivity", "specificity"):
                    ax.set_ylim(0, 1.05)

                plt.tight_layout()
                fname = f"performance_{task}_{demo_col}_{metric}"
                for fmt in ["png", "pdf"]:
                    path = PLOTS_DIR / f"{fname}.{fmt}"
                    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
                    plot_paths.append(path)
                plt.close(fig)

    print(f"  Generated {len(plot_paths)} performance plots")
    return plot_paths


if __name__ == "__main__":
    plot_subgroup_performance()
