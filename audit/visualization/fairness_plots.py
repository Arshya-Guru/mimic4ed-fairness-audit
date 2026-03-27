"""
Radar/spider plots comparing fairness metrics across models.

Shows demographic parity, equalized odds, and equal opportunity differences
in a radar chart for visual comparison across models and demographic groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from audit.config import (
    BIAS_OUTPUT_DIR, PLOTS_DIR, TASK_DISPLAY_NAME, MODEL_DISPLAY_NAME,
    DEMOGRAPHIC_PARITY_THRESHOLD, ensure_output_dirs,
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

MODEL_COLORS = sns.color_palette("husl", 8)


def _apply_dark_theme():
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update(DARK_STYLE)
    plt.rcParams["font.size"] = 11


def plot_fairness_radar():
    """Generate radar/spider plots comparing fairness metrics across models.

    Creates one radar plot per task showing the three fairness metrics
    for each model, with one spoke per demographic column.

    Returns:
        List of paths to generated plot files.
    """
    ensure_output_dirs()
    _apply_dark_theme()
    plot_paths = []

    fair_path = BIAS_OUTPUT_DIR / "fairness_metrics.csv"
    if not fair_path.exists():
        print("  Fairness metrics not found, skipping radar plots")
        return plot_paths

    df = pd.read_csv(fair_path)

    fairness_cols = [
        "demographic_parity_diff",
        "equalized_odds_diff",
        "equal_opportunity_diff",
    ]
    fairness_labels = [
        "Demographic\nParity",
        "Equalized\nOdds",
        "Equal\nOpportunity",
    ]

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        models = task_df["model"].unique()

        # Create a radar plot: axes = demographic columns, one line per model
        # Each point is the max fairness violation across the three metrics
        demo_cols = task_df["demographic_col"].unique()

        if len(demo_cols) < 2:
            continue

        fig, axes = plt.subplots(1, len(fairness_cols), figsize=(6 * len(fairness_cols), 6),
                                  subplot_kw={"polar": True})
        if len(fairness_cols) == 1:
            axes = [axes]

        for ax_idx, (fcol, flabel) in enumerate(zip(fairness_cols, fairness_labels)):
            ax = axes[ax_idx]
            angles = np.linspace(0, 2 * np.pi, len(demo_cols), endpoint=False).tolist()
            angles += angles[:1]  # close the polygon

            for i, model in enumerate(models):
                model_df = task_df[task_df["model"] == model]
                values = []
                for dc in demo_cols:
                    row = model_df[model_df["demographic_col"] == dc]
                    if len(row) > 0:
                        values.append(abs(row[fcol].values[0]))
                    else:
                        values.append(0)
                values += values[:1]

                ax.plot(angles, values, "o-", linewidth=2,
                        label=MODEL_DISPLAY_NAME.get(model, model),
                        color=MODEL_COLORS[i % len(MODEL_COLORS)])
                ax.fill(angles, values, alpha=0.1,
                        color=MODEL_COLORS[i % len(MODEL_COLORS)])

            # Add threshold circle
            threshold_vals = [DEMOGRAPHIC_PARITY_THRESHOLD] * (len(demo_cols) + 1)
            ax.plot(angles, threshold_vals, "--", color="#ff6b6b",
                    linewidth=1, alpha=0.7, label="Threshold")

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([dc.title() for dc in demo_cols], size=9)
            ax.set_title(flabel, size=12, fontweight="bold", pad=20)

        axes[0].legend(loc="upper left", bbox_to_anchor=(-0.3, 1.15), fontsize=8)
        fig.suptitle(
            f"Fairness Metrics - {TASK_DISPLAY_NAME.get(task, task)}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        fname = f"fairness_radar_{task}"
        for fmt in ["png", "pdf"]:
            path = PLOTS_DIR / f"{fname}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plot_paths.append(path)
        plt.close(fig)

    # Also create a summary heatmap
    if not df.empty:
        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.4)))
        pivot_data = df.pivot_table(
            index=["task", "model"],
            columns="demographic_col",
            values="demographic_parity_diff",
            aggfunc="first",
        )
        if not pivot_data.empty:
            sns.heatmap(
                pivot_data.abs(), ax=ax, annot=True, fmt=".3f",
                cmap="YlOrRd", linewidths=0.5, linecolor="#444",
                cbar_kws={"label": "|Demographic Parity Difference|"},
            )
            ax.set_title("Demographic Parity Difference Heatmap",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            for fmt in ["png", "pdf"]:
                path = PLOTS_DIR / f"fairness_heatmap.{fmt}"
                fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
                plot_paths.append(path)
        plt.close(fig)

    print(f"  Generated {len(plot_paths)} fairness plots")
    return plot_paths


if __name__ == "__main__":
    plot_fairness_radar()
