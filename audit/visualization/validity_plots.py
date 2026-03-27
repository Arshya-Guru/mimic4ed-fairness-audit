"""
Scatter plots of model predictions vs severity indicators by demographic group.

Shows regression lines colored by demographic group to visualize whether
the prediction-severity relationship holds equally across populations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from audit.config import (
    TASKS, MODELS, PLOTS_DIR, TASK_DISPLAY_NAME, MODEL_DISPLAY_NAME,
    ensure_output_dirs,
)
from audit.models.utils import load_predictions, add_demographic_groups

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


def plot_validity_scatter(tasks=None, models=None, variant="default"):
    """Generate scatter plots of predictions vs severity indicators by demographic group.

    Args:
        tasks: List of tasks.
        models: List of model names.
        variant: Model variant.

    Returns:
        List of paths to generated plot files.
    """
    ensure_output_dirs()
    _apply_dark_theme()
    tasks = tasks or TASKS
    models = models or MODELS
    plot_paths = []

    severity_cols = {
        "triage_acuity": "ESI Acuity Score",
        "ed_los": "ED Length of Stay",
    }

    demo_groups = {
        "ethnicity": "ethnicity_group",
        "gender": "gender_group",
    }

    for task in tasks:
        for model_name in models:
            try:
                preds = load_predictions(task, model_name, variant)
            except FileNotFoundError:
                continue

            preds = add_demographic_groups(preds)

            for sev_col, sev_label in severity_cols.items():
                if sev_col not in preds.columns:
                    continue

                sev_vals = pd.to_numeric(preds[sev_col], errors="coerce")

                for demo_col, group_col in demo_groups.items():
                    if group_col not in preds.columns:
                        continue

                    groups = preds[group_col].dropna().unique()
                    if len(groups) < 2:
                        continue

                    palette = sns.color_palette("Set2", len(groups))

                    fig, ax = plt.subplots(figsize=(10, 7))

                    for i, group in enumerate(sorted(groups)):
                        mask = preds[group_col] == group
                        gx = preds.loc[mask, "y_prob"]
                        gy = sev_vals[mask]
                        valid = ~(gx.isna() | gy.isna())
                        gx = gx[valid]
                        gy = gy[valid]

                        if len(gx) < 10:
                            continue

                        # Scatter with transparency
                        ax.scatter(gx, gy, alpha=0.15, s=10,
                                   color=palette[i], label=f"{group} (n={len(gx)})")

                        # Regression line
                        if len(gx) > 20:
                            z = np.polyfit(gx.values, gy.values, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(gx.min(), gx.max(), 100)
                            ax.plot(x_line, p(x_line), "-", linewidth=2.5,
                                    color=palette[i])

                    ax.set_xlabel("Predicted Probability", fontsize=12)
                    ax.set_ylabel(sev_label, fontsize=12)
                    ax.set_title(
                        f"Prediction vs {sev_label} by {demo_col.title()}\n"
                        f"{TASK_DISPLAY_NAME.get(task, task)} - "
                        f"{MODEL_DISPLAY_NAME.get(model_name, model_name)}",
                        fontsize=13, fontweight="bold",
                    )
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc="best", fontsize=9, framealpha=0.8)
                    ax.grid(alpha=0.3)

                    plt.tight_layout()
                    fname = f"validity_scatter_{task}_{model_name}_{sev_col}_{demo_col}"
                    for fmt in ["png", "pdf"]:
                        path = PLOTS_DIR / f"{fname}.{fmt}"
                        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
                        plot_paths.append(path)
                    plt.close(fig)

    print(f"  Generated {len(plot_paths)} validity plots")
    return plot_paths


if __name__ == "__main__":
    plot_validity_scatter()
