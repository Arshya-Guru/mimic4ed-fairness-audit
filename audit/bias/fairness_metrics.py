"""
Compute group fairness metrics using Fairlearn's MetricFrame.

For each task × model × demographic column, computes:
- Demographic parity difference
- Equalized odds difference
- Equal opportunity difference

Flags combinations that exceed standard thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)

from audit.config import (
    TASKS, MODELS, DEMOGRAPHIC_PARITY_THRESHOLD,
    EQUALIZED_ODDS_THRESHOLD, BIAS_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.utils import (
    load_predictions, add_demographic_groups, get_optimal_threshold,
)


def _equal_opportunity_difference(y_true, y_pred, sensitive_features):
    """Compute equal opportunity difference (TPR disparity).

    Equal opportunity requires equal true positive rates across groups.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        sensitive_features: Group membership array.

    Returns:
        Maximum difference in TPR between any two groups.
    """
    def tpr(y_true, y_pred):
        """True positive rate."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    mf = MetricFrame(
        metrics=tpr,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return mf.difference(method="between_groups")


def compute_fairness_metrics(tasks=None, models=None, variant="default"):
    """Compute fairness metrics for all task × model × demographic combinations.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant to evaluate.

    Returns:
        DataFrame with fairness metric results and violation flags.
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    rows = []

    for task in tasks:
        for model_name in models:
            try:
                preds = load_predictions(task, model_name, variant)
            except FileNotFoundError:
                continue

            y_true = preds["y_true"].values
            y_prob = preds["y_prob"].values
            threshold = get_optimal_threshold(y_true, y_prob)
            y_pred = (y_prob >= threshold).astype(int)

            preds = add_demographic_groups(preds)

            demo_groups = {
                "ethnicity": "ethnicity_group",
                "gender": "gender_group",
                "age": "age_group",
                "insurance": "insurance_group",
            }

            for demo_col, group_col in demo_groups.items():
                if group_col not in preds.columns:
                    continue

                sensitive = preds[group_col].values
                # Skip if fewer than 2 groups
                unique_groups = pd.Series(sensitive).dropna().unique()
                if len(unique_groups) < 2:
                    continue

                # Filter out rows with NaN in sensitive feature
                mask = pd.notna(sensitive)
                yt = y_true[mask]
                yp = y_pred[mask]
                sf = sensitive[mask]

                try:
                    dp_diff = demographic_parity_difference(
                        yt, yp, sensitive_features=sf
                    )
                except Exception:
                    dp_diff = np.nan

                try:
                    eo_diff = equalized_odds_difference(
                        yt, yp, sensitive_features=sf
                    )
                except Exception:
                    eo_diff = np.nan

                try:
                    eop_diff = _equal_opportunity_difference(yt, yp, sf)
                except Exception:
                    eop_diff = np.nan

                # Compute per-group selection rates for detail
                mf = MetricFrame(
                    metrics={"selection_rate": lambda y_t, y_p: np.mean(y_p)},
                    y_true=yt,
                    y_pred=yp,
                    sensitive_features=sf,
                )
                group_rates = mf.by_group["selection_rate"].to_dict()

                flags = []
                if not np.isnan(dp_diff) and abs(dp_diff) > DEMOGRAPHIC_PARITY_THRESHOLD:
                    flags.append("DEMOGRAPHIC_PARITY_VIOLATION")
                if not np.isnan(eo_diff) and abs(eo_diff) > EQUALIZED_ODDS_THRESHOLD:
                    flags.append("EQUALIZED_ODDS_VIOLATION")
                if not np.isnan(eop_diff) and abs(eop_diff) > EQUALIZED_ODDS_THRESHOLD:
                    flags.append("EQUAL_OPPORTUNITY_VIOLATION")

                rows.append({
                    "task": task,
                    "model": model_name,
                    "demographic_col": demo_col,
                    "demographic_parity_diff": dp_diff,
                    "equalized_odds_diff": eo_diff,
                    "equal_opportunity_diff": eop_diff,
                    "group_selection_rates": str(group_rates),
                    "flags": "; ".join(flags) if flags else "",
                })

    df = pd.DataFrame(rows)
    out_path = BIAS_OUTPUT_DIR / "fairness_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"  Fairness metrics saved to {out_path}")

    violations = df[df["flags"] != ""]
    if len(violations) > 0:
        print(f"  WARNING: {len(violations)} fairness violation(s) detected")

    return df


if __name__ == "__main__":
    compute_fairness_metrics()
