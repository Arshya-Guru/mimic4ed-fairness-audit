"""
Compute AUROC, sensitivity, specificity, PPV, NPV, FPR, FNR per
demographic subgroup for each task × model combination.

Uses bootstrap resampling (n=1000) to compute 95% confidence intervals.
Flags subgroups where sensitivity drops >5% below the overall population.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from audit.config import (
    TASKS, MODELS, BOOTSTRAP_N, BOOTSTRAP_CI, RANDOM_SEED,
    SENSITIVITY_DROP_THRESHOLD, BIAS_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.utils import load_predictions, add_demographic_groups, get_optimal_threshold


def _compute_metrics(y_true, y_prob, threshold):
    """Compute classification metrics at a given threshold.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dict of metric name -> value. Returns NaN-filled dict if computation fails.
    """
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "auroc": auroc, "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv, "fpr": fpr, "fnr": fnr,
    }


def _bootstrap_metrics(y_true, y_prob, threshold, n_boot=BOOTSTRAP_N,
                        ci=BOOTSTRAP_CI, seed=RANDOM_SEED):
    """Bootstrap confidence intervals for all classification metrics.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.
        n_boot: Number of bootstrap iterations.
        ci: Confidence level (e.g., 0.95).
        seed: Random seed.

    Returns:
        Dict mapping metric_name -> (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    point = _compute_metrics(y_true, y_prob, threshold)

    boot_results = {k: [] for k in point}
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        m = _compute_metrics(yt, yp, threshold)
        for k, v in m.items():
            boot_results[k].append(v)

    alpha = (1 - ci) / 2
    result = {}
    for k in point:
        vals = [v for v in boot_results[k] if not np.isnan(v)]
        if len(vals) > 10:
            lo = np.percentile(vals, 100 * alpha)
            hi = np.percentile(vals, 100 * (1 - alpha))
        else:
            lo, hi = np.nan, np.nan
        result[k] = (point[k], lo, hi)

    return result


def compute_subgroup_performance(tasks=None, models=None, variant="default"):
    """Compute per-subgroup performance metrics for all task × model combinations.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant to evaluate.

    Returns:
        DataFrame with columns [task, model, demographic_col, group, metric,
        value, ci_lower, ci_upper, flag].
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
                print(f"  Predictions not found for {task}/{model_name}/{variant}, skipping")
                continue

            y_true = preds["y_true"].values
            y_prob = preds["y_prob"].values
            threshold = get_optimal_threshold(y_true, y_prob)

            # Overall population metrics
            overall = _bootstrap_metrics(y_true, y_prob, threshold)
            overall_sensitivity = overall["sensitivity"][0]

            for metric_name, (val, lo, hi) in overall.items():
                rows.append({
                    "task": task, "model": model_name,
                    "demographic_col": "overall", "group": "All",
                    "metric": metric_name, "value": val,
                    "ci_lower": lo, "ci_upper": hi, "flag": "",
                })

            # Per-subgroup metrics
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
                for group_name, gdf in preds.groupby(group_col):
                    if len(gdf) < 20:
                        continue
                    yt = gdf["y_true"].values
                    yp = gdf["y_prob"].values
                    if yt.sum() == 0 or yt.sum() == len(yt):
                        continue

                    metrics = _bootstrap_metrics(yt, yp, threshold)

                    for metric_name, (val, lo, hi) in metrics.items():
                        flag = ""
                        if (metric_name == "sensitivity" and
                                not np.isnan(val) and not np.isnan(overall_sensitivity)):
                            drop = overall_sensitivity - val
                            if drop > SENSITIVITY_DROP_THRESHOLD:
                                flag = f"SENSITIVITY_DROP_{drop:.3f}"

                        rows.append({
                            "task": task, "model": model_name,
                            "demographic_col": demo_col, "group": str(group_name),
                            "metric": metric_name, "value": val,
                            "ci_lower": lo, "ci_upper": hi, "flag": flag,
                        })

    df = pd.DataFrame(rows)
    out_path = BIAS_OUTPUT_DIR / "subgroup_performance.csv"
    df.to_csv(out_path, index=False)
    print(f"  Subgroup performance saved to {out_path}")

    flagged = df[df["flag"] != ""]
    if len(flagged) > 0:
        print(f"  WARNING: {len(flagged)} subgroup metric(s) flagged for sensitivity drops")

    return df


if __name__ == "__main__":
    compute_subgroup_performance()
