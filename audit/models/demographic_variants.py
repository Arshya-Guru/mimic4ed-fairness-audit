"""
Train demographic-aware vs demographic-agnostic model variants.

Variant A: full feature set (includes gender, age — demographic-aware).
Variant B: drops race/ethnicity, insurance, and any address/zip features
           (demographic-agnostic).

Comparing performance between variants per subgroup reveals whether bias
is encoded through proxy features even when explicit demographics are removed.
"""

import pandas as pd

from audit.config import (
    TASKS, MODELS, TASK_FEATURES, DEMOGRAPHIC_FEATURES,
    TRAIN_CSV, TEST_CSV, BIAS_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.train_models import train_single


def _get_agnostic_features(task):
    """Get feature list with demographic-sensitive columns removed.

    Args:
        task: Task name.

    Returns:
        List of feature column names without demographic features.
    """
    all_features = TASK_FEATURES[task]
    return [f for f in all_features if f not in DEMOGRAPHIC_FEATURES]


def train_demographic_variants(tasks=None, models=None,
                                train_csv=None, test_csv=None):
    """Train both demographic-aware and demographic-agnostic variants for all model×task combos.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        train_csv: Path to training data.
        test_csv: Path to test data.

    Returns:
        Dict mapping (task, model_name, variant) to (model, y_test, y_prob).
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    train_csv = train_csv or TRAIN_CSV
    test_csv = test_csv or TEST_CSV

    results = {}

    for task in tasks:
        agnostic_features = _get_agnostic_features(task)
        for model_name in models:
            print(f"  Training demographic-aware {model_name} for {task}...")
            res_a = train_single(
                task, model_name, train_csv, test_csv,
                variant="demographic_aware"
            )
            results[(task, model_name, "demographic_aware")] = res_a

            print(f"  Training demographic-agnostic {model_name} for {task}...")
            res_b = train_single(
                task, model_name, train_csv, test_csv,
                feature_override=agnostic_features,
                variant="demographic_agnostic"
            )
            results[(task, model_name, "demographic_agnostic")] = res_b

    return results


def compare_variants(tasks=None, models=None):
    """Compare AUROC between aware and agnostic variants per subgroup.

    Loads saved predictions and computes per-subgroup AUROC differences.

    Args:
        tasks: List of tasks.
        models: List of model names.

    Returns:
        DataFrame with comparison results.
    """
    from sklearn.metrics import roc_auc_score
    from audit.models.utils import load_predictions, add_demographic_groups

    tasks = tasks or TASKS
    models = models or MODELS
    rows = []

    for task in tasks:
        for model_name in models:
            for variant in ["demographic_aware", "demographic_agnostic"]:
                try:
                    preds = load_predictions(task, model_name, variant)
                except FileNotFoundError:
                    continue

                preds = add_demographic_groups(preds)
                overall_auc = roc_auc_score(preds["y_true"], preds["y_prob"])
                rows.append({
                    "task": task, "model": model_name, "variant": variant,
                    "group": "Overall", "demographic_col": "all",
                    "auroc": overall_auc,
                })

                for demo_col in ["ethnicity_group", "gender_group", "age_group", "insurance_group"]:
                    if demo_col not in preds.columns:
                        continue
                    for group, gdf in preds.groupby(demo_col):
                        if len(gdf) < 20 or gdf["y_true"].nunique() < 2:
                            continue
                        auc = roc_auc_score(gdf["y_true"], gdf["y_prob"])
                        rows.append({
                            "task": task, "model": model_name, "variant": variant,
                            "group": str(group),
                            "demographic_col": demo_col.replace("_group", ""),
                            "auroc": auc,
                        })

    df = pd.DataFrame(rows)

    # Pivot to show side-by-side comparison
    if not df.empty:
        pivot = df.pivot_table(
            index=["task", "model", "demographic_col", "group"],
            columns="variant", values="auroc"
        ).reset_index()
        if "demographic_aware" in pivot.columns and "demographic_agnostic" in pivot.columns:
            pivot["auroc_difference"] = pivot["demographic_aware"] - pivot["demographic_agnostic"]
        out_path = BIAS_OUTPUT_DIR / "demographic_variant_comparison.csv"
        BIAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(out_path, index=False)
        print(f"  Variant comparison saved to {out_path}")
        return pivot

    return df


if __name__ == "__main__":
    train_demographic_variants()
    compare_variants()
