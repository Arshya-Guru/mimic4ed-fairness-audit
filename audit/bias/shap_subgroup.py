"""
SHAP feature importance analysis stratified by demographic subgroup.

For each model × task: compute SHAP values on the test set, split by
demographic subgroup, compare mean absolute SHAP values (feature importance),
and compute rank correlation (Spearman) of feature importance between subgroups.

Flags features where importance rank shifts by >5 positions between groups.
"""

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from audit.config import (
    TASKS, MODELS, SHAP_SAMPLE_SIZE, RANDOM_SEED,
    RANK_SHIFT_THRESHOLD, BIAS_OUTPUT_DIR, TRAIN_CSV, TEST_CSV,
    ensure_output_dirs,
)
from audit.models.utils import (
    load_model, load_data, add_demographic_groups,
)


def compute_shap_values(model, X, model_name):
    """Compute SHAP values for a dataset using the appropriate explainer.

    Args:
        model: Trained sklearn model.
        X: Feature DataFrame.
        model_name: Model type identifier for choosing explainer.

    Returns:
        numpy array of SHAP values (n_samples, n_features).
    """
    X_arr = X.values if hasattr(X, "values") else X

    if model_name in ("rf", "gbm"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_arr)
        # For binary classifiers, TreeExplainer may return a list or 3D array
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
    elif model_name == "lr":
        # LinearExplainer is much faster for linear models
        bg = shap.sample(X, min(100, len(X)), random_state=RANDOM_SEED)
        explainer = shap.LinearExplainer(model, bg.values)
        sv = explainer.shap_values(X_arr)
    else:
        # Use KernelExplainer for MLP and other non-tree models
        bg = shap.sample(X, min(50, len(X)), random_state=RANDOM_SEED)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], bg.values
                )
            else:
                explainer = shap.KernelExplainer(model.predict, bg.values)
            sv = explainer.shap_values(X_arr, nsamples=50)

    return np.array(sv)


def analyze_shap_subgroups(tasks=None, models=None, variant="default",
                            train_csv=None, test_csv=None):
    """Run full SHAP subgroup analysis for all task × model combinations.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant to load.
        train_csv: Path to training data.
        test_csv: Path to test data.

    Returns:
        Tuple of (importance_df, rank_shift_df) DataFrames.
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    train_csv = train_csv or TRAIN_CSV
    test_csv = test_csv or TEST_CSV

    all_importance_rows = []
    all_rank_shift_rows = []

    for task in tasks:
        X_train, y_train, X_test, y_test, df_test = load_data(
            train_csv, test_csv, task
        )
        df_test_demo = add_demographic_groups(df_test)

        for model_name in models:
            print(f"  Computing SHAP values for {model_name}/{task}...")
            try:
                model = load_model(task, model_name, variant)
            except FileNotFoundError:
                print(f"    Model not found, skipping")
                continue

            # Sample for SHAP computation
            n_sample = min(SHAP_SAMPLE_SIZE, len(X_test))
            rng = np.random.RandomState(RANDOM_SEED)
            sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
            X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
            demo_sample = df_test_demo.iloc[sample_idx].reset_index(drop=True)

            shap_vals = compute_shap_values(model, X_sample, model_name)
            feature_names = list(X_test.columns)

            # Overall feature importance
            overall_importance = np.abs(shap_vals).mean(axis=0)
            overall_rank = pd.Series(overall_importance, index=feature_names).rank(
                ascending=False
            )

            for i, feat in enumerate(feature_names):
                all_importance_rows.append({
                    "task": task, "model": model_name,
                    "demographic_col": "overall", "group": "All",
                    "feature": feat,
                    "mean_abs_shap": overall_importance[i],
                    "rank": int(overall_rank[feat]),
                })

            # Per-subgroup importance
            demo_groups = {
                "ethnicity": "ethnicity_group",
                "gender": "gender_group",
                "age": "age_group",
                "insurance": "insurance_group",
            }

            subgroup_rankings = {}

            for demo_col, group_col in demo_groups.items():
                if group_col not in demo_sample.columns:
                    continue

                for group_name, gdf in demo_sample.groupby(group_col):
                    if len(gdf) < 10:
                        continue
                    group_idx = gdf.index.values
                    group_shap = shap_vals[group_idx]
                    group_importance = np.abs(group_shap).mean(axis=0)
                    group_rank = pd.Series(
                        group_importance, index=feature_names
                    ).rank(ascending=False)

                    subgroup_rankings[(demo_col, str(group_name))] = group_rank

                    for i, feat in enumerate(feature_names):
                        all_importance_rows.append({
                            "task": task, "model": model_name,
                            "demographic_col": demo_col,
                            "group": str(group_name),
                            "feature": feat,
                            "mean_abs_shap": group_importance[i],
                            "rank": int(group_rank[feat]),
                        })

                # Compute rank shifts between subgroups within the same demographic
                group_keys = [
                    k for k in subgroup_rankings if k[0] == demo_col
                ]
                for i in range(len(group_keys)):
                    for j in range(i + 1, len(group_keys)):
                        g1 = group_keys[i]
                        g2 = group_keys[j]
                        r1 = subgroup_rankings[g1]
                        r2 = subgroup_rankings[g2]

                        # Spearman correlation of rankings
                        corr, pval = spearmanr(r1.values, r2.values)

                        for feat in feature_names:
                            shift = abs(r1[feat] - r2[feat])
                            flag = "RANK_SHIFT" if shift > RANK_SHIFT_THRESHOLD else ""
                            all_rank_shift_rows.append({
                                "task": task, "model": model_name,
                                "demographic_col": demo_col,
                                "group_1": g1[1], "group_2": g2[1],
                                "feature": feat,
                                "rank_group_1": int(r1[feat]),
                                "rank_group_2": int(r2[feat]),
                                "rank_shift": int(shift),
                                "flag": flag,
                                "spearman_corr": corr,
                                "spearman_pval": pval,
                            })

    importance_df = pd.DataFrame(all_importance_rows)
    rank_shift_df = pd.DataFrame(all_rank_shift_rows)

    importance_path = BIAS_OUTPUT_DIR / "shap_subgroup_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  SHAP importance saved to {importance_path}")

    rank_shift_path = BIAS_OUTPUT_DIR / "shap_rank_shifts.csv"
    rank_shift_df.to_csv(rank_shift_path, index=False)

    if not rank_shift_df.empty:
        flagged = rank_shift_df[rank_shift_df["flag"] == "RANK_SHIFT"]
        print(f"  {len(flagged)} feature×subgroup pairs with rank shift > {RANK_SHIFT_THRESHOLD}")

    return importance_df, rank_shift_df


if __name__ == "__main__":
    analyze_shap_subgroups()
