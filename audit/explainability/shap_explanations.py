"""
Generate SHAP explanations for individual patient predictions.

For a stratified sample of patients (N per subgroup), generates full SHAP
waterfall data and saves individual explanations as JSON.
"""

import json
import numpy as np
import pandas as pd

from audit.config import (
    TASKS, MODELS, CLINICIAN_SAMPLE_PER_SUBGROUP, RANDOM_SEED,
    SHAP_SAMPLE_SIZE, TRAIN_CSV, TEST_CSV,
    EXPLAINABILITY_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.utils import load_model, load_data, add_demographic_groups
from audit.bias.shap_subgroup import compute_shap_values


def generate_individual_explanations(tasks=None, models=None, variant="default",
                                       n_per_subgroup=None,
                                       train_csv=None, test_csv=None):
    """Generate per-patient SHAP explanations stratified by demographic subgroup.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant.
        n_per_subgroup: Number of patients per subgroup to explain. Defaults to config value.
        train_csv: Path to training data.
        test_csv: Path to test data.

    Returns:
        List of all patient explanation dicts.
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    n_per_subgroup = n_per_subgroup or CLINICIAN_SAMPLE_PER_SUBGROUP
    train_csv = train_csv or TRAIN_CSV
    test_csv = test_csv or TEST_CSV

    all_explanations = []

    for task in tasks:
        X_train, y_train, X_test, y_test, df_test = load_data(
            train_csv, test_csv, task
        )
        df_test_demo = add_demographic_groups(df_test)

        for model_name in models:
            print(f"  Generating explanations for {model_name}/{task}...")
            try:
                model = load_model(task, model_name, variant)
            except FileNotFoundError:
                print(f"    Model not found, skipping")
                continue

            # Get predictions
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            feature_names = list(X_test.columns)

            # Stratified sampling: pick n_per_subgroup from each demographic group
            sample_indices = set()
            rng = np.random.RandomState(RANDOM_SEED)

            demo_groups = {
                "ethnicity": "ethnicity_group",
                "gender": "gender_group",
                "age": "age_group",
                "insurance": "insurance_group",
            }

            for demo_col, group_col in demo_groups.items():
                if group_col not in df_test_demo.columns:
                    continue
                for group_name, gdf in df_test_demo.groupby(group_col):
                    if len(gdf) < n_per_subgroup:
                        chosen = gdf.index.tolist()
                    else:
                        chosen = rng.choice(
                            gdf.index.tolist(), size=n_per_subgroup, replace=False
                        ).tolist()
                    sample_indices.update(chosen)

            sample_indices = sorted(sample_indices)
            if len(sample_indices) == 0:
                continue

            X_sample = X_test.iloc[sample_indices].reset_index(drop=True)

            # Compute SHAP values for the sample
            shap_vals = compute_shap_values(model, X_sample, model_name)

            for i, orig_idx in enumerate(sample_indices):
                sv = shap_vals[i]
                # Top 5 features by absolute SHAP value
                top_idx = np.argsort(np.abs(sv))[::-1][:5]
                top_features = [
                    {
                        "feature": feature_names[fi],
                        "value": float(X_test.iloc[orig_idx][feature_names[fi]]),
                        "shap_value": float(sv[fi]),
                    }
                    for fi in top_idx
                ]

                # Build demographics dict
                demographics = {}
                for col in ["ethnicity", "gender", "age", "anchor_age", "insurance"]:
                    if col in df_test.columns:
                        val = df_test.iloc[orig_idx][col]
                        demographics[col] = str(val) if pd.notna(val) else "Unknown"

                explanation = {
                    "patient_index": int(orig_idx),
                    "task": task,
                    "model": model_name,
                    "demographics": demographics,
                    "prediction": float(y_prob[orig_idx]),
                    "true_label": int(y_test[orig_idx]),
                    "shap_values": {
                        feature_names[fi]: float(sv[fi])
                        for fi in range(len(feature_names))
                    },
                    "top_5_features": top_features,
                }
                all_explanations.append(explanation)

    # Save all explanations
    out_path = EXPLAINABILITY_OUTPUT_DIR / "individual_explanations.json"
    EXPLAINABILITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_explanations, f, indent=2)
    print(f"  {len(all_explanations)} individual explanations saved to {out_path}")

    return all_explanations


if __name__ == "__main__":
    generate_individual_explanations()
