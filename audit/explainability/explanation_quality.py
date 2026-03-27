"""
Metrics for explanation quality: feature agreement across subgroups,
stability, and consistency of SHAP explanations.
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

from audit.config import (
    EXPLAINABILITY_OUTPUT_DIR, BIAS_OUTPUT_DIR, ensure_output_dirs,
)


def compute_explanation_quality():
    """Compute explanation quality metrics from individual explanations and SHAP importance.

    Metrics computed:
    1. Feature agreement: How often do the top-5 features agree across demographic subgroups?
    2. Stability: Variance of SHAP values for the same feature across patients in a subgroup.
    3. Consistency: Spearman correlation of feature rankings between subgroups.

    Returns:
        DataFrame with quality metrics.
    """
    ensure_output_dirs()
    rows = []

    # Load individual explanations
    expl_path = EXPLAINABILITY_OUTPUT_DIR / "individual_explanations.json"
    if not expl_path.exists():
        print("  Individual explanations not found, skipping quality metrics")
        return pd.DataFrame()

    with open(expl_path) as f:
        explanations = json.load(f)

    # Group explanations by task × model
    grouped = defaultdict(list)
    for ex in explanations:
        key = (ex["task"], ex["model"])
        grouped[key].append(ex)

    for (task, model), expl_list in grouped.items():
        # Group by demographic
        demo_groups = defaultdict(lambda: defaultdict(list))
        for ex in expl_list:
            for demo_col in ["ethnicity", "gender"]:
                if demo_col in ex["demographics"]:
                    group = ex["demographics"][demo_col]
                    demo_groups[demo_col][group].append(ex)

        for demo_col, groups in demo_groups.items():
            group_names = list(groups.keys())
            if len(group_names) < 2:
                continue

            # 1. Feature agreement: Jaccard similarity of top-5 features
            group_top5 = {}
            for gname, gexpl in groups.items():
                all_top5 = set()
                for ex in gexpl:
                    for feat in ex["top_5_features"]:
                        all_top5.add(feat["feature"])
                group_top5[gname] = all_top5

            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1, g2 = group_names[i], group_names[j]
                    s1, s2 = group_top5[g1], group_top5[g2]
                    intersection = len(s1 & s2)
                    union = len(s1 | s2)
                    jaccard = intersection / union if union > 0 else 0

                    rows.append({
                        "task": task, "model": model,
                        "demographic_col": demo_col,
                        "group_1": g1, "group_2": g2,
                        "metric": "top5_feature_agreement_jaccard",
                        "value": jaccard,
                    })

            # 2. Stability: mean variance of SHAP values within each subgroup
            for gname, gexpl in groups.items():
                if len(gexpl) < 2:
                    continue
                feature_names = list(gexpl[0]["shap_values"].keys())
                shap_matrix = np.array([
                    [ex["shap_values"][f] for f in feature_names]
                    for ex in gexpl
                ])
                mean_var = np.mean(np.var(shap_matrix, axis=0))
                rows.append({
                    "task": task, "model": model,
                    "demographic_col": demo_col,
                    "group_1": gname, "group_2": "",
                    "metric": "shap_stability_mean_variance",
                    "value": float(mean_var),
                })

            # 3. Consistency: Spearman correlation of mean |SHAP| between groups
            group_mean_shap = {}
            for gname, gexpl in groups.items():
                if len(gexpl) == 0:
                    continue
                feature_names = list(gexpl[0]["shap_values"].keys())
                shap_matrix = np.array([
                    [abs(ex["shap_values"][f]) for f in feature_names]
                    for ex in gexpl
                ])
                group_mean_shap[gname] = np.mean(shap_matrix, axis=0)

            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1, g2 = group_names[i], group_names[j]
                    if g1 in group_mean_shap and g2 in group_mean_shap:
                        corr, pval = spearmanr(
                            group_mean_shap[g1], group_mean_shap[g2]
                        )
                        rows.append({
                            "task": task, "model": model,
                            "demographic_col": demo_col,
                            "group_1": g1, "group_2": g2,
                            "metric": "shap_consistency_spearman",
                            "value": float(corr),
                        })

    df = pd.DataFrame(rows)
    out_path = EXPLAINABILITY_OUTPUT_DIR / "explanation_quality.csv"
    df.to_csv(out_path, index=False)
    print(f"  Explanation quality metrics saved to {out_path}")
    return df


if __name__ == "__main__":
    compute_explanation_quality()
