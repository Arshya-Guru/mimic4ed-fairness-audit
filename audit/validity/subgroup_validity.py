"""
Test whether prediction-severity correlations hold equally across demographic groups.

Runs the same correlations as outcome_correlation.py but stratified by
demographic group. Uses Fisher z-transformation to compare correlation
strengths across groups. Flags groups where correlation drops below r=0.3.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from audit.config import (
    TASKS, MODELS, VALIDITY_CORRELATION_THRESHOLD,
    VALIDITY_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.utils import load_predictions, add_demographic_groups


def _fisher_z(r):
    """Fisher z-transformation for correlation coefficients.

    Args:
        r: Correlation coefficient.

    Returns:
        Fisher z-transformed value.
    """
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def _compare_correlations(r1, n1, r2, n2):
    """Test whether two Spearman correlations differ significantly.

    Uses Fisher z-transformation to compare two independent correlations.

    Args:
        r1: First correlation coefficient.
        n1: Sample size for first group.
        r2: Second correlation coefficient.
        n2: Sample size for second group.

    Returns:
        Tuple of (z_statistic, p_value).
    """
    from scipy.stats import norm
    z1 = _fisher_z(r1)
    z2 = _fisher_z(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val


def compute_subgroup_validity(tasks=None, models=None, variant="default"):
    """Compute stratified prediction-severity correlations by demographic group.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant.

    Returns:
        Tuple of (correlations_df, comparison_df).
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS

    severity_cols = {
        "triage_acuity": "ESI Acuity",
        "ed_los": "ED Length of Stay",
        "outcome_inhospital_mortality": "In-Hospital Mortality",
    }

    corr_rows = []
    comparison_rows = []

    for task in tasks:
        for model_name in models:
            try:
                preds = load_predictions(task, model_name, variant)
            except FileNotFoundError:
                continue

            preds = add_demographic_groups(preds)
            y_prob = preds["y_prob"].values

            demo_groups = {
                "ethnicity": "ethnicity_group",
                "gender": "gender_group",
                "age": "age_group",
                "insurance": "insurance_group",
            }

            for sev_col, sev_label in severity_cols.items():
                if sev_col not in preds.columns:
                    continue

                sev_vals = pd.to_numeric(preds[sev_col], errors="coerce").values

                # Overall correlation (for comparison)
                overall_mask = ~(np.isnan(y_prob) | np.isnan(sev_vals))
                if overall_mask.sum() < 20:
                    continue
                overall_r, _ = spearmanr(y_prob[overall_mask], sev_vals[overall_mask])
                overall_n = int(overall_mask.sum())

                for demo_col, group_col in demo_groups.items():
                    if group_col not in preds.columns:
                        continue

                    group_correlations = {}

                    for group_name, gdf in preds.groupby(group_col):
                        gidx = gdf.index.values
                        gprob = y_prob[gidx]
                        gsev = sev_vals[gidx]
                        mask = ~(np.isnan(gprob) | np.isnan(gsev))

                        if mask.sum() < 10:
                            continue

                        # Skip if either array is constant
                        if np.std(gprob[mask]) == 0 or np.std(gsev[mask]) == 0:
                            continue

                        corr, pval = spearmanr(gprob[mask], gsev[mask])
                        n = int(mask.sum())
                        flag = ""
                        if abs(corr) < VALIDITY_CORRELATION_THRESHOLD:
                            flag = "LOW_CORRELATION"

                        group_correlations[str(group_name)] = (corr, n)

                        corr_rows.append({
                            "task": task, "model": model_name,
                            "severity_indicator": sev_label,
                            "demographic_col": demo_col,
                            "group": str(group_name),
                            "spearman_r": corr,
                            "p_value": pval,
                            "n_samples": n,
                            "flag": flag,
                        })

                    # Pairwise comparison of correlations between groups
                    group_names = list(group_correlations.keys())
                    for i in range(len(group_names)):
                        for j in range(i + 1, len(group_names)):
                            g1, g2 = group_names[i], group_names[j]
                            r1, n1 = group_correlations[g1]
                            r2, n2 = group_correlations[g2]

                            if n1 < 10 or n2 < 10:
                                continue

                            z_stat, z_pval = _compare_correlations(r1, n1, r2, n2)

                            comparison_rows.append({
                                "task": task, "model": model_name,
                                "severity_indicator": sev_label,
                                "demographic_col": demo_col,
                                "group_1": g1, "group_2": g2,
                                "r_group_1": r1, "r_group_2": r2,
                                "z_statistic": z_stat,
                                "p_value": z_pval,
                                "significant": z_pval < 0.05,
                            })

    corr_df = pd.DataFrame(corr_rows)
    comp_df = pd.DataFrame(comparison_rows)

    corr_path = VALIDITY_OUTPUT_DIR / "subgroup_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Subgroup correlations saved to {corr_path}")

    comp_path = VALIDITY_OUTPUT_DIR / "correlation_comparisons.csv"
    comp_df.to_csv(comp_path, index=False)

    flagged = corr_df[corr_df["flag"] == "LOW_CORRELATION"]
    if len(flagged) > 0:
        print(f"  WARNING: {len(flagged)} subgroup(s) with low prediction-severity correlation")

    return corr_df, comp_df


if __name__ == "__main__":
    compute_subgroup_validity()
