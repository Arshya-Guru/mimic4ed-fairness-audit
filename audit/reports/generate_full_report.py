"""
Compile all audit findings into a single comprehensive markdown report
with embedded plot images.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from audit.config import (
    BIAS_OUTPUT_DIR, EXPLAINABILITY_OUTPUT_DIR, VALIDITY_OUTPUT_DIR,
    PLOTS_DIR, REPORTS_DIR, TASK_DISPLAY_NAME, MODEL_DISPLAY_NAME,
    DEMOGRAPHIC_PARITY_THRESHOLD, ensure_output_dirs,
)


def _count_violations(fair_path):
    """Count fairness violations from the metrics CSV.

    Args:
        fair_path: Path to fairness_metrics.csv.

    Returns:
        Tuple of (total_evaluations, violation_count, violation_details_str).
    """
    if not fair_path.exists():
        return 0, 0, ""
    df = pd.read_csv(fair_path)
    violations = df[df["flags"].notna() & (df["flags"] != "")]
    details = []
    for _, row in violations.iterrows():
        details.append(
            f"- **{row['model']}** on **{row['task']}** "
            f"({row['demographic_col']}): {row['flags']}"
        )
    return len(df), len(violations), "\n".join(details)


def _count_sensitivity_flags(perf_path):
    """Count sensitivity drop flags.

    Args:
        perf_path: Path to subgroup_performance.csv.

    Returns:
        Tuple of (total, flagged_count).
    """
    if not perf_path.exists():
        return 0, 0
    df = pd.read_csv(perf_path)
    flagged = df[df["flag"].notna() & (df["flag"] != "")]
    return len(df), len(flagged)


def _embed_plots(section_prefix):
    """Find and format markdown image links for plots matching a prefix.

    Args:
        section_prefix: Filename prefix to match (e.g., 'performance_', 'fairness_').

    Returns:
        String with markdown image references.
    """
    if not PLOTS_DIR.exists():
        return ""
    pngs = sorted(PLOTS_DIR.glob(f"{section_prefix}*.png"))
    if not pngs:
        return "*No plots generated for this section.*\n"
    lines = []
    for p in pngs:
        rel_path = f"../output/plots/{p.name}"
        caption = p.stem.replace("_", " ").title()
        lines.append(f"![{caption}]({rel_path})\n")
    return "\n".join(lines)


def generate_full_report():
    """Compile all audit findings into a comprehensive markdown report.

    Sections:
    1. Executive Summary
    2. Bias Audit Results
    3. Explainability Assessment
    4. Clinical Validity
    5. Demographic-Aware vs Agnostic Comparison
    6. Recommendations

    Returns:
        Path to the generated report file.
    """
    ensure_output_dirs()
    lines = []

    # ── Title ────────────────────────────────────────────────────────────
    lines.append("# MIMIC-IV-ED Fairness, Explainability & Clinical Validity Audit Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")

    # ── 1. Executive Summary ─────────────────────────────────────────────
    lines.append("## 1. Executive Summary\n")

    fair_path = BIAS_OUTPUT_DIR / "fairness_metrics.csv"
    total_fair, n_violations, violation_details = _count_violations(fair_path)

    perf_path = BIAS_OUTPUT_DIR / "subgroup_performance.csv"
    total_perf, n_sens_flags = _count_sensitivity_flags(perf_path)

    rank_path = BIAS_OUTPUT_DIR / "shap_rank_shifts.csv"
    n_rank_shifts = 0
    if rank_path.exists():
        rdf = pd.read_csv(rank_path)
        n_rank_shifts = len(rdf[rdf["flag"] == "RANK_SHIFT"])

    sub_corr_path = VALIDITY_OUTPUT_DIR / "subgroup_correlations.csv"
    n_low_corr = 0
    if sub_corr_path.exists():
        cdf = pd.read_csv(sub_corr_path)
        n_low_corr = len(cdf[cdf["flag"] == "LOW_CORRELATION"])

    lines.append("### Key Findings\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Fairness violations (DP/EO/EOp > {DEMOGRAPHIC_PARITY_THRESHOLD}) | **{n_violations}** out of {total_fair} evaluations |")
    lines.append(f"| Sensitivity drops > 5% below population | **{n_sens_flags}** out of {total_perf} evaluations |")
    lines.append(f"| SHAP feature rank shifts > 5 positions | **{n_rank_shifts}** |")
    lines.append(f"| Subgroups with low validity correlation (|r| < 0.3) | **{n_low_corr}** |")
    lines.append("")

    if n_violations > 0:
        lines.append("### Fairness Violations Detail\n")
        lines.append(violation_details)
        lines.append("")

    # ── 2. Bias Audit Results ────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 2. Bias Audit Results\n")

    lines.append("### 2.1 Subgroup Performance\n")
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        auroc = perf[(perf["metric"] == "auroc") & (perf["demographic_col"] != "overall")]
        if not auroc.empty:
            pivot = auroc.pivot_table(
                index=["demographic_col", "group"],
                columns=["task", "model"],
                values="value",
            )
            lines.append(pivot.round(3).to_markdown())
            lines.append("")
    else:
        lines.append("*Data not available.*\n")

    lines.append(_embed_plots("performance_"))

    lines.append("### 2.2 Fairness Metrics\n")
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        display_cols = ["task", "model", "demographic_col",
                        "demographic_parity_diff", "equalized_odds_diff",
                        "equal_opportunity_diff", "flags"]
        lines.append(fair[display_cols].round(3).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("*Data not available.*\n")

    lines.append(_embed_plots("fairness_"))

    lines.append("### 2.3 SHAP Feature Importance by Subgroup\n")
    if rank_path.exists():
        rdf = pd.read_csv(rank_path)
        flagged = rdf[rdf["flag"] == "RANK_SHIFT"]
        if not flagged.empty:
            top = flagged.nlargest(15, "rank_shift")
            lines.append(top[["task", "model", "demographic_col", "group_1", "group_2",
                              "feature", "rank_shift", "spearman_corr"]].round(3).to_markdown(index=False))
            lines.append("")
        else:
            lines.append("No significant rank shifts detected.\n")
    else:
        lines.append("*Data not available.*\n")

    lines.append(_embed_plots("shap_comparison_"))

    # ── 3. Explainability Assessment ─────────────────────────────────────
    lines.append("---\n")
    lines.append("## 3. Explainability Assessment\n")

    quality_path = EXPLAINABILITY_OUTPUT_DIR / "explanation_quality.csv"
    if quality_path.exists():
        qdf = pd.read_csv(quality_path)
        lines.append("### Explanation Quality Metrics\n")
        lines.append(qdf.round(3).to_markdown(index=False))
        lines.append("")

        # Summary statistics
        jaccard = qdf[qdf["metric"] == "top5_feature_agreement_jaccard"]
        if not jaccard.empty:
            lines.append(
                f"**Mean top-5 feature agreement (Jaccard):** "
                f"{jaccard['value'].mean():.3f}\n"
            )
        consistency = qdf[qdf["metric"] == "shap_consistency_spearman"]
        if not consistency.empty:
            lines.append(
                f"**Mean SHAP consistency (Spearman):** "
                f"{consistency['value'].mean():.3f}\n"
            )
    else:
        lines.append("*Explanation quality data not available.*\n")

    clinician_path = EXPLAINABILITY_OUTPUT_DIR / "clinician_review.html"
    if clinician_path.exists():
        rel_path = "../output/explainability/clinician_review.html"
        lines.append(
            f"\nClinician review document available: [{clinician_path.name}]({rel_path})\n"
        )

    # ── 4. Clinical Validity ─────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 4. Clinical Validity\n")

    overall_corr_path = VALIDITY_OUTPUT_DIR / "outcome_correlations.csv"
    if overall_corr_path.exists():
        lines.append("### 4.1 Overall Prediction-Severity Correlations\n")
        ocdf = pd.read_csv(overall_corr_path)
        lines.append(ocdf.round(4).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("*Overall correlation data not available.*\n")

    if sub_corr_path.exists():
        lines.append("### 4.2 Subgroup Validity\n")
        scdf = pd.read_csv(sub_corr_path)
        flagged = scdf[scdf["flag"] == "LOW_CORRELATION"]
        if not flagged.empty:
            lines.append("**Flagged low-correlation subgroups:**\n")
            lines.append(flagged.round(3).to_markdown(index=False))
            lines.append("")
        else:
            lines.append("No subgroups flagged for low prediction-severity correlation.\n")

    comp_path = VALIDITY_OUTPUT_DIR / "correlation_comparisons.csv"
    if comp_path.exists():
        lines.append("### 4.3 Cross-Group Correlation Comparisons\n")
        ccdf = pd.read_csv(comp_path)
        sig = ccdf[ccdf["significant"] == True]
        if not sig.empty:
            lines.append(f"**{len(sig)} significant differences found:**\n")
            lines.append(sig.round(4).to_markdown(index=False))
            lines.append("")
        else:
            lines.append("No significant differences in correlation strength between groups.\n")

    lines.append(_embed_plots("validity_scatter_"))

    # ── 5. Demographic-Aware vs Agnostic ─────────────────────────────────
    lines.append("---\n")
    lines.append("## 5. Demographic-Aware vs Agnostic Model Comparison\n")
    var_path = BIAS_OUTPUT_DIR / "demographic_variant_comparison.csv"
    if var_path.exists():
        vdf = pd.read_csv(var_path)
        lines.append(vdf.round(4).to_markdown(index=False))
        lines.append("")

        if "auroc_difference" in vdf.columns:
            mean_diff = vdf["auroc_difference"].mean()
            lines.append(
                f"\n**Mean AUROC difference (aware - agnostic):** {mean_diff:.4f}\n"
            )
            lines.append(
                "A near-zero difference suggests bias is encoded through proxy features "
                "even without explicit demographic variables.\n"
            )
    else:
        lines.append("*Variant comparison data not available.*\n")

    # ── 6. Recommendations ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 6. Recommendations\n")
    recommendations = []

    if n_violations > 0:
        recommendations.append(
            "- **Address fairness violations:** Models with demographic parity or "
            "equalized odds violations should be retrained with fairness constraints "
            "(e.g., Fairlearn's ThresholdOptimizer or adversarial debiasing)."
        )
    if n_sens_flags > 0:
        recommendations.append(
            "- **Investigate sensitivity disparities:** Subgroups with significantly "
            "lower sensitivity may be underserved by current models. Consider "
            "oversampling or specialized calibration for these populations."
        )
    if n_rank_shifts > 0:
        recommendations.append(
            "- **Review feature importance shifts:** Large rank shifts in SHAP values "
            "across demographics suggest the model relies on different decision-making "
            "pathways for different populations, which may indicate proxy discrimination."
        )
    if n_low_corr > 0:
        recommendations.append(
            "- **Validate clinical utility per subgroup:** Groups with low "
            "prediction-severity correlation may not benefit from these models. "
            "Consider subgroup-specific model calibration or alternative prediction approaches."
        )

    if not recommendations:
        recommendations.append(
            "- No major issues detected. Continue monitoring with each model update."
        )

    recommendations.append(
        "- **Conduct clinician review:** Use the generated clinician review document "
        "to gather expert feedback on model explanations through focus group sessions."
    )
    recommendations.append(
        "- **Longitudinal monitoring:** Re-run this audit periodically as data "
        "distributions may shift over time (dataset drift)."
    )

    lines.append("\n".join(recommendations))
    lines.append("")

    # ── Write report ─────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    out_path = REPORTS_DIR / "full_audit_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"  Full audit report saved to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_full_report()
