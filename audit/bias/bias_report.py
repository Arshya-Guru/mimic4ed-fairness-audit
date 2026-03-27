"""
Generate a summary report of all bias findings in markdown and CSV.

Aggregates results from subgroup_performance, fairness_metrics, and
shap_subgroup into a cohesive bias audit report.
"""

import pandas as pd
from pathlib import Path

from audit.config import BIAS_OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs


def generate_bias_report():
    """Compile all bias analysis results into a markdown report.

    Reads CSVs produced by subgroup_performance, fairness_metrics, and
    shap_subgroup, and produces a structured markdown summary.

    Returns:
        String containing the full markdown report.
    """
    ensure_output_dirs()
    report_lines = ["# Bias Audit Report\n"]

    # ── Section 1: Subgroup Performance ──────────────────────────────────
    report_lines.append("## 1. Subgroup Performance Metrics\n")
    perf_path = BIAS_OUTPUT_DIR / "subgroup_performance.csv"
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        flagged = perf[perf["flag"].notna() & (perf["flag"] != "")]

        report_lines.append(
            f"Total metric evaluations: {len(perf)}  \n"
            f"Flagged sensitivity drops: **{len(flagged)}**\n"
        )

        if len(flagged) > 0:
            report_lines.append("### Flagged Sensitivity Drops\n")
            report_lines.append(
                "| Task | Model | Demographic | Group | Sensitivity | CI | Flag |"
            )
            report_lines.append(
                "|------|-------|-------------|-------|-------------|-----|------|"
            )
            for _, row in flagged.iterrows():
                report_lines.append(
                    f"| {row['task']} | {row['model']} | "
                    f"{row['demographic_col']} | {row['group']} | "
                    f"{row['value']:.3f} | [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] | "
                    f"{row['flag']} |"
                )
            report_lines.append("")

        # AUROC summary table
        auroc = perf[perf["metric"] == "auroc"].copy()
        if not auroc.empty:
            report_lines.append("### AUROC by Subgroup\n")
            pivot = auroc.pivot_table(
                index=["demographic_col", "group"],
                columns=["task", "model"],
                values="value",
            )
            report_lines.append(pivot.to_markdown())
            report_lines.append("")
    else:
        report_lines.append("*Subgroup performance data not found.*\n")

    # ── Section 2: Fairness Metrics ──────────────────────────────────────
    report_lines.append("## 2. Fairness Metrics\n")
    fair_path = BIAS_OUTPUT_DIR / "fairness_metrics.csv"
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        violations = fair[fair["flags"].notna() & (fair["flags"] != "")]

        report_lines.append(
            f"Total evaluations: {len(fair)}  \n"
            f"Fairness violations: **{len(violations)}**\n"
        )

        report_lines.append("### Fairness Metrics Summary\n")
        report_lines.append(
            "| Task | Model | Demographic | DP Diff | EO Diff | EOp Diff | Flags |"
        )
        report_lines.append(
            "|------|-------|-------------|---------|---------|----------|-------|"
        )
        for _, row in fair.iterrows():
            report_lines.append(
                f"| {row['task']} | {row['model']} | "
                f"{row['demographic_col']} | "
                f"{row['demographic_parity_diff']:.3f} | "
                f"{row['equalized_odds_diff']:.3f} | "
                f"{row['equal_opportunity_diff']:.3f} | "
                f"{row['flags']} |"
            )
        report_lines.append("")
    else:
        report_lines.append("*Fairness metrics data not found.*\n")

    # ── Section 3: SHAP Subgroup Analysis ────────────────────────────────
    report_lines.append("## 3. SHAP Feature Importance by Subgroup\n")
    rank_path = BIAS_OUTPUT_DIR / "shap_rank_shifts.csv"
    if rank_path.exists():
        ranks = pd.read_csv(rank_path)
        flagged_ranks = ranks[ranks["flag"] == "RANK_SHIFT"]

        report_lines.append(
            f"Total feature×subgroup comparisons: {len(ranks)}  \n"
            f"Features with rank shift > 5: **{len(flagged_ranks)}**\n"
        )

        if len(flagged_ranks) > 0:
            report_lines.append("### Top Rank Shifts\n")
            top_shifts = flagged_ranks.nlargest(20, "rank_shift")
            report_lines.append(
                "| Task | Model | Demographic | Group 1 | Group 2 | Feature | Shift | Spearman r |"
            )
            report_lines.append(
                "|------|-------|-------------|---------|---------|---------|-------|------------|"
            )
            for _, row in top_shifts.iterrows():
                report_lines.append(
                    f"| {row['task']} | {row['model']} | "
                    f"{row['demographic_col']} | {row['group_1']} | "
                    f"{row['group_2']} | {row['feature']} | "
                    f"{row['rank_shift']} | {row['spearman_corr']:.3f} |"
                )
            report_lines.append("")
    else:
        report_lines.append("*SHAP rank shift data not found.*\n")

    # ── Section 4: Demographic Variant Comparison ────────────────────────
    report_lines.append("## 4. Demographic-Aware vs Agnostic Comparison\n")
    var_path = BIAS_OUTPUT_DIR / "demographic_variant_comparison.csv"
    if var_path.exists():
        var_df = pd.read_csv(var_path)
        report_lines.append(var_df.to_markdown(index=False))
        report_lines.append("")
    else:
        report_lines.append("*Variant comparison data not found.*\n")

    report_text = "\n".join(report_lines)

    out_path = REPORTS_DIR / "bias_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"  Bias report saved to {out_path}")

    return report_text


if __name__ == "__main__":
    generate_bias_report()
