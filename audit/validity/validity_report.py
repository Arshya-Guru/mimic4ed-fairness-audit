"""
Generate summary report of clinical validity findings.
"""

import pandas as pd

from audit.config import VALIDITY_OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs


def generate_validity_report():
    """Compile validity analysis results into a markdown report.

    Returns:
        String containing the full markdown report.
    """
    ensure_output_dirs()
    lines = ["# Clinical Validity Report\n"]

    # ── Overall correlations ─────────────────────────────────────────────
    lines.append("## 1. Prediction-Severity Correlations (Overall)\n")
    overall_path = VALIDITY_OUTPUT_DIR / "outcome_correlations.csv"
    if overall_path.exists():
        df = pd.read_csv(overall_path)
        lines.append(
            "| Task | Model | Severity Indicator | Spearman r | p-value | N |"
        )
        lines.append(
            "|------|-------|--------------------|------------|---------|---|"
        )
        for _, row in df.iterrows():
            lines.append(
                f"| {row['task']} | {row['model']} | "
                f"{row['severity_indicator']} | "
                f"{row['spearman_r']:.3f} | {row['p_value']:.2e} | "
                f"{row['n_samples']} |"
            )
        lines.append("")
    else:
        lines.append("*Overall correlation data not found.*\n")

    # ── Subgroup correlations ────────────────────────────────────────────
    lines.append("## 2. Prediction-Severity Correlations by Subgroup\n")
    sub_path = VALIDITY_OUTPUT_DIR / "subgroup_correlations.csv"
    if sub_path.exists():
        df = pd.read_csv(sub_path)
        flagged = df[df["flag"] == "LOW_CORRELATION"]
        lines.append(
            f"Total evaluations: {len(df)}  \n"
            f"Low correlation flags (|r| < 0.3): **{len(flagged)}**\n"
        )

        if len(flagged) > 0:
            lines.append("### Flagged Low Correlations\n")
            lines.append(
                "| Task | Model | Severity | Demographic | Group | r | N |"
            )
            lines.append(
                "|------|-------|----------|-------------|-------|---|---|"
            )
            for _, row in flagged.iterrows():
                lines.append(
                    f"| {row['task']} | {row['model']} | "
                    f"{row['severity_indicator']} | "
                    f"{row['demographic_col']} | {row['group']} | "
                    f"{row['spearman_r']:.3f} | {row['n_samples']} |"
                )
            lines.append("")
    else:
        lines.append("*Subgroup correlation data not found.*\n")

    # ── Correlation comparisons ──────────────────────────────────────────
    lines.append("## 3. Cross-Group Correlation Comparisons\n")
    comp_path = VALIDITY_OUTPUT_DIR / "correlation_comparisons.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        sig = df[df["significant"] == True]
        lines.append(
            f"Total pairwise comparisons: {len(df)}  \n"
            f"Significant differences (p < 0.05): **{len(sig)}**\n"
        )
        if len(sig) > 0:
            lines.append(
                "| Task | Model | Severity | Demo | G1 | G2 | r1 | r2 | p |"
            )
            lines.append(
                "|------|-------|----------|------|----|----|----|----|----|"
            )
            for _, row in sig.iterrows():
                lines.append(
                    f"| {row['task']} | {row['model']} | "
                    f"{row['severity_indicator']} | {row['demographic_col']} | "
                    f"{row['group_1']} | {row['group_2']} | "
                    f"{row['r_group_1']:.3f} | {row['r_group_2']:.3f} | "
                    f"{row['p_value']:.3e} |"
                )
            lines.append("")
    else:
        lines.append("*Correlation comparison data not found.*\n")

    report = "\n".join(lines)
    out_path = REPORTS_DIR / "validity_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  Validity report saved to {out_path}")
    return report


if __name__ == "__main__":
    generate_validity_report()
