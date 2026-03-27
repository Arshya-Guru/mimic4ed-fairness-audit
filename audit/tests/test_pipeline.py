"""
Integration test that runs the full audit pipeline on synthetic data
to verify everything works end-to-end.

Can be run directly or via pytest.
"""

import sys
import shutil
from pathlib import Path

import pandas as pd

from audit.config import (
    AUDIT_ROOT, OUTPUT_DIR, MODELS_OUTPUT_DIR, PREDICTIONS_DIR,
    BIAS_OUTPUT_DIR, EXPLAINABILITY_OUTPUT_DIR, VALIDITY_OUTPUT_DIR,
    PLOTS_DIR, REPORTS_DIR, ensure_output_dirs,
)


def test_full_pipeline():
    """Run the complete audit pipeline on synthetic data and verify outputs.

    This test:
    1. Generates synthetic data
    2. Trains all models
    3. Runs bias analysis
    4. Runs explainability analysis
    5. Runs validity analysis
    6. Generates visualizations
    7. Compiles final report
    8. Verifies all expected output files exist
    """
    print("=" * 70)
    print("INTEGRATION TEST: Full Audit Pipeline on Synthetic Data")
    print("=" * 70)

    # ── Step 0: Clean output directory ───────────────────────────────────
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ensure_output_dirs()

    # ── Step 1: Generate synthetic data ──────────────────────────────────
    print("\n[Step 1/7] Generating synthetic data...")
    from audit.tests.test_with_synthetic import save_synthetic_data
    train_path, test_path = save_synthetic_data(n_samples=2000, seed=42)
    assert train_path.exists(), "Train CSV not created"
    assert test_path.exists(), "Test CSV not created"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) > 0, "Train set is empty"
    assert len(test_df) > 0, "Test set is empty"
    print(f"  OK: {len(train_df)} train, {len(test_df)} test rows")

    # ── Step 2: Train models ─────────────────────────────────────────────
    print("\n[Step 2/7] Training models...")
    from audit.models.train_models import train_all
    results = train_all(
        tasks=["hospitalization", "critical", "ed_reattendance"],
        models=["lr", "rf"],  # Use only LR and RF for speed
        train_csv=train_path,
        test_csv=test_path,
    )
    assert len(results) == 6, f"Expected 6 model results, got {len(results)}"

    # Verify model and prediction files
    model_files = list(MODELS_OUTPUT_DIR.glob("*.pkl"))
    pred_files = list(PREDICTIONS_DIR.glob("*.csv"))
    assert len(model_files) >= 6, f"Expected >= 6 model files, got {len(model_files)}"
    assert len(pred_files) >= 6, f"Expected >= 6 prediction files, got {len(pred_files)}"
    print(f"  OK: {len(model_files)} models, {len(pred_files)} prediction files")

    # ── Step 3: Run bias analysis ────────────────────────────────────────
    print("\n[Step 3/7] Running bias analysis...")
    from audit.bias.subgroup_performance import compute_subgroup_performance
    perf_df = compute_subgroup_performance(
        tasks=["hospitalization", "critical", "ed_reattendance"],
        models=["lr", "rf"],
    )
    assert len(perf_df) > 0, "Subgroup performance is empty"
    assert (BIAS_OUTPUT_DIR / "subgroup_performance.csv").exists()

    from audit.bias.fairness_metrics import compute_fairness_metrics
    fair_df = compute_fairness_metrics(
        tasks=["hospitalization", "critical", "ed_reattendance"],
        models=["lr", "rf"],
    )
    assert len(fair_df) > 0, "Fairness metrics empty"
    assert (BIAS_OUTPUT_DIR / "fairness_metrics.csv").exists()

    print("\n  Running SHAP subgroup analysis (this may take a moment)...")
    from audit.bias.shap_subgroup import analyze_shap_subgroups
    imp_df, rank_df = analyze_shap_subgroups(
        tasks=["hospitalization"],  # Just one task for speed
        models=["lr"],
        train_csv=train_path,
        test_csv=test_path,
    )
    assert len(imp_df) > 0, "SHAP importance empty"
    assert (BIAS_OUTPUT_DIR / "shap_subgroup_importance.csv").exists()

    # Train demographic variants
    print("\n  Training demographic variants...")
    from audit.models.demographic_variants import train_demographic_variants, compare_variants
    train_demographic_variants(
        tasks=["hospitalization"],
        models=["lr"],
        train_csv=train_path,
        test_csv=test_path,
    )
    compare_variants(tasks=["hospitalization"], models=["lr"])

    print("  OK: Bias analysis complete")

    # ── Step 4: Run explainability analysis ──────────────────────────────
    print("\n[Step 4/7] Running explainability analysis...")
    from audit.explainability.shap_explanations import generate_individual_explanations
    explanations = generate_individual_explanations(
        tasks=["hospitalization"],
        models=["lr"],
        n_per_subgroup=3,
        train_csv=train_path,
        test_csv=test_path,
    )
    assert len(explanations) > 0, "No individual explanations generated"
    assert (EXPLAINABILITY_OUTPUT_DIR / "individual_explanations.json").exists()

    from audit.explainability.explanation_quality import compute_explanation_quality
    quality_df = compute_explanation_quality()

    from audit.explainability.clinician_review import generate_clinician_review_html
    html_path = generate_clinician_review_html()
    assert html_path is not None and html_path.exists(), "Clinician review HTML not generated"
    print("  OK: Explainability analysis complete")

    # ── Step 5: Run validity analysis ────────────────────────────────────
    print("\n[Step 5/7] Running validity analysis...")
    from audit.validity.outcome_correlation import compute_outcome_correlations
    corr_df = compute_outcome_correlations(
        tasks=["hospitalization", "critical", "ed_reattendance"],
        models=["lr", "rf"],
    )
    assert (VALIDITY_OUTPUT_DIR / "outcome_correlations.csv").exists()

    from audit.validity.subgroup_validity import compute_subgroup_validity
    sub_corr, comp = compute_subgroup_validity(
        tasks=["hospitalization", "critical"],
        models=["lr", "rf"],
    )
    assert (VALIDITY_OUTPUT_DIR / "subgroup_correlations.csv").exists()
    print("  OK: Validity analysis complete")

    # ── Step 6: Generate visualizations ──────────────────────────────────
    print("\n[Step 6/7] Generating visualizations...")
    from audit.visualization.performance_plots import plot_subgroup_performance
    perf_plots = plot_subgroup_performance()

    from audit.visualization.fairness_plots import plot_fairness_radar
    fair_plots = plot_fairness_radar()

    from audit.visualization.shap_comparison import plot_shap_comparison
    shap_plots = plot_shap_comparison()

    from audit.visualization.validity_plots import plot_validity_scatter
    val_plots = plot_validity_scatter(
        tasks=["hospitalization", "critical"],
        models=["lr", "rf"],
    )

    total_plots = len(perf_plots) + len(fair_plots) + len(shap_plots) + len(val_plots)
    print(f"  OK: {total_plots} plots generated")

    # ── Step 7: Generate reports ─────────────────────────────────────────
    print("\n[Step 7/7] Generating reports...")
    from audit.bias.bias_report import generate_bias_report
    generate_bias_report()
    assert (REPORTS_DIR / "bias_report.md").exists()

    from audit.validity.validity_report import generate_validity_report
    generate_validity_report()
    assert (REPORTS_DIR / "validity_report.md").exists()

    from audit.reports.generate_full_report import generate_full_report
    report_path = generate_full_report()
    assert report_path.exists(), "Full report not generated"

    # Read report to verify content
    with open(report_path) as f:
        report_content = f.read()
    assert "Executive Summary" in report_content
    assert "Bias Audit" in report_content
    assert "Clinical Validity" in report_content
    assert "Recommendations" in report_content
    print("  OK: All reports generated")

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    # Print output inventory
    all_outputs = list(OUTPUT_DIR.rglob("*"))
    files = [f for f in all_outputs if f.is_file()]
    print(f"\nTotal output files: {len(files)}")
    for subdir in [MODELS_OUTPUT_DIR, PREDICTIONS_DIR, BIAS_OUTPUT_DIR,
                   EXPLAINABILITY_OUTPUT_DIR, VALIDITY_OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR]:
        count = len(list(subdir.rglob("*"))) if subdir.exists() else 0
        print(f"  {subdir.name}/: {count} files")

    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
