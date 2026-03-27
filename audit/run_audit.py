#!/usr/bin/env python3
"""
Master script that runs the full fairness, explainability, and clinical
validity audit pipeline end-to-end.

Usage:
    python -m audit.run_audit                          # Full audit on real data
    python -m audit.run_audit --synthetic              # Demo on synthetic data
    python -m audit.run_audit --tasks hospitalization   # Single task
    python -m audit.run_audit --models lr rf            # Subset of models
    python -m audit.run_audit --skip-shap              # Skip slow SHAP analysis
"""

import argparse
import time
import sys
from pathlib import Path

from audit.config import (
    TASKS, MODELS, TRAIN_CSV, TEST_CSV,
    ensure_output_dirs,
)


def parse_args():
    """Parse command-line arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="MIMIC-IV-ED Fairness Audit Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data for testing/demo (no MIMIC access needed)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        choices=TASKS,
        help=f"Tasks to audit. Default: all ({', '.join(TASKS)})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=MODELS,
        help=f"Models to audit. Default: all ({', '.join(MODELS)})",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP-based analyses (faster, but no explainability results)",
    )
    parser.add_argument(
        "--skip-variants", action="store_true",
        help="Skip demographic-aware vs agnostic variant training",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=5000,
        help="Number of synthetic samples to generate (default: 5000)",
    )
    return parser.parse_args()


def main():
    """Run the full audit pipeline."""
    args = parse_args()
    tasks = args.tasks or TASKS
    models = args.models or MODELS
    start_time = time.time()

    print("=" * 70)
    print("MIMIC-IV-ED FAIRNESS, EXPLAINABILITY & CLINICAL VALIDITY AUDIT")
    print("=" * 70)
    print(f"Tasks:  {', '.join(tasks)}")
    print(f"Models: {', '.join(models)}")
    print(f"Mode:   {'SYNTHETIC' if args.synthetic else 'REAL DATA'}")
    print("=" * 70)

    ensure_output_dirs()

    # ── Determine data paths ─────────────────────────────────────────────
    if args.synthetic:
        print("\n[1/6] Generating synthetic data...")
        from audit.tests.test_with_synthetic import save_synthetic_data
        train_csv, test_csv = save_synthetic_data(
            n_samples=args.n_synthetic, seed=42
        )
    else:
        train_csv = TRAIN_CSV
        test_csv = TEST_CSV
        if not train_csv.exists() or not test_csv.exists():
            print(f"ERROR: Data files not found at {train_csv} and {test_csv}")
            print("Run with --synthetic flag to use synthetic data, or set "
                  "AUDIT_TRAIN_CSV and AUDIT_TEST_CSV environment variables.")
            sys.exit(1)
        print("\n[1/6] Using existing data files")
        print(f"  Train: {train_csv}")
        print(f"  Test:  {test_csv}")

    # ── Train models ─────────────────────────────────────────────────────
    print("\n[2/6] Training models...")
    from audit.models.train_models import train_all
    train_all(tasks=tasks, models=models, train_csv=train_csv, test_csv=test_csv)

    if not args.skip_variants:
        print("\n  Training demographic variants...")
        from audit.models.demographic_variants import (
            train_demographic_variants, compare_variants,
        )
        train_demographic_variants(
            tasks=tasks, models=models,
            train_csv=train_csv, test_csv=test_csv,
        )
        compare_variants(tasks=tasks, models=models)

    # ── Bias audit ───────────────────────────────────────────────────────
    print("\n[3/6] Running bias audit...")
    from audit.bias.subgroup_performance import compute_subgroup_performance
    compute_subgroup_performance(tasks=tasks, models=models)

    from audit.bias.fairness_metrics import compute_fairness_metrics
    compute_fairness_metrics(tasks=tasks, models=models)

    if not args.skip_shap:
        print("\n  Computing SHAP subgroup analysis (this may take a while)...")
        from audit.bias.shap_subgroup import analyze_shap_subgroups
        analyze_shap_subgroups(
            tasks=tasks, models=models,
            train_csv=train_csv, test_csv=test_csv,
        )

    from audit.bias.bias_report import generate_bias_report
    generate_bias_report()

    # ── Explainability ───────────────────────────────────────────────────
    print("\n[4/6] Running explainability analysis...")
    if not args.skip_shap:
        from audit.explainability.shap_explanations import generate_individual_explanations
        generate_individual_explanations(
            tasks=tasks, models=models,
            train_csv=train_csv, test_csv=test_csv,
        )

        from audit.explainability.explanation_quality import compute_explanation_quality
        compute_explanation_quality()

        from audit.explainability.clinician_review import generate_clinician_review_html
        generate_clinician_review_html()
    else:
        print("  Skipped (--skip-shap flag)")

    # ── Validity ─────────────────────────────────────────────────────────
    print("\n[5/6] Running validity analysis...")
    from audit.validity.outcome_correlation import compute_outcome_correlations
    compute_outcome_correlations(tasks=tasks, models=models)

    from audit.validity.subgroup_validity import compute_subgroup_validity
    compute_subgroup_validity(tasks=tasks, models=models)

    from audit.validity.validity_report import generate_validity_report
    generate_validity_report()

    # ── Visualizations ───────────────────────────────────────────────────
    print("\n[6/6] Generating visualizations and report...")
    from audit.visualization.performance_plots import plot_subgroup_performance
    plot_subgroup_performance()

    from audit.visualization.fairness_plots import plot_fairness_radar
    plot_fairness_radar()

    if not args.skip_shap:
        from audit.visualization.shap_comparison import plot_shap_comparison
        plot_shap_comparison()

    from audit.visualization.validity_plots import plot_validity_scatter
    plot_validity_scatter(tasks=tasks, models=models)

    # ── Final report ─────────────────────────────────────────────────────
    from audit.reports.generate_full_report import generate_full_report
    report_path = generate_full_report()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE in {elapsed:.1f}s")
    print(f"Full report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
