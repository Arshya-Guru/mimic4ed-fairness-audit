# MIMIC-IV-ED Fairness, Explainability & Clinical Validity Audit

A comprehensive audit framework built on top of the [Xie et al. MIMIC-IV-ED benchmark](https://github.com/nliulab/mimic4ed-benchmark). Evaluates ML models for emergency department prediction tasks across three dimensions: **bias/fairness**, **explainability**, and **clinical validity**.

## Quick Start (Synthetic Data Demo)

No MIMIC credentials needed — run the full pipeline on synthetic data:

```bash
# Install dependencies
pixi install

# Run the full audit on synthetic data
pixi run audit-synthetic

# Or run directly with Python
python -m audit.run_audit --synthetic
```

## What This Audits

The benchmark trains models (Logistic Regression, Random Forest, Gradient Boosting, MLP) on three ED prediction tasks:

| Task | Outcome | Description |
|------|---------|-------------|
| Hospitalization | `outcome_hospitalization` | Will the patient be admitted? |
| Critical | `outcome_critical` | ICU transfer or mortality within 12 hours? |
| ED Reattendance | `outcome_ed_revisit_3d` | Will the patient return to ED within 72 hours? |

The audit evaluates these models across demographic subgroups: **race/ethnicity**, **gender**, **age**, and **insurance type**.

## Directory Structure

```
audit/
├── config.py                    # Central configuration
├── run_audit.py                 # Master pipeline script
├── bias/
│   ├── subgroup_performance.py  # AUROC, sensitivity, specificity per subgroup
│   ├── fairness_metrics.py      # Demographic parity, equalized odds (Fairlearn)
│   ├── shap_subgroup.py         # SHAP feature importance per subgroup
│   └── bias_report.py           # Markdown bias report
├── explainability/
│   ├── shap_explanations.py     # Per-patient SHAP explanations (JSON)
│   ├── explanation_quality.py   # Feature agreement, stability metrics
│   └── clinician_review.py      # HTML report for clinician focus groups
├── validity/
│   ├── outcome_correlation.py   # Prediction vs ESI acuity, LOS, mortality
│   ├── subgroup_validity.py     # Stratified correlations + Fisher z-test
│   └── validity_report.py       # Markdown validity report
├── models/
│   ├── train_models.py          # Train LR, RF, GBM, MLP per task
│   ├── demographic_variants.py  # Aware vs agnostic model comparison
│   └── utils.py                 # Data loading, model I/O, thresholds
├── visualization/
│   ├── performance_plots.py     # Grouped bar charts per subgroup
│   ├── fairness_plots.py        # Radar plots of fairness metrics
│   ├── shap_comparison.py       # Side-by-side SHAP plots
│   └── validity_plots.py        # Scatter plots with regression lines
├── reports/
│   └── generate_full_report.py  # Comprehensive markdown report
└── tests/
    ├── test_with_synthetic.py   # Synthetic data generator
    └── test_pipeline.py         # End-to-end integration test
```

## Usage

### Full audit on real MIMIC data

```bash
# Set data paths (or place train.csv/test.csv in data/)
export AUDIT_TRAIN_CSV=/path/to/train.csv
export AUDIT_TEST_CSV=/path/to/test.csv

python -m audit.run_audit
```

### Selective audit

```bash
# Single task, two models
python -m audit.run_audit --tasks hospitalization --models lr rf

# Skip slow SHAP analysis
python -m audit.run_audit --skip-shap

# Skip demographic variant training
python -m audit.run_audit --skip-variants
```

### Integration test

```bash
python -m audit.tests.test_pipeline
```

## Outputs

All outputs are saved under `audit/output/`:

| Directory | Contents |
|-----------|----------|
| `trained_models/` | Pickled sklearn model files |
| `predictions/` | Test-set predictions CSVs (with demographics) |
| `bias/` | Subgroup performance, fairness metrics, SHAP analysis CSVs |
| `explainability/` | Individual patient explanations (JSON), clinician review (HTML) |
| `validity/` | Outcome correlations, subgroup validity CSVs |
| `plots/` | All visualizations (PNG 300dpi + PDF) |
| `reports/` | Final comprehensive markdown report |

## Key Metrics

### Bias Detection
- **Subgroup performance**: AUROC, sensitivity, specificity, PPV, NPV, FPR, FNR with bootstrap 95% CIs
- **Fairness**: Demographic parity difference, equalized odds difference, equal opportunity difference
- **SHAP disparity**: Spearman rank correlation of feature importance across subgroups

### Explainability
- **Individual explanations**: Top-5 features per patient with SHAP values
- **Quality metrics**: Feature agreement (Jaccard), SHAP stability (variance), consistency (Spearman)
- **Clinician review**: Formatted HTML with response fields for focus groups

### Clinical Validity
- **Prediction-severity correlation**: Spearman r against ESI acuity, ED LOS, mortality
- **Cross-group comparison**: Fisher z-transformation to test equal validity across demographics

## Synthetic Data

The synthetic data generator (`test_with_synthetic.py`) creates 5,000 patients matching the MIMIC-IV-ED schema with:
- Realistic demographic distributions
- Physiologically plausible vital signs
- **Injected bias**: Hospitalization rate is 5% lower for Black patients at equivalent acuity

This known disparity validates that the bias detection pipeline correctly identifies disparities.

## Dependencies

- Python 3.9+
- scikit-learn, shap, fairlearn, matplotlib, seaborn, pandas, numpy, scipy, tabulate

See `requirements.txt` for exact versions.
