MIMIC-IV-ED Benchmark + Fairness Audit
=========================================

Python workflow for generating benchmark datasets and machine learning models from the MIMIC-IV-ED database, extended with a **fairness, explainability, and clinical validity audit framework**. See the original [Scientific Data](https://www.nature.com/articles/s41597-022-01782-9) article for the benchmark details.

## Table of contents
* [General Info](#general-info)
* [Structure](#structure)
* [Requirements and Setup](#requirements-and-setup)
* [Benchmark Workflow](#benchmark-workflow)
    1. [Benchmark Data Generation](#1-benchmark-data-generation)
    2. [Cohort Filtering](#2-cohort-filtering-and-data-processing)
    3. [Prediction Task and Model Selection](#3-prediction-task-selection-and-model-evaluation)
* [Fairness Audit](#fairness-audit)
    * [Quick Start (Synthetic Data)](#quick-start-synthetic-data)
    * [Running on Real MIMIC Data](#running-on-real-mimic-data)
    * [Selective Audit Options](#selective-audit-options)
    * [Audit Pipeline Stages](#audit-pipeline-stages)
    * [Audit Outputs](#audit-outputs)
    * [Key Metrics](#key-metrics)
* [Citation](#citation)

## General Info

Clinical decisions in the emergency department play an important role in optimizing urgent patient care and scarce resources. And unsurprisingly, machine learning based clinical prediction models have been widely adopted in the field of emergency medicine.

In parallel to the rise of clinical prediction models, there has also been a rapid increase in adoption of Electronic Health Records (EHR) for patient data. The Medical Information Mart for Intensive Care ([MIMIC-IV](https://physionet.org/content/mimiciv/1.0/)) and [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/) are examples of EHR databases that contain a vast amount of patient information.

There is therefore a need for publicly available benchmark datasets and models that allow researchers to produce comparable and reproducible results.

For the previous iteration of the MIMIC database (MIMIC-III), several benchmark pipelines have been published in [2019](https://github.com/YerevaNN/mimic3-benchmarks) and [2020](https://github.com/MLforHealth/MIMIC_Extract).

Here, we present a workflow that generates a benchmark dataset from the MIMIC-IV-ED database, constructs benchmark models for three ED-based prediction tasks, and **audits those models for fairness, explainability, and clinical validity across demographic subgroups**.

## Structure

```
├── Benchmark_scripts/            # Original benchmark pipeline
│   ├── extract_master_dataset.py # Step 1: Build master dataset from MIMIC tables
│   ├── data_general_processing.py# Step 2: Cohort filtering, outlier handling, train/test split
│   ├── Task_1_*.ipynb            # Step 3: Hospitalization prediction models
│   ├── Task_2_*.ipynb            # Step 3: Critical outcome prediction models
│   ├── Task_3_*.ipynb            # Step 3: 72h ED revisit prediction models
│   ├── Task_4_*.ipynb            # Appendix: Critical outcome (disposition variant)
│   ├── medcodes/                 # ICD code mappings and drug classifications
│   └── helpers.py, utils.py      # Shared utilities
├── audit/                        # Fairness audit framework
│   ├── config.py                 # Central configuration (paths, thresholds, features)
│   ├── run_audit.py              # Master pipeline script (6-stage audit)
│   ├── bias/                     # Subgroup performance, fairness metrics, SHAP disparity
│   ├── explainability/           # SHAP explanations, quality metrics, clinician review
│   ├── validity/                 # Outcome correlations, subgroup validity, Fisher z-tests
│   ├── models/                   # Model training (LR, RF, GBM, MLP) + demographic variants
│   ├── visualization/            # Performance, fairness, SHAP, and validity plots
│   ├── reports/                  # Comprehensive markdown report generator
│   └── tests/                    # Synthetic data generator + integration tests
├── data/                         # MIMIC data directory (not included)
├── pixi.toml                     # Pixi project config with task runners
└── requirements.txt              # Python dependencies
```

## Requirements and Setup

### MIMIC Data (for benchmark + real-data audit)

MIMIC-IV-ED and MIMIC-IV databases are not provided with this repository and are **required** for the benchmark workflow and real-data auditing. MIMIC-IV-ED can be downloaded from [https://physionet.org/content/mimic-iv-ed/1.0/](https://physionet.org/content/mimic-iv-ed/1.0/) and MIMIC-IV can be downloaded from [https://physionet.org/content/mimiciv/1.0/](https://physionet.org/content/mimiciv/1.0/).

***NOTE** Upon downloading and extracting the MIMIC databases from their compressed files, the directory `/mimic-iv-ed-1.0/ed` should be moved/copied to the directory containing MIMIC-IV data `/mimic-iv-1.0`.

### Python Dependencies

**Option A — Pixi (recommended):**

```bash
pixi install
```

**Option B — pip:**

```bash
pip install -r requirements.txt
pip install shap fairlearn scipy tabulate seaborn
```

The audit framework requires: Python 3.9+, scikit-learn, shap, fairlearn, matplotlib, seaborn, pandas, numpy, scipy, and tabulate.

## Benchmark Workflow

The following sub-sections describe the sequential modules within the MIMIC-IV-ED benchmark workflow.

Prior to these steps, this repository, MIMIC-IV-ED and MIMIC-IV should be downloaded and set up locally.

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py --mimic4_path {mimic4_path} --output_path {output_path}
~~~

**Arguments**:

- `mimic4_path` : Path to directory containing MIMIC-IV data. Refer to [Requirements and Setup](#requirements-and-setup) for details.
- `output_path` : Path to output directory.
- `icu_transfer_timerange` : Timerange in hours for ICU transfer outcome. Default set to 12.
- `next_ed_visit_timerange` : Timerange in days for next ED visit outcome. Default set to 3.

**Output**:

`master_dataset.csv` output to `output_path`

**Details**:

The input `edstays.csv` from the MIMIC-IV-ED database is taken to be the root table, with `subject_id` as the unique identifier for each patient and `stay_id` as the unique identifier for each ED visit. This root table is then merged with other tables from the main MIMIC-IV database to capture an informative array of clinical variables for each patient.

A total of **81** variables are included in `master_dataset.csv` (Refer to Table 3 for full variable list).


### 2. Cohort Filtering and Data Processing
~~~
python data_general_processing.py --master_dataset_path {master_dataset_path} --output_path {output_path}
~~~

**Arguments**:

- `master_dataset_path` : Path to directory containing "master_dataset.csv".
- `output_path` : Path to output directory.

**Output**:

`train.csv` and `test.csv` output to `output_path`

**Details**:

`master_dataset.csv` is first filtered to remove pediatric subjects (Age < 18).

Outlier values in vital sign and lab test variables are then detected using an identical method to [Wang et al.](https://github.com/MLforHealth/MIMIC_Extract), with outlier thresholds defined previously by [Harutyunyan et al.](https://github.com/YerevaNN/mimic3-benchmarks) Outliers are then imputed with the nearest valid value.

The data is then split into `train.csv` and `test.csv` and clinical scores for each patient are then added as additional variables.


### 3. Prediction Task Selection and Model Evaluation

Prediction modelling is currently handled by python notebooks (.ipynb files) that correspond to each of the 3 prediction tasks.

**Arguments**:

- `path` : Path to directory containing `train.csv` and `test.csv`

**Output**:

`result_*.csv` and `importances_*.csv` output to `path`.

`*` denotes the task specific wildcard string, i.e for the hospitalization prediction task, output files are `result_hospitalization_triage.csv` and `importances_hospitalization_triage.csv`.

**Details**:

For each ED prediction task, various models are implemented and compared. These include: Logistic Regression, MLP Neural Networks, Random Forests and several validated early warning scores. Each model's performance metrics are then compared (`result_*.csv`), in addition to an overall variable importance ranking using Random Forests (`importances_*.csv`).

## Fairness Audit

The audit framework evaluates the benchmark models across three dimensions — **bias/fairness**, **explainability**, and **clinical validity** — stratified by demographic subgroups (race/ethnicity, gender, age, insurance type).

It trains four model types (Logistic Regression, Random Forest, Gradient Boosting, MLP) on each of the three prediction tasks:

| Task | Outcome Column | Description |
|------|----------------|-------------|
| Hospitalization | `outcome_hospitalization` | Will the patient be admitted? |
| Critical | `outcome_critical` | ICU transfer or mortality within 12 hours? |
| ED Reattendance | `outcome_ed_revisit_3d` | Will the patient return to ED within 72 hours? |

### Quick Start (Synthetic Data)

No MIMIC credentials needed — run the full pipeline on synthetic data with an injected bias signal:

```bash
# Using pixi
pixi install
pixi run audit-synthetic

# Or directly with Python
python -m audit.run_audit --synthetic
```

The synthetic data generator creates 5,000 patients matching the MIMIC-IV-ED schema with realistic demographics and an **injected hospitalization disparity** (5% lower rate for Black patients at equivalent acuity) to validate that the bias detection pipeline correctly identifies disparities.

### Running on Real MIMIC Data

After running the benchmark workflow (steps 1-2) to produce `train.csv` and `test.csv`:

```bash
# Option A: Place files in data/
cp /path/to/train.csv data/
cp /path/to/test.csv data/
python -m audit.run_audit

# Option B: Set environment variables
export AUDIT_TRAIN_CSV=/path/to/train.csv
export AUDIT_TEST_CSV=/path/to/test.csv
python -m audit.run_audit
```

### Selective Audit Options

```bash
# Single task, subset of models
python -m audit.run_audit --tasks hospitalization --models lr rf

# Skip slow SHAP analysis
python -m audit.run_audit --skip-shap

# Skip demographic-aware vs agnostic model comparison
python -m audit.run_audit --skip-variants

# Fast demo: synthetic data, no SHAP, no variants
pixi run audit-fast
```

### Audit Pipeline Stages

The audit runs in six sequential stages:

1. **Data loading** — Load real data or generate synthetic data
2. **Model training** — Train LR, RF, GBM, MLP per task; optionally train demographic-aware vs agnostic variants
3. **Bias audit** — Subgroup performance metrics, Fairlearn fairness metrics, SHAP feature importance per subgroup
4. **Explainability** — Per-patient SHAP explanations, explanation quality metrics, clinician review HTML
5. **Validity analysis** — Prediction vs ESI acuity/LOS/mortality correlations, cross-group Fisher z-tests
6. **Visualization & reporting** — Plots (PNG + PDF) and a comprehensive markdown report

### Audit Outputs

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

### Key Metrics

**Bias Detection:**
- Subgroup performance: AUROC, sensitivity, specificity, PPV, NPV, FPR, FNR with bootstrap 95% CIs
- Fairness: demographic parity difference, equalized odds difference, equal opportunity difference
- SHAP disparity: Spearman rank correlation of feature importance across subgroups

**Explainability:**
- Individual explanations: top-5 features per patient with SHAP values
- Quality metrics: feature agreement (Jaccard), SHAP stability (variance), consistency (Spearman)
- Clinician review: formatted HTML with response fields for focus groups

**Clinical Validity:**
- Prediction-severity correlation: Spearman r against ESI acuity, ED LOS, mortality
- Cross-group comparison: Fisher z-transformation to test equal validity across demographics

### Integration Tests

```bash
# Run end-to-end integration test
pixi run test
# Or directly
python -m audit.tests.test_pipeline
```

## Citation

Xie F, Zhou J, Lee JW, Tan M, Li SQ, Rajnthern L, Chee ML, Chakraborty B, Wong AKI, Dagan A, Ong MEH, Gao F, Liu N. Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data 2022 Oct; 9: 658. <https://doi.org/10.1038/s41597-022-01782-9>
