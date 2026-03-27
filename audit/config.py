"""
Central configuration for the MIMIC-IV-ED fairness audit framework.

All paths, demographic definitions, task configurations, model lists,
and analysis thresholds are defined here.
"""

import os
from pathlib import Path

# ── Project root paths ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "Benchmark_scripts"

# ── Data paths (override via environment variables if needed) ────────────────
DATA_DIR = Path(os.environ.get("AUDIT_DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_CSV = Path(os.environ.get("AUDIT_TRAIN_CSV", DATA_DIR / "train.csv"))
TEST_CSV = Path(os.environ.get("AUDIT_TEST_CSV", DATA_DIR / "test.csv"))

# ── Output directories ──────────────────────────────────────────────────────
OUTPUT_DIR = AUDIT_ROOT / "output"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "trained_models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
BIAS_OUTPUT_DIR = OUTPUT_DIR / "bias"
EXPLAINABILITY_OUTPUT_DIR = OUTPUT_DIR / "explainability"
VALIDITY_OUTPUT_DIR = OUTPUT_DIR / "validity"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ── Demographic columns and groupings ───────────────────────────────────────
DEMOGRAPHIC_COLS = ["ethnicity", "gender", "anchor_age", "insurance"]

# Age bins used to discretize continuous age into subgroups
AGE_BINS = [18, 35, 50, 65, 80, 200]
AGE_LABELS = ["18-35", "36-50", "51-65", "66-80", "80+"]

# Ethnicity groups to consolidate rare categories
ETHNICITY_MAP = {
    "WHITE": "White",
    "BLACK/AFRICAN AMERICAN": "Black",
    "HISPANIC/LATINO": "Hispanic",
    "HISPANIC OR LATINO": "Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - CUBAN": "Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic",
    "HISPANIC/LATINO - COLOMBIAN": "Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Hispanic",
    "ASIAN": "Asian",
    "ASIAN - CHINESE": "Asian",
    "ASIAN - SOUTH EAST ASIAN": "Asian",
    "ASIAN - ASIAN INDIAN": "Asian",
    "ASIAN - KOREAN": "Asian",
    "AMERICAN INDIAN/ALASKA NATIVE": "Other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Other",
    "OTHER": "Other",
    "UNABLE TO OBTAIN": "Unknown",
    "PATIENT DECLINED TO ANSWER": "Unknown",
    "UNKNOWN": "Unknown",
}

# Insurance consolidation
INSURANCE_MAP = {
    "Medicare": "Medicare",
    "Medicaid": "Medicaid",
    "Other": "Private/Other",
}

# ── Prediction tasks ────────────────────────────────────────────────────────
TASKS = ["hospitalization", "critical", "ed_reattendance"]

TASK_OUTCOME_COL = {
    "hospitalization": "outcome_hospitalization",
    "critical": "outcome_critical",
    "ed_reattendance": "outcome_ed_revisit_3d",
}

TASK_DISPLAY_NAME = {
    "hospitalization": "Hospitalization Prediction",
    "critical": "Critical Outcome (ICU/Mortality 12h)",
    "ed_reattendance": "72-Hour ED Reattendance",
}

# Features per task (matching the original benchmark notebooks exactly)
_COMMON_FEATURES_TASKS_1_2 = [
    "age", "gender",
    "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "triage_temperature", "triage_heartrate", "triage_resprate",
    "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
    "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
    "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
    "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
    "chiefcom_dizziness",
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
    "cci_Cancer2", "cci_HIV",
    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
]

_TASK_3_FEATURES = [
    "age", "gender",
    "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "triage_pain", "triage_acuity",
    "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
    "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
    "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
    "chiefcom_dizziness",
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
    "cci_Cancer2", "cci_HIV",
    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
    "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last",
    "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon",
]

TASK_FEATURES = {
    "hospitalization": _COMMON_FEATURES_TASKS_1_2,
    "critical": _COMMON_FEATURES_TASKS_1_2,
    "ed_reattendance": _TASK_3_FEATURES,
}

# Demographic-sensitive features to drop for demographic-agnostic models
DEMOGRAPHIC_FEATURES = ["ethnicity", "insurance"]

# ── Models ───────────────────────────────────────────────────────────────────
MODELS = ["lr", "rf", "gbm", "mlp"]

MODEL_DISPLAY_NAME = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gbm": "Gradient Boosting",
    "mlp": "MLP Neural Network",
}

# ── Analysis parameters ─────────────────────────────────────────────────────
RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 1000
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD = 0.1
SENSITIVITY_DROP_THRESHOLD = 0.05   # flag if >5% below population
RANK_SHIFT_THRESHOLD = 5            # flag features shifting >5 ranks
VALIDITY_CORRELATION_THRESHOLD = 0.3  # flag if correlation drops below

# ── Clinician review ────────────────────────────────────────────────────────
CLINICIAN_SAMPLE_PER_SUBGROUP = 5


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        OUTPUT_DIR, MODELS_OUTPUT_DIR, PREDICTIONS_DIR,
        BIAS_OUTPUT_DIR, EXPLAINABILITY_OUTPUT_DIR, VALIDITY_OUTPUT_DIR,
        PLOTS_DIR, REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_age_group(age):
    """Map a numeric age to its age-bin label."""
    import pandas as pd
    return pd.cut(
        [age], bins=AGE_BINS, labels=AGE_LABELS, right=True
    )[0]


def map_ethnicity(eth):
    """Consolidate raw MIMIC ethnicity string into a simplified group."""
    if isinstance(eth, str):
        return ETHNICITY_MAP.get(eth.strip().upper(), "Other")
    return "Unknown"


def map_insurance(ins):
    """Consolidate insurance type."""
    if isinstance(ins, str):
        return INSURANCE_MAP.get(ins, "Private/Other")
    return "Unknown"
