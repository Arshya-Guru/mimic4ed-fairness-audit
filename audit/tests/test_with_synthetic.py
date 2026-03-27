"""
Generate synthetic data mimicking the MIMIC-IV-ED master_dataset.csv schema
for testing the full audit pipeline without credentialed MIMIC access.

Includes realistic distributions for demographics, vitals, and lab values.
Injects known bias: hospitalization rate is 5% lower for one racial group
at equivalent acuity, enabling validation that the bias detection pipeline works.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from audit.config import (
    TASK_FEATURES, AGE_BINS, AGE_LABELS,
    AUDIT_ROOT, ensure_output_dirs,
)


# Realistic distributions based on published MIMIC-IV-ED statistics
ETHNICITY_DIST = {
    "WHITE": 0.55,
    "BLACK/AFRICAN AMERICAN": 0.20,
    "HISPANIC/LATINO": 0.10,
    "ASIAN": 0.05,
    "OTHER": 0.05,
    "UNKNOWN": 0.05,
}

INSURANCE_DIST = {
    "Medicare": 0.40,
    "Medicaid": 0.20,
    "Other": 0.40,
}

GENDER_DIST = {"F": 0.52, "M": 0.48}

# Vital sign distributions (mean, std)
VITAL_PARAMS = {
    "triage_temperature": (36.8, 0.6),
    "triage_heartrate": (85, 18),
    "triage_resprate": (18, 4),
    "triage_o2sat": (97, 2.5),
    "triage_sbp": (135, 22),
    "triage_dbp": (78, 14),
    "triage_pain": (4, 3),
    "triage_acuity": (3, 0.9),
    "ed_temperature_last": (36.8, 0.5),
    "ed_heartrate_last": (82, 16),
    "ed_resprate_last": (17, 3.5),
    "ed_o2sat_last": (97.5, 2),
    "ed_sbp_last": (132, 20),
    "ed_dbp_last": (76, 13),
}


def generate_synthetic_dataset(n_samples=5000, seed=42):
    """Generate a synthetic dataset matching MIMIC-IV-ED schema.

    Creates realistic patient data with injected bias:
    - Black patients have 5% lower hospitalization rate at equivalent acuity
    - This known disparity validates the bias detection pipeline

    Args:
        n_samples: Number of synthetic patients to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df) DataFrames.
    """
    rng = np.random.RandomState(seed)

    # ── Demographics ─────────────────────────────────────────────────────
    ethnicities = list(ETHNICITY_DIST.keys())
    eth_probs = list(ETHNICITY_DIST.values())
    ethnicity = rng.choice(ethnicities, size=n_samples, p=eth_probs)

    genders = list(GENDER_DIST.keys())
    gen_probs = list(GENDER_DIST.values())
    gender = rng.choice(genders, size=n_samples, p=gen_probs)

    insurances = list(INSURANCE_DIST.keys())
    ins_probs = list(INSURANCE_DIST.values())
    insurance = rng.choice(insurances, size=n_samples, p=ins_probs)

    # Age: realistic ED distribution (skewed older)
    age = rng.normal(55, 20, n_samples).clip(18, 100).astype(int)
    anchor_age = age  # simplified for synthetic
    anchor_year = np.full(n_samples, 2020)

    # ── Vital signs ──────────────────────────────────────────────────────
    data = {
        "subject_id": np.arange(10000, 10000 + n_samples),
        "stay_id": np.arange(20000, 20000 + n_samples),
        "hadm_id": np.arange(30000, 30000 + n_samples),
        "gender": gender,
        "ethnicity": ethnicity,
        "insurance": insurance,
        "age": age,
        "anchor_age": anchor_age,
        "anchor_year": anchor_year,
        "in_year": np.full(n_samples, 2020),
    }

    # Generate vitals
    for vital, (mean, std) in VITAL_PARAMS.items():
        vals = rng.normal(mean, std, n_samples)
        if vital == "triage_acuity":
            vals = np.round(vals).clip(1, 5).astype(int)
        elif vital == "triage_pain":
            vals = np.round(vals).clip(0, 10).astype(int)
        elif "o2sat" in vital:
            vals = vals.clip(70, 100)
        elif "temperature" in vital:
            vals = vals.clip(34, 42)
        elif "heartrate" in vital:
            vals = vals.clip(30, 200)
        elif "resprate" in vital:
            vals = vals.clip(6, 50)
        elif "sbp" in vital or "dbp" in vital:
            vals = vals.clip(40, 250)
        data[vital] = vals

    # ── Historical visit counts ──────────────────────────────────────────
    for timerange in ["30d", "90d", "365d"]:
        data[f"n_ed_{timerange}"] = rng.poisson(0.5, n_samples)
        data[f"n_hosp_{timerange}"] = rng.poisson(0.3, n_samples)
        data[f"n_icu_{timerange}"] = rng.poisson(0.1, n_samples)

    # ── Chief complaints (binary) ────────────────────────────────────────
    complaints = [
        "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
        "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
        "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
        "chiefcom_dizziness",
    ]
    for cc in complaints:
        data[cc] = rng.binomial(1, 0.1, n_samples)

    # ── Comorbidity flags ────────────────────────────────────────────────
    cci_flags = [
        "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
        "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
        "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
        "cci_Cancer2", "cci_HIV",
    ]
    for flag in cci_flags:
        prob = 0.05 + rng.uniform(0, 0.1)
        data[flag] = rng.binomial(1, prob, n_samples)

    eci_flags = [
        "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
        "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
        "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
        "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
    ]
    for flag in eci_flags:
        prob = 0.03 + rng.uniform(0, 0.08)
        data[flag] = rng.binomial(1, prob, n_samples)

    # ── ED-specific features (Task 3) ────────────────────────────────────
    data["ed_los"] = pd.to_timedelta(
        rng.exponential(4, n_samples) * 3600, unit="s"
    ).astype(str)  # Store as timedelta string like the real data
    data["n_med"] = rng.poisson(2, n_samples)
    data["n_medrecon"] = rng.poisson(3, n_samples)

    # ── Generate outcomes with injected bias ─────────────────────────────
    df = pd.DataFrame(data)

    # Hospitalization: base rate depends on acuity, with racial bias injected
    acuity = df["triage_acuity"].values
    base_hosp_prob = 0.7 - 0.12 * acuity  # higher acuity (lower number) = more likely
    base_hosp_prob = np.clip(base_hosp_prob, 0.05, 0.85)

    # INJECT BIAS: reduce hospitalization probability for Black patients by 5%
    black_mask = df["ethnicity"] == "BLACK/AFRICAN AMERICAN"
    base_hosp_prob[black_mask] -= 0.05

    df["outcome_hospitalization"] = rng.binomial(1, base_hosp_prob).astype(bool)

    # Critical outcome: based on vitals severity
    severity_score = (
        (df["triage_heartrate"] > 100).astype(float) * 0.15 +
        (df["triage_o2sat"] < 92).astype(float) * 0.2 +
        (df["triage_sbp"] < 90).astype(float) * 0.15 +
        (df["triage_acuity"] <= 2).astype(float) * 0.15 +
        (df["age"] > 70).astype(float) * 0.05
    )
    critical_prob = np.clip(severity_score + 0.02, 0.01, 0.5)
    df["outcome_critical"] = rng.binomial(1, critical_prob).astype(bool)

    # ED revisit: slightly higher for younger patients, those with more prior visits
    revisit_prob = np.clip(
        0.08 + 0.02 * df["n_ed_30d"] - 0.001 * df["age"] + 0.01 * (df["triage_acuity"] - 3),
        0.02, 0.3
    )
    df["outcome_ed_revisit_3d"] = rng.binomial(1, revisit_prob).astype(bool)

    # In-hospital mortality
    mort_prob = np.clip(severity_score * 0.3, 0.001, 0.15)
    df["outcome_inhospital_mortality"] = rng.binomial(1, mort_prob).astype(bool)

    # ── Times (synthetic) ────────────────────────────────────────────────
    base_time = pd.Timestamp("2020-01-01")
    df["intime"] = [base_time + pd.Timedelta(hours=rng.randint(0, 8760)) for _ in range(n_samples)]
    df["outtime"] = df["intime"] + pd.to_timedelta(rng.exponential(4, n_samples), unit="h")

    # ── Train/test split (80/20) ─────────────────────────────────────────
    split_idx = int(n_samples * 0.8)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = shuffled.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def save_synthetic_data(output_dir=None, n_samples=5000, seed=42):
    """Generate and save synthetic train/test CSVs.

    Args:
        output_dir: Directory to save CSVs. Defaults to audit/output/synthetic/.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        Tuple of (train_path, test_path).
    """
    if output_dir is None:
        output_dir = AUDIT_ROOT / "output" / "synthetic"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = generate_synthetic_dataset(n_samples, seed)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Synthetic data saved: {train_path} ({len(train_df)} rows), "
          f"{test_path} ({len(test_df)} rows)")
    print(f"  Hospitalization rate (Black): "
          f"{train_df[train_df['ethnicity'] == 'BLACK/AFRICAN AMERICAN']['outcome_hospitalization'].mean():.3f}")
    print(f"  Hospitalization rate (White): "
          f"{train_df[train_df['ethnicity'] == 'WHITE']['outcome_hospitalization'].mean():.3f}")

    return train_path, test_path


if __name__ == "__main__":
    save_synthetic_data()
