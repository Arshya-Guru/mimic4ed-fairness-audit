"""
Shared utilities for model loading, prediction, data preparation,
and threshold calibration.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from audit.config import (
    TASK_FEATURES, TASK_OUTCOME_COL, MODELS_OUTPUT_DIR,
    PREDICTIONS_DIR, DEMOGRAPHIC_COLS, AGE_BINS, AGE_LABELS,
    map_ethnicity, map_insurance,
)


def load_data(train_path, test_path, task):
    """Load train/test CSVs and return X_train, y_train, X_test, y_test, df_test_full.

    Applies the same filtering as the original benchmark notebooks:
    - Task 3 (ed_reattendance): filter to non-hospitalized patients only.
    - Gender is label-encoded to numeric.
    - ed_los is converted from timedelta string to minutes (Task 3).

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.
        task: One of 'hospitalization', 'critical', 'ed_reattendance'.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, df_test_full).
        df_test_full contains all columns for demographic analysis.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Task 3: filter to non-hospitalized patients
    if task == "ed_reattendance":
        df_train = df_train[df_train["outcome_hospitalization"] == False].reset_index(drop=True)
        df_test = df_test[df_test["outcome_hospitalization"] == False].reset_index(drop=True)

    features = TASK_FEATURES[task]
    outcome_col = TASK_OUTCOME_COL[task]

    y_train = df_train[outcome_col].astype(int).values
    y_test = df_test[outcome_col].astype(int).values

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()

    # Encode gender
    le = LabelEncoder()
    X_train["gender"] = le.fit_transform(X_train["gender"].astype(str))
    X_test["gender"] = le.transform(X_test["gender"].astype(str))

    # Convert ed_los from timedelta string to minutes for task 3
    if task == "ed_reattendance" and "ed_los" in features:
        X_train["ed_los"] = pd.to_timedelta(X_train["ed_los"]).dt.total_seconds() / 60
        X_test["ed_los"] = pd.to_timedelta(X_test["ed_los"]).dt.total_seconds() / 60
        X_train["ed_los"] = X_train["ed_los"].fillna(0)
        X_test["ed_los"] = X_test["ed_los"].fillna(0)

    # Fill remaining NaNs with median
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X_train, y_train, X_test, y_test, df_test


def add_demographic_groups(df):
    """Add consolidated demographic group columns to a dataframe.

    Adds: ethnicity_group, age_group, insurance_group.
    These are derived from the raw MIMIC columns.

    Args:
        df: DataFrame containing raw demographic columns.

    Returns:
        DataFrame with added group columns.
    """
    df = df.copy()
    if "ethnicity" in df.columns:
        df["ethnicity_group"] = df["ethnicity"].apply(map_ethnicity)
    if "anchor_age" in df.columns:
        df["age_group"] = pd.cut(
            df["anchor_age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
        )
    elif "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
        )
    if "insurance" in df.columns:
        df["insurance_group"] = df["insurance"].apply(map_insurance)
    if "gender" in df.columns:
        df["gender_group"] = df["gender"].astype(str)
    return df


def save_model(model, task, model_name, variant="default"):
    """Save a trained model to disk as a pickle file.

    Args:
        model: Trained model object.
        task: Task name string.
        model_name: Model identifier (e.g., 'lr', 'rf').
        variant: Model variant ('default', 'demographic_aware', 'demographic_agnostic').
    """
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_OUTPUT_DIR / f"{task}_{model_name}_{variant}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(task, model_name, variant="default"):
    """Load a trained model from disk.

    Args:
        task: Task name string.
        model_name: Model identifier.
        variant: Model variant.

    Returns:
        Trained model object.
    """
    path = MODELS_OUTPUT_DIR / f"{task}_{model_name}_{variant}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def save_predictions(y_true, y_prob, task, model_name, variant="default", extra_cols=None):
    """Save test-set predictions to CSV.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        task: Task name.
        model_name: Model identifier.
        variant: Model variant.
        extra_cols: Optional dict of additional columns to include.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    path = PREDICTIONS_DIR / f"{task}_{model_name}_{variant}_predictions.csv"
    df.to_csv(path, index=False)


def load_predictions(task, model_name, variant="default"):
    """Load saved predictions from CSV.

    Args:
        task: Task name.
        model_name: Model identifier.
        variant: Model variant.

    Returns:
        DataFrame with y_true and y_prob columns.
    """
    path = PREDICTIONS_DIR / f"{task}_{model_name}_{variant}_predictions.csv"
    return pd.read_csv(path)


def get_optimal_threshold(y_true, y_prob):
    """Find the threshold that maximizes Youden's J statistic (sensitivity + specificity - 1).

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold float.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]
