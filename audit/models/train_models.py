"""
Train the Xie et al. benchmark models (LR, RF, GBM, MLP) for each prediction task.

Wraps the original notebook logic into callable functions. Saves trained model
objects (.pkl) and test-set predictions (.csv) that downstream audit modules consume.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from audit.config import (
    TASKS, MODELS, RANDOM_SEED, TRAIN_CSV, TEST_CSV,
    ensure_output_dirs,
)
from audit.models.utils import (
    load_data, save_model, save_predictions, add_demographic_groups,
)


def _build_model(model_name):
    """Instantiate a scikit-learn model matching the benchmark configuration.

    Args:
        model_name: One of 'lr', 'rf', 'gbm', 'mlp'.

    Returns:
        Unfitted sklearn estimator.
    """
    if model_name == "lr":
        return LogisticRegression(random_state=RANDOM_SEED, max_iter=5000)
    elif model_name == "rf":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)
    elif model_name == "gbm":
        return GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=100)
    elif model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=200,
            batch_size=200,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_single(task, model_name, train_csv=None, test_csv=None,
                 feature_override=None, variant="default"):
    """Train a single model for a single task and save artifacts.

    Args:
        task: Task name ('hospitalization', 'critical', 'ed_reattendance').
        model_name: Model identifier ('lr', 'rf', 'gbm', 'mlp').
        train_csv: Path to training data. Defaults to config.TRAIN_CSV.
        test_csv: Path to test data. Defaults to config.TEST_CSV.
        feature_override: Optional list of feature columns to use instead of defaults.
        variant: Label for this model variant (e.g., 'default', 'demographic_agnostic').

    Returns:
        Tuple of (trained_model, y_test, y_prob).
    """
    train_csv = train_csv or TRAIN_CSV
    test_csv = test_csv or TEST_CSV

    X_train, y_train, X_test, y_test, df_test = load_data(train_csv, test_csv, task)

    # Apply feature override (for demographic-agnostic variants)
    if feature_override is not None:
        available = [f for f in feature_override if f in X_train.columns]
        X_train = X_train[available]
        X_test = X_test[available]

    model = _build_model(model_name)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    save_model(model, task, model_name, variant=variant)

    # Include demographic columns in predictions for downstream analysis
    demo_cols = {}
    for col in ["ethnicity", "gender", "anchor_age", "age", "insurance",
                 "triage_acuity", "ed_los", "outcome_inhospital_mortality"]:
        if col in df_test.columns:
            demo_cols[col] = df_test[col].values

    save_predictions(y_test, y_prob, task, model_name, variant=variant,
                     extra_cols=demo_cols)

    return model, y_test, y_prob


def train_all(tasks=None, models=None, train_csv=None, test_csv=None):
    """Train all model×task combinations.

    Args:
        tasks: List of tasks to train. Defaults to all tasks.
        models: List of model names. Defaults to all models.
        train_csv: Path to training data.
        test_csv: Path to test data.

    Returns:
        Dict mapping (task, model_name) to (model, y_test, y_prob).
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    results = {}

    for task in tasks:
        for model_name in models:
            print(f"  Training {model_name} for {task}...")
            result = train_single(task, model_name, train_csv, test_csv)
            results[(task, model_name)] = result
            print(f"    Done. AUROC will be computed in bias audit.")

    return results


if __name__ == "__main__":
    ensure_output_dirs()
    train_all()
