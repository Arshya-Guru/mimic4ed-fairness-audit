"""
Correlate model predicted probabilities against clinical severity indicators.

For each task, correlate predictions against:
- ESI acuity score (triage_acuity)
- ED length of stay (ed_los or computed from outtime - intime)
- In-hospital mortality (if available)

Computes Spearman correlation + p-value.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from audit.config import (
    TASKS, MODELS, VALIDITY_OUTPUT_DIR, ensure_output_dirs,
)
from audit.models.utils import load_predictions


def compute_outcome_correlations(tasks=None, models=None, variant="default"):
    """Compute Spearman correlations between predictions and severity indicators.

    Args:
        tasks: List of tasks. Defaults to all.
        models: List of model names. Defaults to all.
        variant: Model variant.

    Returns:
        DataFrame with correlation results.
    """
    ensure_output_dirs()
    tasks = tasks or TASKS
    models = models or MODELS
    rows = []

    severity_cols = {
        "triage_acuity": "ESI Acuity",
        "ed_los": "ED Length of Stay",
        "outcome_inhospital_mortality": "In-Hospital Mortality",
    }

    for task in tasks:
        for model_name in models:
            try:
                preds = load_predictions(task, model_name, variant)
            except FileNotFoundError:
                continue

            y_prob = preds["y_prob"].values

            for sev_col, sev_label in severity_cols.items():
                if sev_col not in preds.columns:
                    continue

                sev_vals = pd.to_numeric(preds[sev_col], errors="coerce").values
                mask = ~(np.isnan(y_prob) | np.isnan(sev_vals))

                if mask.sum() < 20:
                    continue

                corr, pval = spearmanr(y_prob[mask], sev_vals[mask])

                rows.append({
                    "task": task,
                    "model": model_name,
                    "severity_indicator": sev_label,
                    "severity_column": sev_col,
                    "spearman_r": corr,
                    "p_value": pval,
                    "n_samples": int(mask.sum()),
                })

    df = pd.DataFrame(rows)
    out_path = VALIDITY_OUTPUT_DIR / "outcome_correlations.csv"
    VALIDITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Outcome correlations saved to {out_path}")
    return df


if __name__ == "__main__":
    compute_outcome_correlations()
