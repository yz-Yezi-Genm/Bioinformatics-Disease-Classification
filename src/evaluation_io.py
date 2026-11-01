"""Read and validate the inputs used by evaluation figures."""

import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import align_predictions, binary_values, compute_metrics, load_reference, validate_ids

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "specificity", "balanced_accuracy"]
COUNT_KEYS = ["n_samples", "n_normal", "n_tumor", "tn", "fp", "fn", "tp"]
MODELS = ["Random Forest", "SVM"]
COLORS = {"Random Forest": "#167D8D", "SVM": "#D57442"}
INK = "#172C3D"
MUTED = "#62727D"
GRID = "#E4E9ED"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 13,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#C6D0D7", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "path",
})

def load_inputs(results_dir, test_path):
    """Check saved metrics against sample predictions and reference labels."""
    metrics = pd.read_csv(results_dir / "metrics.csv")
    if "model" not in metrics or metrics.model.duplicated().any() or set(metrics.model) != set(MODELS):
        raise ValueError("metrics.csv must contain one row per model: Random Forest and SVM")
    if not set(KEYS + COUNT_KEYS).issubset(metrics.columns):
        raise ValueError("metrics.csv is missing required metrics or counts")
    metrics = metrics.set_index("model").loc[MODELS]
    pred = pd.read_csv(results_dir / "evaluation/aligned_predictions.csv", dtype={"sample_id": str})
    required = {"sample_id", "y_true", "rf_y_pred", "rf_score", "svm_y_pred", "svm_score"}
    if not required.issubset(pred):
        raise ValueError("aligned_predictions.csv is missing required columns")
    validate_ids(pred, "aligned predictions")
    for name in ["y_true", "rf_y_pred", "svm_y_pred"]:
        pred[name] = binary_values(pred[name], name)
    reference = load_reference(test_path)
    pred = align_predictions(reference, pred, "aligned predictions")
    for model, prefix in zip(MODELS, ["rf", "svm"]):
        score = pd.to_numeric(pred[f"{prefix}_score"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise ValueError(f"{model}: invalid scores")
        if prefix == "rf" and ((score < 0) | (score > 1)).any():
            raise ValueError("Random Forest: probabilities outside [0, 1]")
        pred[f"{prefix}_score"] = score
        normalized = pred[["sample_id", "y_true", f"{prefix}_y_pred", f"{prefix}_score"]].rename(
            columns={f"{prefix}_y_pred": "y_pred", f"{prefix}_score": "score"})
        recomputed = compute_metrics(normalized, model)
        for key in KEYS + COUNT_KEYS:
            reported = float(metrics.loc[model, key])
            if not np.isfinite(reported) or not np.isclose(reported, recomputed[key], rtol=0, atol=1e-12):
                raise ValueError(f"{model}: {key} disagrees with predictions")
    return metrics, pred

def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
