"""Compute binary classification metrics from saved test predictions.

Run from the project root: python -m src.evaluate
The positive class is tumor (1); normal tissue is class 0.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/tcga_brca_20261029_v2"
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "roc_auc": "ROC-AUC",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced accuracy",
}
COLORS = {"Random Forest": "#24788C", "SVM": "#D87942"}
OUTPUT_FILES = [
    "metrics.csv",
    "evaluation/aligned_predictions.csv",
    "evaluation/misclassified_samples.csv",
    "evaluation/roc_points.csv",
    "evaluation/confusion_matrices.csv",
    "evaluation/audit.json",
    "figures/rf_confusion_matrix.png",
    "figures/svm_confusion_matrix.png",
    "figures/roc_comparison.png",
    "figures/model_metrics_comparison.png",
]


def validate_ids(frame: pd.DataFrame, source: str) -> None:
    if frame.empty:
        raise ValueError(f"{source}: empty table")
    if "sample_id" not in frame:
        raise ValueError(f"{source}: missing sample_id")
    ids = frame["sample_id"]
    if ids.isna().any() or not ids.map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"{source}: sample IDs must be non-null strings")
    if ids.str.strip().eq("").any() or not ids.eq(ids.str.strip()).all():
        raise ValueError(f"{source}: empty ID or whitespace around sample ID")
    if ids.duplicated().any():
        raise ValueError(f"{source}: duplicate sample IDs")


def binary_values(values: pd.Series, source: str) -> np.ndarray:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{source}: labels must be 0 or 1")
    return array.astype(np.int64)


def load_predictions(path: Path, model: str) -> pd.DataFrame:
    """Read RF or SVM predictions and standardize their column names."""
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"sample_id": str})
    schema = {
        "Random Forest": ("true_label", "predicted_label", "tumor_probability"),
        "SVM": ("y_true", "y_pred", "decision_score"),
    }
    true_col, pred_col, score_col = schema[model]
    required = ["sample_id", true_col, pred_col, score_col]
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    table = table[required].rename(
        columns={true_col: "y_true", pred_col: "y_pred", score_col: "score"}
    )
    validate_ids(table, str(path))
    for key in ("y_true", "y_pred"):
        table[key] = binary_values(table[key], f"{path}: {key}")
    score = pd.to_numeric(table["score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise ValueError(f"{path}: scores must be finite")
    if np.isin(score, [0, 1]).all():
        raise ValueError(f"{path}: ROC requires continuous scores, not class labels")
    if model == "Random Forest" and ((score < 0) | (score > 1)).any():
        raise ValueError(f"{path}: tumor probabilities must be within [0, 1]")
    table["score"] = score
    return table


def load_reference(test_path: Path) -> pd.DataFrame:
    with np.load(test_path, allow_pickle=False) as data:
        ids, labels = data["sample_id"], data["y"]
    if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(labels):
        raise ValueError("Test NPZ: sample_id and y must be matching 1-D arrays")
    reference = pd.DataFrame({"sample_id": ids, "y_true": labels})
    validate_ids(reference, "test NPZ")
    reference["y_true"] = binary_values(reference["y_true"], "test NPZ")
    if set(reference["y_true"]) != {0, 1}:
        raise ValueError("Test NPZ: both classes are required for ROC-AUC")
    return reference


def align_predictions(reference: pd.DataFrame, table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Align predictions to test sample IDs and check their true labels."""
    validate_ids(reference, "reference")
    validate_ids(table, model)
    expected, actual = set(reference["sample_id"]), set(table["sample_id"])
    if actual != expected:
        raise ValueError(
            f"{model}: sample IDs do not match test set "
            f"(missing={len(expected - actual)}, extra={len(actual - expected)})"
        )
    aligned = table.set_index("sample_id").loc[reference["sample_id"]].reset_index()
    if not np.array_equal(aligned["y_true"], reference["y_true"]):
        raise ValueError(f"{model}: true labels disagree with test NPZ")
    return aligned


def validate_manifest(
    manifest_path: Path, reference: pd.DataFrame, train_path: Path
) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"sample_id": str, "patient_id": str})
    required = {"sample_id", "patient_id", "label", "split"}
    if not required.issubset(manifest):
        raise ValueError("Manifest: missing required columns")
    validate_ids(manifest, "manifest")
    if manifest["patient_id"].isna().any() or manifest["patient_id"].str.strip().eq("").any():
        raise ValueError("Manifest: missing patient ID")
    if not manifest["patient_id"].eq(manifest["sample_id"].str[:12]).all():
        raise ValueError("Manifest: patient ID disagrees with TCGA sample barcode")
    if not manifest["split"].isin(["train", "test"]).all():
        raise ValueError("Manifest: split must be train or test")
    manifest["label"] = binary_values(manifest["label"], "manifest")
    for split, target in (("test", reference), ("train", load_reference(train_path))):
        subset = manifest.loc[manifest["split"] == split, ["sample_id", "label"]]
        align_predictions(target, subset.rename(columns={"label": "y_true"}), f"manifest {split}")
    train = manifest.loc[manifest["split"] == "train"]
    test = manifest.loc[manifest["split"] == "test"].set_index("sample_id").loc[reference["sample_id"]]
    overlap = set(train["patient_id"]) & set(test["patient_id"])
    if overlap:
        raise ValueError(f"Manifest: {len(overlap)} patients occur in both train and test")
    return test.reset_index(), {
        "train_samples": len(train),
        "test_samples": len(test),
        "train_patients": int(train["patient_id"].nunique()),
        "test_patients": int(test["patient_id"].nunique()),
        "train_test_patient_overlap": len(overlap),
    }


def compute_metrics(table: pd.DataFrame, model: str) -> dict:
    true, pred, score = table["y_true"], table["y_pred"], table["score"]
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(true, pred, labels=[0, 1]).ravel())
    return {
        "model": model,
        "n_samples": len(table),
        "n_normal": int((true == 0).sum()),
        "n_tumor": int((true == 1).sum()),
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(true, pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(true, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(true, score)),
        "specificity": float(recall_score(true, pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_confusion_plot(row: dict, path: Path) -> None:
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    im = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]} = {matrix[i, j]}\n{proportions[i, j]:.2%} of row",
                    ha="center", va="center", fontsize=13,
                    color="white" if proportions[i, j] > 0.55 else "#17283A")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal (0)", "Tumor (1)"],
           yticklabels=["Normal (0)", "Tumor (1)"], xlabel="Predicted label",
           ylabel="True label", title=f"{row['model']} | test confusion matrix\nN = {row['n_samples']}; positive class = tumor")
    fig.colorbar(im, ax=ax, label="Proportion within true class", shrink=0.78)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_roc_plot(tables: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(8, 6.3), layout="constrained")
    zoom = ax.inset_axes([0.40, 0.20, 0.55, 0.40])
    records = []
    for model, table in tables.items():
        fpr, tpr, thresholds = roc_curve(table["y_true"], table["score"], pos_label=1, drop_intermediate=False)
        auc = metrics.set_index("model").loc[model, "roc_auc"]
        # Linear links match sklearn's trapezoidal AUC treatment of tied scores.
        ax.plot(fpr, tpr, color=COLORS[model], lw=2,
                label=f"{model} (AUC = {auc:.6f})")
        zoom.plot(fpr, tpr, color=COLORS[model], lw=2)
        row = metrics.set_index("model").loc[model]
        ax.scatter(row["fp"] / row["n_normal"], row["recall"], color=COLORS[model], s=40, zorder=4)
        zoom.scatter(row["fp"] / row["n_normal"], row["recall"], color=COLORS[model], s=30, zorder=4)
        records.append(pd.DataFrame({"model": model, "false_positive_rate": fpr,
                                     "true_positive_rate": tpr, "threshold": thresholds}))
    ax.plot([0, 1], [0, 1], "--", color="#9BA3AC", lw=1, label="Chance")
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.025), xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (tumor recall)", title="Random Forest vs SVM | independent test ROC")
    zoom.set(xlim=(-0.002, 0.08), ylim=(0.97, 1.001), title="Upper-left detail", xlabel="False positive rate", ylabel="True positive rate")
    zoom.tick_params(labelsize=9)
    zoom.grid(alpha=0.2)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return pd.concat(records, ignore_index=True)


def save_comparison_plot(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
    positions = np.arange(len(METRICS))
    for i, row in metrics.iterrows():
        values = row[list(METRICS)].to_numpy(dtype=float)
        y = positions + (i - 0.5) * 0.32
        ax.barh(y, values, height=0.28, color=COLORS[row["model"]], label=row["model"])
        for yi, value in zip(y, values):
            ax.text(value + 0.008, yi, f"{value:.6f}", va="center", fontsize=10)
    ax.set(yticks=positions, yticklabels=list(METRICS.values()), xlim=(0, 1.13),
           xlabel="Score (0 to 1)")
    ax.set_title("Independent test metrics | positive class = tumor (1)", pad=36)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.01), frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_evaluation(args: argparse.Namespace) -> pd.DataFrame:
    inputs = {"rf_predictions": args.rf_predictions, "svm_predictions": args.svm_predictions,
              "test_npz": args.test_data, "train_npz": args.train_data, "manifest": args.manifest}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Input not found: {path}")
    destinations = [args.output_dir / name for name in OUTPUT_FILES]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise ValueError("Outputs already exist; use a new --output-dir or --overwrite")
    if any(path.resolve() in {p.resolve() for p in inputs.values()} for path in destinations):
        raise ValueError("Output would overwrite an input")

    reference = load_reference(args.test_data)
    manifest, split_audit = validate_manifest(args.manifest, reference, args.train_data)
    tables = {
        "Random Forest": align_predictions(reference, load_predictions(args.rf_predictions, "Random Forest"), "Random Forest"),
        "SVM": align_predictions(reference, load_predictions(args.svm_predictions, "SVM"), "SVM"),
    }
    metrics = pd.DataFrame([compute_metrics(table, model) for model, table in tables.items()])
    aligned = reference.copy()
    aligned.insert(1, "patient_id", manifest["patient_id"])
    error_frames = []
    for prefix, (model, table) in zip(["rf", "svm"], tables.items()):
        aligned[f"{prefix}_y_pred"] = table["y_pred"]
        aligned[f"{prefix}_score"] = table["score"]
        errors = table.loc[table["y_true"] != table["y_pred"]].copy()
        errors.insert(0, "model", model)
        errors.insert(2, "patient_id", errors["sample_id"].map(manifest.set_index("sample_id")["patient_id"]))
        errors["error_type"] = np.where(errors["y_true"] == 1, "FN", "FP")
        errors["score_type"] = "tumor_probability" if prefix == "rf" else "decision_function"
        error_frames.append(errors)

    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    aligned.to_csv(args.output_dir / "evaluation/aligned_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(error_frames, ignore_index=True).to_csv(args.output_dir / "evaluation/misclassified_samples.csv", index=False, encoding="utf-8-sig")
    matrices = []
    for _, row in metrics.iterrows():
        name = "rf" if row["model"] == "Random Forest" else "svm"
        save_confusion_plot(row.to_dict(), args.output_dir / f"figures/{name}_confusion_matrix.png")
        matrices.extend([
            {"model": row["model"], "true_label": 0, "predicted_normal": int(row["tn"]), "predicted_tumor": int(row["fp"])},
            {"model": row["model"], "true_label": 1, "predicted_normal": int(row["fn"]), "predicted_tumor": int(row["tp"])},
        ])
    pd.DataFrame(matrices).to_csv(args.output_dir / "evaluation/confusion_matrices.csv", index=False, encoding="utf-8-sig")
    roc_points = save_roc_plot(tables, metrics, args.output_dir / "figures/roc_comparison.png")
    roc_points.to_csv(args.output_dir / "evaluation/roc_points.csv", index=False, encoding="utf-8-sig")
    save_comparison_plot(metrics, args.output_dir / "figures/model_metrics_comparison.png")
    audit = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "saved independent-test predictions; no training or parameter selection",
        "positive_label": 1,
        "negative_label": 0,
        "alignment_order": "test_data.npz sample_id",
        "checks": {"identical_test_sample_sets": True, "unique_ids": True,
                   "true_labels_match_test_npz_and_manifest": True,
                   "manifest_matches_train_npz": True, "finite_continuous_scores": True},
        "class_counts": {"normal_0": int((reference["y_true"] == 0).sum()),
                         "tumor_1": int((reference["y_true"] == 1).sum())},
        "split": split_audit,
        "always_tumor_accuracy": float(reference["y_true"].mean()),
        "score_semantics": {"Random Forest": "provided tumor_probability, larger favors label 1",
                            "SVM": "provided decision_score, larger favors label 1; not a probability"},
        "classification_policy": "use original y_pred; do not rethreshold scores",
        "zero_division_policy": 0,
        "expression_arrays_loaded": False,
        "models_loaded": False,
        "training_or_parameter_search_performed": False,
        "inputs": {key: {"path": path_label(path), "sha256": sha256(path)} for key, path in inputs.items()},
        "evaluator_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}},
        "outputs": {name: sha256(args.output_dir / name) for name in OUTPUT_FILES if not name.endswith("audit.json")},
    }
    (args.output_dir / "evaluation/audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-predictions", type=Path, default=ROOT / "results/rf_test_manual/predictions.csv")
    parser.add_argument("--svm-predictions", type=Path, default=ROOT / "results/svm/test_predictions.csv")
    parser.add_argument("--test-data", type=Path, default=DATA_DIR / "test_data.npz")
    parser.add_argument("--train-data", type=Path, default=DATA_DIR / "train_data.npz")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "sample_manifest.tsv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate this evaluator's output files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metrics = run_evaluation(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(metrics[["model", *METRICS]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Saved evaluation outputs to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
