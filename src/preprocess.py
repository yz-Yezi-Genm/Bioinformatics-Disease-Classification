"""Prepare TCGA-BRCA RNA-seq data for tumor-vs-normal classification.

Expected expression input: a TSV/TSV.GZ with Ensembl genes in rows, the first
column named ``Ensembl_ID``, and TCGA sample barcodes in the remaining columns.
Expected label input: a TSV/TSV.GZ with sample IDs and sample types. By default
the columns are ``sample`` and ``sample_type.samples``.

The script keeps all input genes. Feature filtering and model fitting are left
to downstream teammates and must be fit on training folds only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


LABEL_TO_INT = {"Solid Tissue Normal": 0, "Primary Tumor": 1}
BARCODE_TO_TYPE = {
    "01": "Primary Tumor",
    "11": "Solid Tissue Normal",
    "06": "Metastatic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path, *, dtype: object = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", dtype=dtype, low_memory=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", required=True, type=Path, help="Expression TSV or TSV.GZ")
    parser.add_argument("--labels", required=True, type=Path, help="Sample-label TSV or TSV.GZ")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--sample-id-column", default="sample", help="Sample ID column in label file")
    parser.add_argument("--label-column", default="sample_type.samples", help="Label column in label file")
    parser.add_argument("--split-manifest", type=Path,
                        help="Optional manifest with sample_id and split columns to reproduce an existing split")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--folds", type=int, default=5, help="Grouped stratified folds; first fold is held out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_path = args.expression.resolve()
    labels_path = args.labels.resolve()
    output_dir = args.output_dir.resolve()
    if not expression_path.is_file():
        raise FileNotFoundError(expression_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    clinical = read_tsv(labels_path, dtype=str)
    missing_columns = {args.sample_id_column, args.label_column} - set(clinical.columns)
    if missing_columns:
        raise ValueError(f"Label file is missing columns: {sorted(missing_columns)}")
    if clinical[args.sample_id_column].isna().any() or clinical[args.sample_id_column].duplicated().any():
        raise ValueError("Label sample IDs must be present and unique")

    expression = read_tsv(expression_path)
    if expression.columns[0] != "Ensembl_ID":
        raise ValueError("The first expression column must be named 'Ensembl_ID'")
    sample_ids = expression.columns[1:].astype(str).to_numpy(dtype=str)
    gene_ids = expression.iloc[:, 0].astype(str).to_numpy(dtype=str)
    if len(sample_ids) == 0 or len(gene_ids) == 0:
        raise ValueError("Expression matrix is empty")
    if pd.Index(sample_ids).has_duplicates:
        raise ValueError("Expression sample IDs are duplicated")
    if pd.Index(gene_ids).has_duplicates:
        raise ValueError("Gene IDs are duplicated")

    clinical_by_sample = clinical.set_index(args.sample_id_column).reindex(sample_ids)
    if clinical_by_sample[args.label_column].isna().any():
        missing = clinical_by_sample.index[clinical_by_sample[args.label_column].isna()].tolist()
        raise ValueError(f"Expression samples without labels (first 10): {missing[:10]}")

    sample_types = clinical_by_sample[args.label_column].astype(str)
    barcode_codes = np.asarray([sample.split("-")[3][:2] for sample in sample_ids])
    barcode_types = pd.Series(barcode_codes, index=sample_ids).map(BARCODE_TO_TYPE)
    known = barcode_types.notna().to_numpy()
    mismatch = known & (barcode_types.to_numpy() != sample_types.to_numpy())
    if mismatch.any():
        rows = np.flatnonzero(mismatch)[:10]
        examples = [(sample_ids[i], barcode_types.iloc[i], sample_types.iloc[i]) for i in rows]
        raise ValueError(f"Clinical labels conflict with TCGA barcode types: {examples}")

    keep = sample_types.isin(LABEL_TO_INT).to_numpy()
    selected_sample_ids = sample_ids[keep]
    selected_types = sample_types.to_numpy(dtype=str)[keep]
    y = np.asarray([LABEL_TO_INT[value] for value in selected_types], dtype=np.uint8)
    patient_ids = np.asarray([sample[:12] for sample in selected_sample_ids])

    # Input is documented as log2(count + 1); retain its scale and only transpose.
    raw = expression.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(raw).all():
        raise ValueError("Expression matrix contains non-finite values")
    if np.any(raw < 0):
        raise ValueError("Expression matrix contains negative values")
    values = raw.T[keep]

    if args.split_manifest:
        split_manifest_path = args.split_manifest.resolve()
        split_manifest = read_tsv(split_manifest_path, dtype=str)
        required = {"sample_id", "split"}
        if not required.issubset(split_manifest.columns):
            raise ValueError("--split-manifest must contain sample_id and split columns")
        if split_manifest["sample_id"].duplicated().any():
            raise ValueError("Split manifest sample IDs are duplicated")
        split_by_sample = split_manifest.set_index("sample_id")["split"]
        missing_split = [sample for sample in selected_sample_ids if sample not in split_by_sample.index]
        if missing_split:
            raise ValueError(f"Selected samples missing from split manifest (first 10): {missing_split[:10]}")
        split_values = split_by_sample.reindex(selected_sample_ids).to_numpy(dtype=str)
        if not np.isin(split_values, ["train", "test"]).all():
            raise ValueError("Split manifest must label every selected sample as train or test")
        if "label" in split_manifest.columns:
            label_by_sample = pd.to_numeric(
                split_manifest.set_index("sample_id")["label"], errors="coerce")
            manifest_labels = label_by_sample.reindex(selected_sample_ids).to_numpy()
            if np.isnan(manifest_labels).any() or not np.array_equal(manifest_labels.astype(np.uint8), y):
                raise ValueError("Split manifest labels do not match labels derived from the label file")
        train_idx = np.flatnonzero(split_values == "train")
        test_idx = np.flatnonzero(split_values == "test")
        split_method = f"provided manifest: {split_manifest_path.name}"
    else:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(values, y, groups=patient_ids))
        split_method = f"StratifiedGroupKFold(n_splits={args.folds}, shuffle=True); first fold"
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Both train and test splits must contain samples")
    if set(np.unique(y[train_idx])) != {0, 1} or set(np.unique(y[test_idx])) != {0, 1}:
        raise ValueError("Both classes must be represented in train and test splits")
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])
    if train_patients & test_patients:
        raise AssertionError("Patient leakage detected between train and test")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both tumor and normal classes must be present")

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = np.ascontiguousarray(values[train_idx], dtype=np.float32)
    X_test = np.ascontiguousarray(values[test_idx], dtype=np.float32)
    y_train, y_test = y[train_idx], y[test_idx]
    np.savez_compressed(output_dir / "train_data.npz", X=X_train, y=y_train,
                        sample_id=selected_sample_ids[train_idx])
    np.savez_compressed(output_dir / "test_data.npz", X=X_test, y=y_test,
                        sample_id=selected_sample_ids[test_idx])

    pd.DataFrame({"feature_index": np.arange(len(gene_ids), dtype=np.int32),
                  "ensembl_id": gene_ids}).to_csv(
        output_dir / "feature_ids.tsv", sep="\t", index=False, lineterminator="\n")
    split = np.full(len(selected_sample_ids), "", dtype=object)
    split[train_idx], split[test_idx] = "train", "test"
    pd.DataFrame({
        "sample_id": selected_sample_ids,
        "patient_id": patient_ids,
        "sample_type": selected_types,
        "label": y,
        "split": split,
    }).to_csv(output_dir / "sample_manifest.tsv", sep="\t", index=False, lineterminator="\n")

    def counts(labels: np.ndarray) -> dict[str, int]:
        return {"normal_0": int((labels == 0).sum()), "tumor_1": int((labels == 1).sum())}

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TCGA-BRCA",
        "task": "primary tumor vs solid tissue normal",
        "expression_input": expression_path.name,
        "expression_sha256": sha256_file(expression_path),
        "labels_input": labels_path.name,
        "labels_sha256": sha256_file(labels_path),
        "expression_shape_genes_by_samples": [int(len(gene_ids)), int(len(sample_ids))],
        "label_rows": int(len(clinical)),
        "expression_samples_without_label": 0,
        "label_rows_without_expression": int(clinical[args.sample_id_column].nunique() - len(sample_ids)),
        "label_column": args.label_column,
        "label_source": "provided sample-label table; cross-checked against TCGA barcode when recognizable",
        "sample_type_counts_expression": {str(k): int(v) for k, v in sample_types.value_counts().items()},
        "selected_samples": int(len(selected_sample_ids)),
        "excluded_sample_type_counts": {str(k): int(v) for k, v in sample_types.value_counts().items()
                                         if k not in LABEL_TO_INT},
        "selected_class_counts_all": counts(y),
        "expression_missing_values": int(expression.iloc[:, 1:].isna().to_numpy().sum()),
        "expression_nonfinite_values": 0,
        "expression_negative_values": 0,
        "duplicate_gene_ids": 0,
        "duplicate_sample_ids": 0,
        "expression_orientation_output": "samples_by_genes",
        "expression_dtype_output": "float32",
        "label_dtype_output": "uint8",
        "label_mapping": LABEL_TO_INT,
        "expression_transform": "retained as supplied; expected log2(count + 1) per source README",
        "features_filtered_in_preprocessing": 0,
        "features_retained": int(len(gene_ids)),
        "split_method": split_method,
        "split_seed": int(args.seed),
        "group_key": "first 12 characters of TCGA sample barcode",
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "patient_overlap": 0,
        "train_class_counts": counts(y_train),
        "test_class_counts": counts(y_test),
    }
    for filename, key in (("train_data.npz", "train_npz_sha256"),
                          ("test_data.npz", "test_npz_sha256"),
                          ("feature_ids.tsv", "feature_ids_sha256"),
                          ("sample_manifest.tsv", "sample_manifest_sha256")):
        audit[key] = sha256_file(output_dir / filename)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Reopen arrays and verify schema, dimensions, labels, and patient isolation.
    with np.load(output_dir / "train_data.npz", allow_pickle=False) as train, \
         np.load(output_dir / "test_data.npz", allow_pickle=False) as test:
        assert train["X"].shape == (len(train_idx), len(gene_ids))
        assert test["X"].shape == (len(test_idx), len(gene_ids))
        assert train["y"].dtype == np.uint8 and test["y"].dtype == np.uint8
        assert np.isfinite(train["X"]).all() and np.isfinite(test["X"]).all()
        assert len(train["sample_id"]) == len(train["y"])
        assert len(test["sample_id"]) == len(test["y"])

    print(json.dumps({
        "output_dir": str(output_dir),
        "genes": int(len(gene_ids)),
        "samples": int(len(selected_sample_ids)),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_class_counts": counts(y_train),
        "test_class_counts": counts(y_test),
        "patient_overlap": len(train_patients & test_patients),
        "files": sorted(path.name for path in output_dir.iterdir()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
