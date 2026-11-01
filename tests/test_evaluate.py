"""Tests for evaluation metrics and prediction input validation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluate import (
    DATA_DIR,
    ROOT,
    align_predictions,
    compute_metrics,
    load_predictions,
    load_reference,
    parse_args,
    run_evaluation,
    validate_manifest,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"sample_id": ["a", "b", "c", "d", "e", "f"],
                                       "y_true": [0, 0, 0, 1, 1, 1]})
        self.predictions = self.reference.assign(y_pred=[0, 0, 1, 0, 1, 1],
                                                 score=[0.1, 0.2, 0.6, 0.3, 0.8, 0.9])

    def write_predictions(self, directory, table):
        path = Path(directory) / "rf.csv"
        table.rename(columns={"y_true": "true_label", "y_pred": "predicted_label",
                              "score": "tumor_probability"}).to_csv(path, index=False)
        return path

    def test_known_counts_and_pairwise_auc(self):
        row = compute_metrics(self.predictions, "test")
        self.assertEqual([row[k] for k in ("tn", "fp", "fn", "tp")], [2, 1, 1, 2])
        for key in ("accuracy", "precision", "recall", "f1_score", "specificity", "balanced_accuracy"):
            self.assertAlmostEqual(row[key], 2 / 3)
        # Of 3*3 tumor-normal score pairs, only (0.3, 0.6) is incorrectly ordered.
        self.assertAlmostEqual(row["roc_auc"], 8 / 9)

    def test_auc_uses_continuous_score_with_tie_half_credit(self):
        table = self.predictions.copy()
        table["y_pred"] = 0
        table["score"] = [0.1, 0.2, 0.3, 0.3, 0.8, 0.9]
        row = compute_metrics(table, "test")
        self.assertEqual(row["precision"], 0)
        self.assertEqual(row["recall"], 0)
        self.assertAlmostEqual(row["roc_auc"], 8.5 / 9)

    def test_shuffled_rows_are_aligned_by_id(self):
        shuffled = self.predictions.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(align_predictions(self.reference, shuffled, "test"), self.predictions)

    def test_missing_extra_and_duplicate_ids_are_rejected(self):
        for bad in (self.predictions.iloc[:-1],
                    self.predictions.assign(sample_id=["a", "b", "c", "d", "e", "extra"]),
                    self.predictions.assign(sample_id=["a", "a", "c", "d", "e", "f"])):
            with self.subTest(ids=bad.sample_id.tolist()), self.assertRaises(ValueError):
                align_predictions(self.reference, bad, "test")

    def test_shared_wrong_labels_cannot_pass_reference_check(self):
        wrong = self.predictions.copy()
        wrong.loc[0, "y_true"] = 1
        for model in ("Random Forest", "SVM"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "true labels"):
                align_predictions(self.reference, wrong, model)

    def test_invalid_labels_and_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for column, value in [("y_true", 2), ("y_pred", 0.5), ("score", np.nan),
                                  ("score", np.inf), ("score", -0.01), ("score", 1.01)]:
                bad = self.predictions.copy()
                bad[column] = bad[column].astype(float)
                bad.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_predictions(self.write_predictions(directory, bad), "Random Forest")

    def test_binary_class_labels_cannot_be_used_as_roc_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = self.predictions.copy()
            bad["score"] = bad["y_pred"]
            with self.assertRaisesRegex(ValueError, "continuous"):
                load_predictions(self.write_predictions(directory, bad), "Random Forest")

    def test_blank_ids_and_missing_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in (self.predictions.assign(sample_id=[" ", "b", "c", "d", "e", "f"]),
                        self.predictions.drop(columns="score")):
                with self.assertRaises(ValueError):
                    load_predictions(self.write_predictions(directory, bad), "Random Forest")

    def test_svm_negative_margins_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "svm.csv"
            self.predictions.rename(columns={"score": "decision_score"}).assign(
                decision_score=[-3, -2, 1, -1, 2, 3]).to_csv(path, index=False)
            self.assertEqual(load_predictions(path, "SVM")["score"].iloc[0], -3)

    def test_single_class_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            np.savez(path, sample_id=np.array(["a", "b"]), y=np.array([1, 1]))
            with self.assertRaisesRegex(ValueError, "both classes"):
                load_reference(path)

    def test_patient_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # Same patient has a normal training sample and a tumor test sample.
            ids = ["TCGA-AA-0001-11A", "TCGA-AA-0002-01A", "TCGA-AA-0001-01A", "TCGA-AA-0003-11A"]
            manifest = pd.DataFrame({"sample_id": ids, "patient_id": [x[:12] for x in ids],
                                     "label": [0, 1, 1, 0], "split": ["train", "train", "test", "test"]})
            manifest.to_csv(directory / "manifest.tsv", sep="\t", index=False)
            np.savez(directory / "train.npz", sample_id=np.array(ids[:2]), y=np.array([0, 1]))
            reference = pd.DataFrame({"sample_id": ids[2:], "y_true": [1, 0]})
            with self.assertRaisesRegex(ValueError, "patients occur"):
                validate_manifest(directory / "manifest.tsv", reference, directory / "train.npz")

    def test_saved_predictions_match_npz_manifest_and_counts(self):
        reference = load_reference(DATA_DIR / "test_data.npz")
        _, audit = validate_manifest(DATA_DIR / "sample_manifest.tsv", reference, DATA_DIR / "train_data.npz")
        self.assertEqual((len(reference), audit["test_patients"], audit["train_test_patient_overlap"]), (243, 218, 0))
        for model, path, counts, auc in [
            ("Random Forest", "results/rf_test_manual/predictions.csv", [24, 1, 1, 217], 0.9997247706422019),
            ("SVM", "results/svm/test_predictions.csv", [25, 0, 2, 216], 0.9990825688073394),
        ]:
            table = align_predictions(reference, load_predictions(ROOT / path, model), model)
            result = compute_metrics(table, model)
            self.assertEqual([result[k] for k in ("tn", "fp", "fn", "tp")], counts)
            self.assertAlmostEqual(result["roc_auc"], auc)
            self.assertEqual(result["n_samples"], 243)

    def test_full_run_exports_and_protects_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output-dir", directory])
            metrics = run_evaluation(args)
            self.assertEqual(len(metrics), 2)
            output = Path(directory)
            errors = pd.read_csv(output / "evaluation/misclassified_samples.csv")
            self.assertEqual(len(errors), 4)
            self.assertEqual(errors.sample_id.nunique(), 3)
            self.assertEqual(len(pd.read_csv(output / "evaluation/aligned_predictions.csv")), 243)
            audit = json.loads((output / "evaluation/audit.json").read_text())
            self.assertFalse(audit["expression_arrays_loaded"])
            self.assertFalse(audit["training_or_parameter_search_performed"])
            for name in audit["outputs"]:
                self.assertTrue((output / name).stat().st_size > 0)
            with self.assertRaisesRegex(ValueError, "Outputs already exist"):
                run_evaluation(args)


if __name__ == "__main__":
    unittest.main()
