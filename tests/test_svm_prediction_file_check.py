"""Checks for saved SVM prediction comparison and output protection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.check_svm_prediction_files import (
    ROOT, compare_tables, read_predictions, run_check, validate_reported_metrics,
)


class PredictionFileCheckTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({"sample_id": ["A", "B", "C", "D"],
            "y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1],
            "score": [-0.8, 0.1, 0.6, 1.2]})
        self.reference = self.table[["sample_id", "y_true"]].copy()

    def compare(self, new):
        return compare_tables(self.table, new, self.reference)[0]

    def test_reordered_rows_align_by_id(self):
        result = self.compare(self.table.iloc[::-1].copy())
        self.assertEqual(result["prediction_consistency"], "exact_match")
        self.assertFalse(result["row_order_equal"])

    def test_duplicate_id_is_rejected(self):
        new = self.table.copy(); new.loc[1, "sample_id"] = "A"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare(new)

    def test_different_sample_set_is_rejected(self):
        new = self.table.copy(); new.loc[1, "sample_id"] = "E"
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            self.compare(new)

    def test_changed_true_label_is_rejected(self):
        new = self.table.copy(); new.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(ValueError, "True labels"):
            self.compare(new)

    def test_changed_prediction_is_reported(self):
        new = self.table.copy(); new.loc[0, "y_pred"] = 1
        result = self.compare(new)
        self.assertEqual(result["prediction_consistency"], "different")
        self.assertEqual(result["predicted_label_mismatch_count"], 1)

    def test_roundoff_is_not_reported_as_exact(self):
        new = self.table.copy(); new.loc[0, "score"] += 1e-15
        result = self.compare(new)
        self.assertEqual(result["prediction_consistency"], "within_tolerance")
        self.assertFalse(result["decision_scores_exact_equal"])

    def test_material_score_difference_is_reported(self):
        new = self.table.copy(); new.loc[0, "score"] += 1e-5
        self.assertEqual(self.compare(new)["prediction_consistency"], "different")

    def test_nonfinite_score_is_rejected(self):
        new = self.table.copy(); new.loc[0, "score"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.compare(new)

    def test_invalid_tolerance_is_rejected(self):
        for atol in [-1, np.nan, np.inf]:
            with self.subTest(atol=atol), self.assertRaises(ValueError):
                compare_tables(self.table, self.table, self.reference, atol)

    def test_csv_round_trip_preserves_saved_score(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"predictions.csv"
            path.write_text("sample_id,y_true,y_pred,decision_score\nA,0,0,0.12345678901234567\n", encoding="utf-8")
            self.assertEqual(read_predictions(path).score.iloc[0], float("0.12345678901234567"))

    def test_reported_metrics_must_match_predictions(self):
        table = read_predictions(ROOT/"results/svm_reproduce_v1/test_predictions.csv")
        metrics = json.loads((ROOT/"results/svm_reproduce_v1/test_metrics.json").read_text())
        validate_reported_metrics(metrics, table)
        for key, value in [("accuracy", 0.5), ("macro_f1", float("nan")),
                           ("confusion_matrix", [[25, 0], [1, 217]])]:
            changed = copy.deepcopy(metrics); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reported_metrics(changed, table)

    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"records"; output.mkdir()
            keep = output/"keep.txt"; keep.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                run_check(ROOT/"results/svm", ROOT/"results/svm_reproduce_v1", output, Path(temp)/"report.md")
            self.assertEqual(keep.read_text(), "unchanged")

    def test_input_directories_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            run_check(ROOT/"results/svm", ROOT/"results/svm", Path(temp)/"records", Path(temp)/"report.md")


if __name__ == "__main__":
    unittest.main()
