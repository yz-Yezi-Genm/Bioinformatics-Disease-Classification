"""Tests for visualization inputs and output directory protection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation_io import ROOT, load_inputs
from src.visualize import main


class VisualizationInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name)
        (self.results / "evaluation").mkdir()
        self.metrics = pd.read_csv(ROOT / "results/metrics.csv")
        self.predictions = pd.read_csv(ROOT / "results/evaluation/aligned_predictions.csv")
        self.test_path = ROOT / "data/processed/tcga_brca_20261029_v2/test_data.npz"

    def save_inputs(self):
        self.metrics.to_csv(self.results / "metrics.csv", index=False)
        self.predictions.to_csv(self.results / "evaluation/aligned_predictions.csv", index=False)

    def test_real_results_reconcile(self):
        self.save_inputs()
        metrics, predictions = load_inputs(self.results, self.test_path)
        self.assertEqual(len(predictions), 243)
        self.assertEqual(metrics.loc["Random Forest", "fp"], 1)
        self.assertEqual(metrics.loc["SVM", "fn"], 2)

    def test_shuffled_predictions_keep_test_order(self):
        expected = self.predictions.sample_id.tolist()
        self.predictions = self.predictions.sample(frac=1, random_state=12)
        self.save_inputs()
        _, predictions = load_inputs(self.results, self.test_path)
        self.assertEqual(predictions.sample_id.tolist(), expected)

    def test_changed_metric_is_rejected(self):
        self.metrics.loc[0, "accuracy"] = 1.0
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "accuracy disagrees"):
            load_inputs(self.results, self.test_path)

    def test_duplicate_ids_are_rejected(self):
        self.predictions.loc[0, "sample_id"] = self.predictions.loc[1, "sample_id"]
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_inputs(self.results, self.test_path)

    def test_wrong_reference_label_is_rejected(self):
        self.predictions.loc[0, "y_true"] = 1 - self.predictions.loc[0, "y_true"]
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "true labels"):
            load_inputs(self.results, self.test_path)

    def test_invalid_score_is_rejected(self):
        self.predictions.loc[0, "svm_score"] = np.inf
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "invalid scores"):
            load_inputs(self.results, self.test_path)

    def test_existing_output_directory_is_protected(self):
        with self.assertRaisesRegex(ValueError, "new or empty"):
            main(["--output-dir", str(ROOT / "results/figures")])


if __name__ == "__main__":
    unittest.main()
