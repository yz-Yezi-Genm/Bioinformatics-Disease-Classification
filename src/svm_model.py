"""Issue #8: training-only SVM search followed by one held-out evaluation.

Run from the repository root: python -m src.svm_model
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

# Support both python -m src.svm_model and python src/svm_model.py.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_selection import create_selector, load_training_data, sha256, versions, write_json


def build_pipeline():
    return Pipeline([
        ('select', create_selector(k=100)),
        ('scale', StandardScaler()),
        ('classifier', SVC(class_weight='balanced', probability=False)),
    ])


def parameter_grid(cs, gammas):
    return [
        {'classifier__kernel': ['linear'], 'classifier__C': cs},
        {'classifier__kernel': ['rbf'], 'classifier__C': cs,
         'classifier__gamma': gammas},
    ]


def load_cv_splits(path, samples, y, groups):
    """Align D's saved fold IDs by sample ID; never regenerate random folds."""
    frame = pd.read_csv(path, dtype={'sample_id': str, 'patient_id': str})
    if not frame.sample_id.is_unique or set(frame.sample_id) != set(samples):
        raise ValueError('CV assignment sample IDs must match the training set exactly.')
    frame = frame.set_index('sample_id').loc[samples].reset_index()
    if not np.array_equal(frame.label.to_numpy(), y):
        raise ValueError('CV labels differ from training labels.')
    if not np.array_equal(frame.patient_id.to_numpy(), groups):
        raise ValueError('CV patient IDs differ from training patient IDs.')
    if set(frame.validation_fold) != {1, 2, 3, 4, 5}:
        raise ValueError('Expected the five saved folds 1..5.')
    if frame.groupby('patient_id').validation_fold.nunique().max() != 1:
        raise ValueError('A patient appears in multiple validation folds.')
    splits = []
    for fold in range(1, 6):
        valid = np.flatnonzero(frame.validation_fold.to_numpy() == fold)
        train = np.flatnonzero(frame.validation_fold.to_numpy() != fold)
        if set(y[train]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError('Both classes must occur in every training/validation fold.')
        splits.append((train, valid))
    return splits, frame


def select_best(results):
    """Highest mean AUC, then balanced accuracy, then candidate order."""
    return min(range(len(results['params'])), key=lambda i: (
        -results['mean_test_roc_auc'][i],
        -results['mean_test_balanced_accuracy'][i], i))


def evaluate(y, prediction, scores):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(n_samples=len(y), accuracy=accuracy_score(y, prediction),
                balanced_accuracy=balanced_accuracy_score(y, prediction),
                roc_auc=roc_auc_score(y, scores),
                tumor_precision=precision_score(y, prediction, zero_division=0),
                tumor_recall=recall_score(y, prediction),
                specificity=recall_score(y, prediction, pos_label=0),
                tumor_f1=f1_score(y, prediction),
                macro_f1=f1_score(y, prediction, average='macro'),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                confusion_matrix_order='rows=true, columns=predicted; labels=[0,1]')


def load_test(data_dir, audit, train_samples, feature_count):
    path = data_dir / 'test_data.npz'
    if sha256(path) != audit['test_npz_sha256']:
        raise ValueError('Test NPZ checksum mismatch.')
    with np.load(path, allow_pickle=False) as arrays:
        X, y, samples = arrays['X'], arrays['y'], arrays['sample_id'].astype(str)
    manifest = pd.read_csv(data_dir / 'sample_manifest.tsv', sep='\t').set_index('sample_id')
    if len(set(samples)) != len(samples) or set(samples) != set(manifest.index[manifest.split == 'test']):
        raise ValueError('Test sample IDs differ from manifest.')
    if not np.array_equal(manifest.loc[samples, 'label'].to_numpy(), y):
        raise ValueError('Test labels differ from manifest.')
    if X.shape != (len(y), feature_count) or not np.isfinite(X).all() or (X < 0).any():
        raise ValueError('Invalid test matrix.')
    if set(y) != {0, 1} or set(s[:12] for s in samples) & set(s[:12] for s in train_samples):
        raise ValueError('Test classes or patient isolation are invalid.')
    return X, y, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/tcga_brca_20261029_v2'))
    parser.add_argument('--folds-file', type=Path, default=Path('results/feature_selection/cv_fold_assignments.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/svm'))
    parser.add_argument('--model-file', type=Path, default=Path('models/svm_pipeline.joblib'))
    parser.add_argument('--c-values', nargs='+', type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument('--gamma-values', nargs='+', default=['scale', '0.001', '0.01'])
    args = parser.parse_args()
    if any(not np.isfinite(c) or c <= 0 for c in args.c_values):
        parser.error('C values must be positive and finite.')
    gammas = [g if g in ('scale', 'auto') else float(g) for g in args.gamma_values]
    if any(isinstance(g, float) and (not np.isfinite(g) or g <= 0) for g in gammas):
        parser.error('Numeric gamma values must be positive and finite.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; use a new directory to preserve existing results.')
    if args.model_file.exists():
        parser.error('Model file already exists; use a new model path.')
    start = time.perf_counter()
    X, y, samples, groups, feature_ids, audit, checksums = load_training_data(args.data_dir)
    splits, assignments = load_cv_splits(args.folds_file, samples, y, groups)
    grid = parameter_grid(args.c_values, gammas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(issue=8, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    training_shape=list(X.shape), input_hashes=checksums,
                    folds_sha256=sha256(args.folds_file), versions=versions(),
                    joblib_version=joblib.__version__, feature_count=100,
                    pipeline=['D GeneSelector', 'StandardScaler', 'SVC'],
                    class_weight='balanced', probability=False, decision_threshold=0.0,
                    parameter_grid=grid, candidate_count=len(list(ParameterGrid(grid))),
                    cv='D saved patient-grouped folds, aligned by sample_id',
                    fold_reference_seed=20261031,
                    selection_rule='maximum mean validation ROC-AUC; exact ties: maximum mean balanced accuracy, then grid order',
                    test_used_for_selection=False, n_jobs=1, numerical_threads=2)
    write_json(args.output_dir / 'protocol.json', protocol)
    assignments.to_csv(args.output_dir / 'cv_fold_assignments.csv', index=False)
    scoring = dict(roc_auc='roc_auc', balanced_accuracy='balanced_accuracy',
                   accuracy='accuracy', tumor_f1='f1', macro_f1='f1_macro',
                   tumor_recall='recall',
                   specificity=make_scorer(recall_score, pos_label=0))
    search = GridSearchCV(build_pipeline(), grid, cv=splits, scoring=scoring,
                          refit=False, n_jobs=1, error_score='raise', verbose=2)
    with threadpool_limits(limits=2):
        search.fit(X, y)
        results = search.cv_results_
        table = pd.DataFrame(results)
        table['candidate_id'] = np.arange(len(table))
        table['params'] = [json.dumps(p, sort_keys=True) for p in results['params']]
        table.to_csv(args.output_dir / 'cv_results.csv', index=False)
        best = select_best(results)
        chosen = results['params'][best]
        selection = dict(candidate_id=best, parameters=chosen, feature_count=100,
                         mean_cv={name: float(results['mean_test_' + name][best]) for name in scoring},
                         std_cv={name: float(results['std_test_' + name][best]) for name in scoring},
                         test_npz_loaded=False,
                         frozen_at_utc=datetime.now(timezone.utc).isoformat())
        # Persist the decision BEFORE opening the held-out matrix.
        write_json(args.output_dir / 'selected_parameters.json', selection)
        print('Frozen training-only selection:', json.dumps(selection), flush=True)
        model = clone(build_pipeline()).set_params(**chosen).fit(X, y)
        model.named_steps['select'].feature_table(feature_ids).to_csv(
            args.output_dir / 'selected_features.csv', index=False)
        args.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_file, compress=3)
        X_test, y_test, test_samples = load_test(args.data_dir, audit, samples, len(feature_ids))
        prediction = model.predict(X_test)
        scores = model.decision_function(X_test)
        pd.DataFrame(dict(sample_id=test_samples, y_true=y_test, y_pred=prediction,
                          decision_score=scores)).to_csv(args.output_dir / 'test_predictions.csv', index=False)
        metrics = evaluate(y_test, prediction, scores)
        metrics['evaluation_scope'] = 'held-out test; no parameter or threshold changes after evaluation'
        write_json(args.output_dir / 'test_metrics.json', metrics)
        restored = joblib.load(args.model_file)
        np.testing.assert_array_equal(restored.predict(X_test), prediction)
        np.testing.assert_allclose(restored.decision_function(X_test), scores, rtol=0, atol=1e-12)
    write_json(args.output_dir / 'run_summary.json', dict(
        status='completed', elapsed_seconds=time.perf_counter()-start,
        model_file=args.model_file.as_posix(), model_sha256=sha256(args.model_file),
        test_npz_sha256=audit['test_npz_sha256'], model_reload_prediction_check='PASS',
        source_sha256=sha256(Path(__file__)), selector_source_sha256=sha256(Path(__file__).with_name('feature_selection.py')),
        artifacts_sha256={p.name:sha256(p) for p in args.output_dir.iterdir() if p.is_file()}))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
