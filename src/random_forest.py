"""Training-only RF selection: python -m src.random_forest --help."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.feature_selection import create_selector

SEED = 20261031
METRICS = ['roc_auc', 'balanced_accuracy', 'tumor_f1', 'tumor_recall', 'specificity']
# Fixed before evaluation; candidate 0 is also the D-compatible basic RF.
CANDIDATES = [
    dict(max_features='sqrt', min_samples_leaf=1, max_depth=None, class_weight='balanced_subsample'),
    dict(max_features='sqrt', min_samples_leaf=1, max_depth=None, class_weight=None),
    dict(max_features='sqrt', min_samples_leaf=2, max_depth=None, class_weight='balanced'),
    dict(max_features='sqrt', min_samples_leaf=4, max_depth=12, class_weight='balanced_subsample'),
    dict(max_features=0.3, min_samples_leaf=1, max_depth=None, class_weight='balanced_subsample'),
    dict(max_features=0.3, min_samples_leaf=2, max_depth=12, class_weight='balanced'),
    dict(max_features=0.3, min_samples_leaf=4, max_depth=None, class_weight=None),
    dict(max_features=0.5, min_samples_leaf=1, max_depth=8, class_weight='balanced'),
    dict(max_features=0.5, min_samples_leaf=2, max_depth=None, class_weight='balanced_subsample'),
    dict(max_features='sqrt', min_samples_leaf=2, max_depth=8, class_weight=None),
]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_training_inputs(data_dir, folds_path):
    """Validate the training contract once, without D's manifest-reading loader."""
    paths = [data_dir / 'train_data.npz', data_dir / 'feature_ids.tsv',
             folds_path, folds_path.with_name('cv_metadata.json')]
    checksums = {str(p): sha256(p) for p in paths}
    metadata = json.loads(paths[3].read_text(encoding='utf-8'))
    for path in paths[:2]:
        if checksums[str(path)] != metadata['data_checksums'][path.name]:
            raise ValueError(f'Training input differs from D reference: {path.name}')
    with np.load(paths[0], allow_pickle=False) as arrays:
        X, y, samples = arrays['X'], arrays['y'], arrays['sample_id'].astype(str)
    features = pd.read_csv(paths[1], sep='\t')
    folds = pd.read_csv(folds_path, dtype={'sample_id': str, 'patient_id': str})
    if X.ndim != 2 or y.ndim != 1 or samples.ndim != 1 or X.shape[0] != len(y) or len(y) != len(samples):
        raise ValueError('X, y and sample_id dimensions do not agree.')
    if not np.isfinite(X).all() or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('X must be finite and y must contain exactly both labels 0/1.')
    if len(set(samples)) != len(samples):
        raise ValueError('Training sample_id must be unique.')
    if (X.shape[1] != 60660 or not np.array_equal(features.feature_index.to_numpy(), np.arange(X.shape[1]))
            or features.ensembl_id.isna().any() or not features.ensembl_id.is_unique):
        raise ValueError('Expected original 60,660-column feature mapping in original order.')
    if not folds.sample_id.is_unique or set(folds.sample_id) != set(samples):
        raise ValueError('D fold samples differ from training samples.')
    folds = folds.set_index('sample_id').loc[samples].reset_index()
    patients = np.asarray([s[:12] for s in samples])
    if not np.array_equal(folds.label.to_numpy(), y) or not np.array_equal(folds.patient_id.to_numpy(), patients):
        raise ValueError('D fold labels or patients differ from training data.')
    if metadata['seed'] != SEED or metadata['folds'] != 5 or set(folds.validation_fold) != {1, 2, 3, 4, 5}:
        raise ValueError('Expected D five-fold assignment with reference seed 20261031.')
    if not folds.groupby('patient_id').validation_fold.nunique().eq(1).all():
        raise ValueError('A patient occurs in multiple validation folds.')
    splits = []
    for fold in range(1, 6):
        valid = np.flatnonzero(folds.validation_fold.to_numpy() == fold)
        train = np.flatnonzero(folds.validation_fold.to_numpy() != fold)
        if not np.array_equal(np.unique(y[train]), [0, 1]) or not np.array_equal(np.unique(y[valid]), [0, 1]):
            raise ValueError(f'Both classes required in train and validation fold {fold}.')
        splits.append((train, valid))
    return X, y, samples, features, folds, splits, checksums


def evaluate(estimator, candidate, X, y, samples, splits):
    rows, predictions = [], []
    for fold, (train, valid) in enumerate(splits, 1):
        started = time.perf_counter()
        fitted = clone(estimator).fit(X[train], y[train])
        pred = fitted.predict(X[valid])
        probability = fitted.predict_proba(X[valid])[:, 1]
        rows.append(dict(candidate=candidate, fold=fold, n_train=len(train), n_validation=len(valid),
                         roc_auc=roc_auc_score(y[valid], probability),
                         balanced_accuracy=balanced_accuracy_score(y[valid], pred),
                         tumor_f1=f1_score(y[valid], pred, pos_label=1),
                         tumor_recall=recall_score(y[valid], pred, pos_label=1),
                         specificity=recall_score(y[valid], pred, pos_label=0),
                         seconds=time.perf_counter() - started))
        predictions.append(pd.DataFrame(dict(candidate=candidate, fold=fold, sample_id=samples[valid],
                                             label=y[valid], prediction=pred, tumor_probability=probability)))
        print(f'{candidate} fold {fold}: AUC={rows[-1]["roc_auc"]:.6f}', flush=True)
    return rows, pd.concat(predictions, ignore_index=True)


def environment_versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, scikit_learn=sklearn.__version__, joblib=joblib.__version__,
                    threadpoolctl=__import__('threadpoolctl').__version__)


def run_supplement(args, X, y, samples, folds, splits, checksums):
    """Post-test training-only analysis; the frozen model is never refitted."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    original_dir = Path('results/rf')
    original = json.loads((original_dir / 'run_config.json').read_text(encoding='utf-8'))
    versions = environment_versions()
    changes = [('trees_100', dict(n_estimators=100)), ('trees_400', dict(n_estimators=400)),
               ('split_5', dict(min_samples_split=5)), ('split_10', dict(min_samples_split=10))]
    baseline = dict(n_estimators=200, min_samples_split=2, random_state=SEED, n_jobs=args.jobs, **CANDIDATES[0])
    reuse = (versions == original['versions'] and checksums == original['input_files_sha256']
             and original['seed'] == SEED and original['k'] == 100 and original['n_estimators'] == 200
             and original['candidates'][0] == CANDIDATES[0] and original['rf_jobs'] == args.jobs
             and original['source_sha256'][str(Path('src/feature_selection.py').resolve())]
                 == sha256(Path('src/feature_selection.py')))
    config = dict(scope='post_independent_test_training_only_parameter_analysis',
                  purpose='Parameter analysis only; no model replacement or new selection',
                  primary_metric='mean validation ROC-AUC', k=100, seed=SEED, baseline=baseline,
                  changes=dict(changes), baseline_reused=reuse, versions=versions,
                  input_files_sha256=checksums, command=sys.argv, source_sha256=sha256(__file__),
                  started_utc=datetime.now(timezone.utc).isoformat(), rf_jobs=args.jobs,
                  outer_jobs=1, numerical_threads=2, decision_rule='predict default; no threshold tuning',
                  test_expression_read=False, model_written=False)
    write_json(out / 'run_config.json', config)
    folds.to_csv(out / 'cv_fold_assignments.csv', index=False)
    rows, predictions = [], []
    with threadpool_limits(limits=2):
        if reuse:
            old_metrics = pd.read_csv(original_dir / 'cv_fold_metrics.csv')
            rows.extend(old_metrics.loc[old_metrics.candidate == 'rf_00'].assign(candidate='baseline').to_dict('records'))
            old_predictions = pd.read_csv(original_dir / 'best_oof_predictions.csv')
            if set(old_predictions.candidate) != {'rf_00'}:
                raise ValueError('Saved baseline predictions are not rf_00.')
            predictions.append(old_predictions.assign(candidate='baseline'))
        for name, change in ([] if reuse else [('baseline', {})]) + changes:
            params = baseline | change
            pipeline = Pipeline([('select', create_selector(k=100)),
                                 ('classifier', RandomForestClassifier(**params))])
            r, p = evaluate(pipeline, name, X, y, samples, splits)
            rows.extend(r)
            predictions.append(p)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'cv_fold_metrics.csv', index=False)
    summary = metrics.groupby('candidate', sort=False)[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out / 'cv_summary.csv')
    pd.concat(predictions, ignore_index=True).to_csv(out / 'cv_predictions.csv', index=False)
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/tcga_brca_20261029_v2'))
    parser.add_argument('--folds', type=Path, default=Path('results/feature_selection/cv_fold_assignments.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/rf'))
    parser.add_argument('--model-path', type=Path, default=Path('models/rf_pipeline.joblib'))
    parser.add_argument('--jobs', type=int, default=2, help='Positive RF thread count; folds and candidates run sequentially.')
    parser.add_argument('--supplement', action='store_true', help='Four post-test parameter analyses only; never save or select a model.')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    started = time.perf_counter()
    X, y, samples, features, folds, splits, checksums = load_training_inputs(args.data_dir, args.folds)
    if args.supplement:
        run_supplement(args, X, y, samples, folds, splits, checksums)
        return
    # New output locations prevent accidental replacement of an earlier experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model_path.exists():
        raise FileExistsError(args.model_path)
    folds.to_csv(args.output_dir / 'cv_fold_assignments.csv', index=False)
    versions = environment_versions()
    config = dict(scope='training_set_internal_model_selection', primary_metric='mean validation ROC-AUC',
                  tie_rule='first candidate in predefined order', seed=SEED, k=100, n_estimators=200,
                  candidates=CANDIDATES, search_combinations=len(CANDIDATES), baseline_candidate='rf_00',
                  cv='D saved patient folds; no new split', decision_rule='estimator.predict default; no threshold tuning',
                  rf_jobs=args.jobs, outer_jobs=1, numerical_threads=2, shape=list(X.shape),
                  classes={str(c): int((y == c).sum()) for c in (0, 1)},
                  patients=int(folds.patient_id.nunique()), input_files_sha256=checksums,
                  versions=versions, python_executable=sys.executable, command=sys.argv,
                  model_path=str(args.model_path), started_utc=datetime.now(timezone.utc).isoformat(),
                  test_data_accessed=False, source_sha256={str(p): sha256(p) for p in
                      (Path(__file__), Path(__file__).with_name('feature_selection.py'))})
    write_json(args.output_dir / 'run_config.json', config)
    rows, predictions = [], []
    with threadpool_limits(limits=2):
        r, p = evaluate(DummyClassifier(strategy='most_frequent'), 'majority', X, y, samples, splits)
        rows.extend(r)
        predictions.append(p)
        for index, params in enumerate(CANDIDATES):
            pipeline = Pipeline([('select', create_selector(k=100)),
                                 ('classifier', RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                                      n_jobs=args.jobs, **params))])
            r, p = evaluate(pipeline, f'rf_{index:02d}', X, y, samples, splits)
            rows.extend(r)
            predictions.append(p)
            pd.DataFrame(rows).to_csv(args.output_dir / 'cv_fold_metrics.csv', index=False)
        metrics = pd.DataFrame(rows)
        summary = metrics.groupby('candidate', sort=False)[METRICS].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.to_csv(args.output_dir / 'cv_summary.csv')
        search = summary.drop(index='majority').reset_index()
        search['parameters'] = [json.dumps(p, sort_keys=True) for p in CANDIDATES]
        search.to_csv(args.output_dir / 'parameter_search.csv', index=False)
        best = search.loc[search.roc_auc_mean.idxmax()]
        best_params = CANDIDATES[int(best.candidate.split('_')[1])]
        final = Pipeline([('select', create_selector(k=100)),
                          ('classifier', RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                               n_jobs=args.jobs, **best_params))]).fit(X, y)
        joblib.dump(final, args.model_path, compress=3)
        # Use a few TRAINING samples only to verify exact serialization round trip.
        positions = np.r_[np.flatnonzero(y == 0)[:4], np.flatnonzero(y == 1)[:4]]
        before_pred, before_prob = final.predict(X[positions]), final.predict_proba(X[positions])
        reloaded = joblib.load(args.model_path)
        np.testing.assert_array_equal(before_pred, reloaded.predict(X[positions]))
        np.testing.assert_array_equal(before_prob, reloaded.predict_proba(X[positions]))
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(args.output_dir / 'cv_predictions.csv', index=False)
    all_predictions.loc[all_predictions.candidate == best.candidate].to_csv(args.output_dir / 'best_oof_predictions.csv', index=False)
    table = final.named_steps['select'].feature_table(features.ensembl_id.to_numpy(dtype=str))
    table['rf_importance'] = final.named_steps['classifier'].feature_importances_
    table.to_csv(args.output_dir / 'selected_genes_importance.csv', index=False)
    result = dict(scope='training_set_internal_model_selection_not_independent_test', best_candidate=best.candidate,
                  best_params=best_params, metrics={m: {'mean': float(best[m + '_mean']),
                    'std_ddof1': float(best[m + '_std'])} for m in METRICS},
                  final_fit_samples=len(y), final_fit_features=X.shape[1], selected_genes=len(table),
                  model_path=str(args.model_path), model_sha256=sha256(args.model_path),
                  reload_check=dict(samples=samples[positions].tolist(), predictions_exact=True, probabilities_exact=True),
                  elapsed_seconds=time.perf_counter() - started, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / 'best_model.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
