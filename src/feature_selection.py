"""Reusable, training-only gene selection for issue #5.

Run from repository root: python -m src.feature_selection --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, scikit_learn=sklearn.__version__)


class GeneSelector(TransformerMixin, BaseEstimator):
    """VarianceThreshold followed by SelectKBest(f_classif).

    fit receives only current training-fold X/y. transform never refits.
    NumPy callers MUST keep the original feature order. DataFrame callers get
    exact column-name/order validation. Output columns retain input order.
    Ensembl IDs belong in the column map, never in the numeric X matrix.
    """

    def __init__(self, k=500, variance_threshold=0.0):
        self.k = k
        self.variance_threshold = variance_threshold

    def fit(self, X, y):
        # A failed refit must not leave an older fitted selector usable.
        for name in list(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        if self.k != 'all' and (isinstance(self.k, bool) or not isinstance(self.k, numbers.Integral) or self.k < 1):
            raise ValueError('k must be a positive integer or "all".')
        if not isinstance(self.variance_threshold, numbers.Real) or not np.isfinite(self.variance_threshold) or self.variance_threshold < 0:
            raise ValueError('variance_threshold must be finite and nonnegative.')
        names = None
        if isinstance(X, pd.DataFrame):
            names = np.asarray(X.columns, dtype=object)
            if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError('DataFrame feature names must be unique strings.')
        X, y = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError('Both labels 0=normal and 1=tumor must occur in each training fold.')
        if min(np.bincount(y.astype(int))) < 2:
            raise ValueError('Each training class needs at least two samples for this analysis.')
        variance = VarianceThreshold(threshold=self.variance_threshold)
        filtered = variance.fit_transform(X)
        if self.k != 'all' and self.k > filtered.shape[1]:
            raise ValueError(f'k={self.k} exceeds {filtered.shape[1]} features remaining after variance filtering.')
        kbest = SelectKBest(score_func=f_classif, k=self.k)
        kbest.fit(filtered, y)
        if np.isnan(kbest.scores_).any() or (kbest.scores_ < 0).any():
            raise ValueError('Invalid ANOVA F scores; inspect the training data. No features were exported.')
        # Positive infinity denotes perfect class separation; preserve and report it.
        retained_indices = np.flatnonzero(variance.get_support())
        selected_indices = retained_indices[kbest.get_support()]
        self.variance_filter_ = variance
        self.kbest_ = kbest
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self.n_features_after_variance_ = filtered.shape[1]
        self.selected_indices_ = selected_indices
        self.n_features_out_ = len(selected_indices)
        self.scores_ = np.full(X.shape[1], np.nan)
        self.pvalues_ = np.full(X.shape[1], np.nan)
        self.scores_[retained_indices] = kbest.scores_
        self.pvalues_[retained_indices] = kbest.pvalues_
        if names is not None:
            self.feature_names_in_ = names
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_indices_')
        if hasattr(self, 'feature_names_in_'):
            if not isinstance(X, pd.DataFrame) or not np.array_equal(np.asarray(X.columns, dtype=object), self.feature_names_in_):
                raise ValueError('Feature names/order differ from fit. Supply the same DataFrame columns.')
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} feature columns, got {X.shape[1]}.')
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, 'selected_indices_')
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_indices_')
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', np.asarray([f'x{i}' for i in range(self.n_features_in_)]))
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or len(names) != self.n_features_in_:
            raise ValueError('Feature ID count must match the original number of columns.')
        if hasattr(self, 'feature_names_in_') and not np.array_equal(names, self.feature_names_in_):
            raise ValueError('Feature IDs differ from fitted DataFrame columns.')
        return names[self.selected_indices_]

    def feature_table(self, feature_ids, all_features=False):
        """Output is in original-column order; rank is for presentation only."""
        check_is_fitted(self, 'selected_indices_')
        feature_ids = np.asarray(feature_ids, dtype=str)
        if feature_ids.ndim != 1 or len(feature_ids) != self.n_features_in_ or len(set(feature_ids)) != len(feature_ids):
            raise ValueError('Unique feature IDs in original column order are required.')
        self.get_feature_names_out(feature_ids)
        valid = self.variance_filter_.get_support()
        ranks = np.full(self.n_features_in_, np.nan)
        ranks[valid] = pd.Series(self.scores_[valid]).rank(method='min', ascending=False).to_numpy()
        output_column = np.full(self.n_features_in_, -1, dtype=int)
        output_column[self.selected_indices_] = np.arange(self.n_features_out_)
        frame = pd.DataFrame({'feature_index':np.arange(self.n_features_in_), 'ensembl_id':feature_ids,
                              'score':self.scores_, 'pvalue_nominal':self.pvalues_, 'rank':ranks,
                              'passed_variance':valid, 'selected':self.get_support(), 'output_column':output_column})
        return frame if all_features else frame.loc[frame.selected].reset_index(drop=True)


def create_selector(k=500):
    """Return a fresh UNFITTED estimator for an E/F training Pipeline."""
    return GeneSelector(k=k)


def load_training_data(data_dir):
    """Validate and load only train_data.npz; never load test expression or labels."""
    data_dir = Path(data_dir)
    audit = json.loads((data_dir / 'audit.json').read_text(encoding='utf-8'))
    checksums = {}
    for file, key in [('train_data.npz','train_npz_sha256'), ('feature_ids.tsv','feature_ids_sha256'),
                      ('sample_manifest.tsv','sample_manifest_sha256')]:
        actual = sha256(data_dir / file)
        if actual != audit[key]:
            raise ValueError(f'Checksum mismatch: {file}')
        checksums[file] = actual
    with np.load(data_dir / 'train_data.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {'X','y','sample_id'}:
            raise ValueError('Unexpected NPZ keys.')
        X, y, sample_ids = arrays['X'], arrays['y'], arrays['sample_id'].astype(str)
    features = pd.read_csv(data_dir / 'feature_ids.tsv', sep='\t')
    manifest = pd.read_csv(data_dir / 'sample_manifest.tsv', sep='\t', dtype={'sample_id':str,'patient_id':str})
    if not manifest.sample_id.is_unique or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate sample IDs.')
    if not np.array_equal(features.feature_index.to_numpy(), np.arange(X.shape[1])) or not features.ensembl_id.is_unique:
        raise ValueError('Invalid feature mapping.')
    rows = manifest.set_index('sample_id').loc[sample_ids]
    if set(sample_ids) != set(manifest.loc[manifest.split == 'train','sample_id']):
        raise ValueError('Training sample set differs from manifest.')
    if not (rows.split == 'train').all() or not np.array_equal(rows.label.to_numpy(), y):
        raise ValueError('Training labels/split do not match manifest.')
    groups = rows.patient_id.to_numpy(dtype=str)
    if not np.array_equal(groups, np.asarray([s[:12] for s in sample_ids])):
        raise ValueError('Patient IDs differ from barcode prefixes.')
    if set(groups) & set(manifest.loc[manifest.split == 'test','patient_id']):
        raise ValueError('Patient leakage between provided train/test sets.')
    if X.shape[0] != len(y) or len(y) != len(sample_ids) or not np.isfinite(X).all() or (X < 0).any():
        raise ValueError('Expression/label dimensions or numerical values are invalid.')
    if not np.array_equal(np.unique(y), [0,1]):
        raise ValueError('Both binary labels are required.')
    return X, y, sample_ids, groups, features.ensembl_id.to_numpy(dtype=str), audit, checksums


def run_selection(data_dir, output_dir, ks):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X,y,samples,groups,ids,audit,checksums = load_training_data(data_dir)
    records = []
    for k in ks:
        started = time.perf_counter()
        selector = create_selector(k).fit(X,y)
        table = selector.feature_table(ids)
        table.to_csv(output_dir / f'selected_features_k{k}.csv', index=False, encoding='utf-8')
        if not records:
            selector.feature_table(ids, all_features=True).drop(columns=['selected','output_column']).to_csv(
                output_dir / 'feature_scores.csv', index=False, encoding='utf-8')
        records.append(dict(k=k, input_features=X.shape[1], constant_removed=X.shape[1]-selector.n_features_after_variance_,
                            after_variance=selector.n_features_after_variance_, selected_features=selector.n_features_out_,
                            fit_scope='full_training_set_descriptive_only', elapsed_seconds=round(time.perf_counter()-started,3),
                            infinite_scores=int(np.isposinf(selector.scores_).sum())))
        print(json.dumps(records[-1]), flush=True)
    pd.DataFrame(records).to_csv(output_dir / 'selection_summary.csv', index=False)
    metadata = dict(dataset='TCGA-BRCA preprocessed v2', task='normal 0 vs primary tumor 1',
                    data_checksums=checksums, versions=versions(), training_shape=list(X.shape),
                    classes={str(c):int((y==c).sum()) for c in [0,1]}, patients=len(set(groups)),
                    expression_min=float(X.min()), expression_max=float(X.max()), finite_values=True,
                    train_constant_features_recomputed=int(np.count_nonzero(np.ptp(X, axis=0)==0)),
                    audit_constant_features=audit['train_constant_features_detected'],
                    variance_threshold=0.0, score_func='f_classif (float64 computation)', candidate_ks=ks,
                    fit_scope='full_training_set_descriptive_only', test_npz_loaded=False,
                    note='Do not reuse full-training selected lists inside CV; refit selector in each training fold.')
    write_json(output_dir / 'run_metadata.json', metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='results/feature_selection')
    parser.add_argument('--ks', nargs='+', type=int, default=[100,500,1000])
    args=parser.parse_args()
    if len(set(args.ks)) != len(args.ks):
        parser.error('Duplicate k values are not allowed.')
    run_selection(args.data_dir,args.output_dir,args.ks)


if __name__=='__main__':
    main()
