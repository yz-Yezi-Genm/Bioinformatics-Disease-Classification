"""Verify exported gene mapping, grouped CV outputs and reported metrics."""
import argparse
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from src.feature_selection import load_training_data, create_selector, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--results-dir',default='results/feature_selection')
    args=p.parse_args()
    out=Path(args.results_dir)
    suite=unittest.defaultTestLoader.discover('tests')
    test_result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise RuntimeError('Unit tests failed.')
    X,y,samples,groups,ids,audit,checksums=load_training_data(args.data_dir)
    candidate_sets={}
    for k in [100,500,1000]:
        table=pd.read_csv(out/f'selected_features_k{k}.csv')
        s=create_selector(k).fit(X,y)
        assert len(table)==k and table.feature_index.is_unique and table.ensembl_id.is_unique
        np.testing.assert_array_equal(table.feature_index,s.get_support(indices=True))
        np.testing.assert_array_equal(table.ensembl_id,ids[table.feature_index])
        np.testing.assert_array_equal(table.output_column,np.arange(k))
        np.testing.assert_allclose(table.score,s.scores_[table.feature_index],rtol=1e-12)
        np.testing.assert_allclose(s.transform(X[:10]),X[:10,table.feature_index])
        candidate_sets[k]=set(table.feature_index)
    assert candidate_sets[100]<=candidate_sets[500]<=candidate_sets[1000]
    assignments=pd.read_csv(out/'cv_fold_assignments.csv')
    np.testing.assert_array_equal(assignments.sample_id,samples)
    np.testing.assert_array_equal(assignments.patient_id,groups)
    np.testing.assert_array_equal(assignments.label,y)
    assert assignments.groupby('patient_id').validation_fold.nunique().max()==1
    for fold in range(1,6):
        is_val=assignments.validation_fold==fold
        assert not set(groups[is_val]) & set(groups[~is_val])
        assert set(y[is_val])=={0,1} and set(y[~is_val])=={0,1}
    metrics=pd.read_csv(out/'cv_fold_metrics.csv')
    predictions=pd.read_csv(out/'cv_predictions.csv')
    comparison=pd.read_csv(out/'k_comparison.csv')
    fold_selected=pd.read_csv(out/'cv_selected_features.csv')
    assert len(metrics)==30 and len(predictions)==6*len(y)
    for (model,k),part in predictions.groupby(['model','k']):
        assert len(part)==len(y) and part.sample_id.is_unique and set(part.sample_id)==set(samples)
        joined=part.merge(assignments,on='sample_id',validate='one_to_one',suffixes=('','_expected'))
        np.testing.assert_array_equal(joined.label,joined.label_expected)
        np.testing.assert_array_equal(joined.fold,joined.validation_fold)
    for row in metrics.itertuples():
        part=predictions[(predictions.model==row.model)&(predictions.k==row.k)&(predictions.fold==row.fold)]
        for actual,expected in [(roc_auc_score(part.label,part.score),row.roc_auc),
                                (balanced_accuracy_score(part.label,part.prediction),row.balanced_accuracy),
                                (f1_score(part.label,part.prediction),row.tumor_f1)]:
            np.testing.assert_allclose(actual,expected,atol=1e-12)
    for (model,k),part in metrics.groupby(['model','k']):
        row=comparison[(comparison.model==model)&(comparison.k==k)].iloc[0]
        for metric in ['roc_auc','balanced_accuracy','tumor_f1','tumor_recall','normal_recall_specificity']:
            np.testing.assert_allclose(part[metric].mean(),row[metric+'_mean'],atol=1e-12)
            np.testing.assert_allclose(part[metric].std(ddof=1),row[metric+'_std'],atol=1e-12)
    for (k,fold),part in fold_selected.groupby(['k','fold']):
        assert len(part)==k and part.feature_index.is_unique
        np.testing.assert_array_equal(part.ensembl_id,ids[part.feature_index])
    recommendations=json.loads((out/'reference_recommendations.json').read_text(encoding='utf-8'))
    for model,choice in recommendations['models'].items():
        frame=comparison[comparison.model==model]
        expected=int(frame.loc[frame.roc_auc_mean>=frame.roc_auc_mean.max()-.001,'k'].min())
        assert choice['reference_k']==expected
        pd.testing.assert_frame_equal(pd.read_csv(out/f'selected_features_{model}_reference.csv'),pd.read_csv(out/f'selected_features_k{expected}.csv'))
    payload=dict(status='PASS',unit_tests=test_result.testsRun,real_data_checks=[
        'input hashes and training labels/IDs match C manifest',
        'all three exported gene lists match fresh fit and original matrix columns',
        'candidate lists nested and output columns correctly ordered',
        'patient groups never cross CV validation folds',
        '30 model/k/fold runs, each sample predicted out-of-fold once per model/k',
        'fold ROC-AUC, balanced accuracy and F1 recomputed from stored predictions',
        'summary means and sample standard deviations recomputed',
        'fold feature IDs and reference k decisions checked'],
        test_npz_loaded=False)
    write_json(out/'verification.json',payload)
    print(json.dumps(payload,indent=2))

if __name__=='__main__':
    main()
