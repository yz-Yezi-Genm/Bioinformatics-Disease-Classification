"""Training-only reference comparison; does NOT replace E/F/G final evaluation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.feature_selection import create_selector, load_training_data, versions, write_json

SEED=20261031
AUC_TOLERANCE=0.001  # Predeclared practical tolerance, not a significance test.

def make_pipeline(model,k,jobs=2):
    steps=[('select',create_selector(k))]
    if model=='rf':
        clf=RandomForestClassifier(n_estimators=200,max_features='sqrt',class_weight='balanced_subsample',
                                   random_state=SEED,n_jobs=jobs)
    elif model=='svm':
        steps.append(('scale',StandardScaler()))
        clf=SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced',cache_size=512)
    else:
        raise ValueError(model)
    return Pipeline(steps+[('classifier',clf)])

def run(data_dir,output_dir,ks=(100,500,1000),jobs=2):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    X,y,sample_ids,groups,feature_ids,audit,checksums=load_training_data(data_dir)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    splits=list(cv.split(X,y,groups))
    assignments=np.full(len(y),-1,dtype=int)
    fold_info=[]
    for fold,(train_idx,val_idx) in enumerate(splits,1):
        overlap=set(groups[train_idx]) & set(groups[val_idx])
        if overlap or len(np.unique(y[train_idx]))!=2 or len(np.unique(y[val_idx]))!=2:
            raise ValueError('Invalid group split or missing class in a fold.')
        assignments[val_idx]=fold
        fold_info.append(dict(fold=fold,train_samples=len(train_idx),validation_samples=len(val_idx),
                              train_patients=len(set(groups[train_idx])),validation_patients=len(set(groups[val_idx])),
                              train_normal=int((y[train_idx]==0).sum()),train_tumor=int((y[train_idx]==1).sum()),
                              validation_normal=int((y[val_idx]==0).sum()),validation_tumor=int((y[val_idx]==1).sum()),patient_overlap=0))
    pd.DataFrame({'sample_id':sample_ids,'patient_id':groups,'label':y,'validation_fold':assignments}).to_csv(out/'cv_fold_assignments.csv',index=False)
    pd.DataFrame(fold_info).to_csv(out/'cv_fold_summary.csv',index=False)
    metadata=dict(scope='training-only fixed-model reference; not final E/F/G results',seed=SEED,folds=5,
                  grouping='patient_id',data_checksums=checksums,versions=versions(),candidate_ks=list(ks),
                  k_rule=f'Smallest k with mean validation ROC-AUC >= best mean minus {AUC_TOLERANCE}.',
                  auc_tolerance=AUC_TOLERANCE,test_npz_loaded=False,
                  rf=dict(n_estimators=200,max_features='sqrt',class_weight='balanced_subsample',random_state=SEED,n_jobs=jobs),
                  svm=dict(kernel='rbf',C=1.0,gamma='scale',class_weight='balanced',scaling='StandardScaler inside fold'),
                  decision_threshold='RF predict default; SVM decision_function zero',
                  note='CV scores guide k selection and are not an unbiased final performance estimate.')
    write_json(out/'cv_metadata.json',metadata)
    metrics=[]
    predictions=[]
    selections=[]
    # Bound numeric threads; CV is sequential to avoid copying a high-dimensional X in many workers.
    with threadpool_limits(limits=2):
        for fold,(train_idx,val_idx) in enumerate(splits,1):
            X_train,X_val=X[train_idx],X[val_idx]
            for k in ks:
                for model in ('rf','svm'):
                    start=time.perf_counter()
                    pipeline=make_pipeline(model,k,jobs)
                    pipeline.fit(X_train,y[train_idx])
                    score=(pipeline.predict_proba(X_val)[:,1] if model=='rf' else pipeline.decision_function(X_val))
                    pred=pipeline.predict(X_val)
                    tn,fp,fn,tp=confusion_matrix(y[val_idx],pred,labels=[0,1]).ravel()
                    row=dict(model=model,k=k,fold=fold,roc_auc=float(roc_auc_score(y[val_idx],score)),
                             balanced_accuracy=float(balanced_accuracy_score(y[val_idx],pred)),
                             tumor_f1=float(f1_score(y[val_idx],pred,pos_label=1)),
                             tumor_recall=float(tp/(tp+fn)),normal_recall_specificity=float(tn/(tn+fp)),
                             tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                             constant_removed=int(X.shape[1]-pipeline.named_steps['select'].n_features_after_variance_),
                             selected_features=int(pipeline.named_steps['select'].n_features_out_),
                             elapsed_seconds=round(time.perf_counter()-start,3))
                    metrics.append(row)
                    predictions.extend(dict(model=model,k=k,fold=fold,sample_id=sample_ids[i],label=int(y[i]),
                                            prediction=int(p),score=float(s)) for i,p,s in zip(val_idx,pred,score))
                    if model=='rf':  # Same deterministic selector for both classifiers in a given fold/k.
                        selections.extend(dict(k=k,fold=fold,feature_index=int(i),ensembl_id=feature_ids[i])
                                          for i in pipeline.named_steps['select'].get_support(indices=True))
                    pd.DataFrame(metrics).to_csv(out/'cv_fold_metrics.csv',index=False)
                    print(json.dumps(row),flush=True)
                    del pipeline
            del X_train,X_val
    pd.DataFrame(predictions).to_csv(out/'cv_predictions.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'cv_selected_features.csv',index=False)
    frame=pd.DataFrame(metrics)
    metric_names=['roc_auc','balanced_accuracy','tumor_f1','tumor_recall','normal_recall_specificity','elapsed_seconds']
    summary=frame.groupby(['model','k'])[metric_names].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary['folds']=5
    summary['evaluation_scope']='training_group_cv_reference'
    summary.to_csv(out/'k_comparison.csv',index=False)
    choices={}
    for model in ('rf','svm'):
        candidates=summary[summary.model==model]
        best=float(candidates.roc_auc_mean.max())
        choice=int(candidates.loc[candidates.roc_auc_mean>=best-AUC_TOLERANCE,'k'].min())
        selector=create_selector(choice).fit(X,y)
        selector.feature_table(feature_ids).to_csv(out/f'selected_features_{model}_reference.csv',index=False)
        choices[model]=dict(reference_k=choice,best_mean_auc=best,
                            selected_mean_auc=float(candidates.loc[candidates.k==choice,'roc_auc_mean'].iloc[0]),
                            status='reference recommendation; E/F must confirm for their final model',
                            feature_list_fit_scope='entire training set after reference CV k choice')
    write_json(out/'reference_recommendations.json',dict(rule=metadata['k_rule'],models=choices,test_npz_loaded=False))
    print(json.dumps(choices),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output-dir',default='results/feature_selection')
    p.add_argument('--ks',nargs='+',type=int,default=[100,500,1000])
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    if not args.ks or len(set(args.ks))!=len(args.ks) or min(args.ks)<1:
        p.error('Provide distinct positive k values.')
    run(args.data_dir,args.output_dir,args.ks,args.jobs)

if __name__=='__main__':
    main()
