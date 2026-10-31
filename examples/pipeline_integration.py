"""Copyable E/F integration example, with patient-grouped training-only CV."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.feature_selection import create_selector, load_training_data


def build_pipelines(k=100):
    # These are NEW unfitted selectors, never selectors fitted on all training data.
    rf=Pipeline([
        ('select',create_selector(k)),
        ('classifier',RandomForestClassifier(n_estimators=200,max_features='sqrt',class_weight='balanced_subsample',random_state=20261031,n_jobs=2)),
    ])
    svm=Pipeline([
        ('select',create_selector(k)),
        ('scale',StandardScaler()),
        ('classifier',SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced')),
    ])
    return {'rf':rf,'svm':svm}


def evaluate_training_only(data_dir,k=100):
    X,y,sample_ids,patient_ids,feature_ids,audit,checksums=load_training_data(data_dir)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261031)
    splits=list(cv.split(X,y,groups=patient_ids))
    # Sequential outer CV avoids high-dimensional matrix copies in many processes.
    return {name:cross_validate(pipeline,X,y,cv=splits,scoring=['roc_auc','balanced_accuracy','f1'],
                               n_jobs=1,error_score='raise')
            for name,pipeline in build_pipelines(k).items()}

# After all choices are frozen, E/F can fit a fresh final Pipeline on full X_train.
# They must load X_test in the identical original feature order and apply that
# fitted Pipeline directly. Never select test genes separately or tune on y_test.
