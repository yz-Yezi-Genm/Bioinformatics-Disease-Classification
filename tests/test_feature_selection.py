import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from src.feature_selection import create_selector

class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(43)
        self.y=np.tile([0,1],30)
        self.X=rng.normal(size=(60,8))
        self.X[:,0]=0
        self.X[:,3]=7
        self.X[:,6]+=self.y*4
        self.ids=np.asarray([f'GENE{i}' for i in range(8)])

    def test_original_column_mapping_after_two_filters(self):
        s=create_selector(2).fit(self.X,self.y)
        self.assertIn(6,s.get_support(indices=True))
        self.assertFalse(s.get_support()[0] or s.get_support()[3])
        table=s.feature_table(self.ids)
        np.testing.assert_allclose(s.transform(self.X),self.X[:,table.feature_index])
        np.testing.assert_array_equal(table.ensembl_id,self.ids[table.feature_index])
        np.testing.assert_array_equal(table.output_column,[0,1])

    def test_transform_does_not_refit(self):
        s=create_selector(2).fit(self.X,self.y)
        before=s.get_support().copy()
        changed=self.X.copy()
        changed[:,0]=np.arange(60)*1000
        s.transform(changed)
        np.testing.assert_array_equal(before,s.get_support())

    def test_dataframe_column_order_is_checked(self):
        frame=pd.DataFrame(self.X,columns=self.ids)
        s=create_selector(2).fit(frame,self.y)
        with self.assertRaises(ValueError): s.transform(frame.iloc[:,::-1])
        with self.assertRaises(ValueError): s.transform(self.X)
        with self.assertRaises(ValueError): s.feature_table(self.ids[::-1])

    def test_invalid_k_and_single_class(self):
        for k in [0,-1,7,1.5,True]:
            with self.subTest(k=k), self.assertRaises(ValueError): create_selector(k).fit(self.X,self.y)
        with self.assertRaises(ValueError): create_selector(2).fit(self.X,np.zeros(60))

    def test_bad_values_and_wrong_shape(self):
        bad=self.X.copy(); bad[0,1]=np.nan
        with self.assertRaises(ValueError): create_selector(2).fit(bad,self.y)
        s=create_selector(2).fit(self.X,self.y)
        with self.assertRaises(ValueError): s.transform(self.X[:,:-1])
        with self.assertRaises(ValueError): s.feature_table(self.ids[:-1])
        with self.assertRaises(ValueError): create_selector(2).fit(np.ones((60,8)),self.y)

    def test_failed_refit_invalidates_old_model(self):
        s=create_selector(2).fit(self.X,self.y)
        with self.assertRaises(ValueError): s.fit(self.X,np.zeros(60))
        with self.assertRaises(NotFittedError): s.transform(self.X)

    def test_clone_and_real_group_cv_pipeline(self):
        selector=create_selector(2)
        fitted=clone(selector).fit(self.X,self.y)
        self.assertFalse(hasattr(selector,'selected_indices_'))
        pipeline=Pipeline([('select',selector),('clf',LogisticRegression())])
        groups=np.repeat(np.arange(30),2)
        cv=StratifiedGroupKFold(3,shuffle=True,random_state=1)
        splits=list(cv.split(self.X,self.y,groups))
        result=cross_validate(pipeline,self.X,self.y,cv=splits,scoring='roc_auc',return_estimator=True,error_score='raise')
        for estimator,(train,val) in zip(result['estimator'],splits):
            self.assertEqual(estimator.named_steps['select'].n_samples_fit_,len(train))
            self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertTrue(np.isfinite(result['test_score']).all())

    def test_repeatability_and_all_mode(self):
        a=create_selector(2).fit(self.X,self.y)
        b=create_selector(2).fit(self.X,self.y)
        np.testing.assert_array_equal(a.get_support(),b.get_support())
        self.assertEqual(create_selector('all').fit(self.X,self.y).transform(self.X).shape,(60,6))

if __name__=='__main__':
    unittest.main()
