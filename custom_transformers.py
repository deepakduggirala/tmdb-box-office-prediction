import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import mean_squared_error


import datetime
import ast

class ReleaseDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        
        def year_convertor(s):
            yy = int(s.split('/')[2])
            if yy > 17:
                return 1900+yy
            else:
                return 2000+yy
            
        def day_of_week_convertor(s):
            year = year_convertor(s)
            month = int(s.split('/')[0])
            day = int(s.split('/')[1])
            
            return datetime.date(year, month, day).weekday() # Monday is 0
        
        release_date = pd.Series(X.reshape(-1,))
        release_year = release_date.apply(year_convertor)
        release_month = release_date.apply(lambda s: int(s.split('/')[1]))
        release_day_of_week = release_date.apply(day_of_week_convertor)
        
        release_year.name = 'release_year'
        release_month.name = 'release_month'
        release_day_of_week.name = 'release_day_of_week'
        
        return pd.concat([release_year, release_month, release_day_of_week], axis=1)
    
class AttributeCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.Series(X.reshape(-1,)).apply(lambda text: len(ast.literal_eval(text))).values.reshape(-1, 1)


    
def make_attribute_count_pipeline():
    return make_pipeline(
            SimpleImputer(strategy='constant', fill_value='[]'), 
            AttributeCountTransformer(), 
            SimpleImputer(strategy='median', missing_values=0),
            StandardScaler()
    )
    
class MultinomialAttributeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = []
        pass
    def fit(self, X, y=None):
        self.attr_set = set()
        for attrList in X:
            self.attr_set.update(set(attrList))
        self.attr_cardinality = len(self.attr_set)
        
        self.categories_ = [s.replace(' ', '_') for s in sorted(self.attr_set)]
        
        self.attr_idx_map = dict(zip(self.categories_, range(self.attr_cardinality)))
        return self
    def transform(self, X):
        arr = np.zeros((X.shape[0], self.attr_cardinality))
        
        for i, attrList in enumerate(X):
            idx = list(map(lambda attr: self.attr_idx_map[attr.replace(' ', '_')], set(attrList)&self.attr_set))
            arr[i, idx] = 1
            
        return arr

def convert_attr(key):
    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i[key])
        return L
    return convert    
    
class ExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.Series(X.reshape(-1,)).apply(convert_attr(self.key))
    
def make_name_extraction_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='constant', fill_value='[]'), 
        ExtractionTransformer('name'), 
        MultinomialAttributeTransformer()
    )
    

    
class ConditionalPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, pipelineA=None, pipelineB=None, make_mask=None, thresh=None):
        self.pipelineA = pipelineA
        self.pipelineB = pipelineB
        self.make_mask = make_mask
        self.thresh = thresh
    def fit(self, X, y=None):
        mask = self.make_mask(X, self.thresh)
        self.pipelineA.fit(X[mask], y[mask])
        self.pipelineB.fit(X[~mask], y[~mask])
        return self
    def predict(self, X):
        mask = self.make_mask(X, self.thresh)
        self.pred_mask = mask
        y_pred_a = self.pipelineA.predict(X[mask])
        y_pred_b = self.pipelineB.predict(X[~mask])
        
        y = np.zeros_like(np.hstack((y_pred_a, y_pred_b)))
        y[mask] = y_pred_a
        y[~mask] = y_pred_b
        
        return y
        
    def score(self, y_test, y_pred, scorer='rmse'):
        if scorer == 'rmse':
            scorer = lambda y_test, y_pred: mean_squared_error(y_test, y_pred, squared=False)
        mask = self.pred_mask
        self.scoreA_ = scorer(y_test[mask], y_pred[mask])
        self.scoreB_ = scorer(y_test[~mask], y_pred[~mask])
        self.score_ = scorer(y_test, y_pred)
        
        return (self.score_, self.scoreA_, self.scoreB_)
        
        
    
RuntimeImputer = make_pipeline(SimpleImputer(strategy='median'), SimpleImputer(missing_values=0, strategy='median'))

presenceBinarizer = FunctionTransformer(lambda x: x.where(cond=x.isna(), other=1).where(cond=~x.isna(), other=0))

class UniqueCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(map(lambda c: dict(zip(*np.unique(c, return_counts=True))), X))