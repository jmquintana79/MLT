# -*- coding: utf-8 -*-
# @Author: Juan Quintana
# @Date:   2018-09-18 13:25:04
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-18 13:35:04

"""
CUSTOM ESTIMATOR AS DECORATORS for Scikit-Learn Pipelines
"""

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import pandas as pd


class SKTransform(BaseEstimator, TransformerMixin):
    """Sklearn Custom Transformer Decorator"""

    def __init__(self, f):
        self.transform_func = f

    def __call__(self, X):
        return self.transform_func(X)

    def __iter__(self):
        return (i for i in [self.transform_func.__name__, self])

    def __getitem__(self, i):
        return [self.transform_func.__name__, self][i]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return self.transform_func(X.values)
        return self.transform_func(X)


class SKClassify(BaseEstimator, ClassifierMixin):
    """Sklearn Custom Classifier Decorator"""

    def __init__(self, f):
        self.predict_func = f

    def __call__(self, X):
        return self.predict_func(X)

    def __iter__(self):
        return (i for i in [self.predict_func.__name__, self])

    def __getitem__(self, i):
        return [self.predict_func.__name__, self][i]

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def predict(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return self.predict_func(X.values)
        return self.predict_func(X)


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    import numpy as np

    @SKTransform
    def power2(x):
        return x**2

    @SKClassify
    def lessThan50(x):
        return x < 50

    ppl = Pipeline([
        power2,
        lessThan50,
    ])
    print('Prediction:\n', ppl.predict(np.array([3, 6, 8, 10])))
