# -*- coding: utf-8 -*-
# @Author: Juan Quintana
# @Date:   2018-09-11 13:15:43
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-18 13:34:48

"""
Custom classes for scikit-learn pipelines.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Data columns selector by data type.
    dtype -- data type selector:
        float: np.float or 'float' or float
        integer: np.int or 'int' or int
        bool: np.bool or 'bool' or bool
        categorical: 'category'
        numerical: np.number
    """

    def __init__(self, dtype: 'data type'):
        self.dtype = dtype

    def fit(self, X: 'df', y: 'df'=None):
        return self

    def transform(self, X: 'df'):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class StringIndexer(BaseEstimator, TransformerMixin):
    """
    Indexation of strings. Each different string will be indexed with an unique index.
    WARNING: it can work extremely slow if it is required to index too much strings.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))


class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom OneHotEncoder where is used pandas.get_dummies function. For this reason it is
    not necessary a previous StringIndexer.
    dummy_na -- add a column to indicate NaNs, if False NaNs are ignored (default False).
    drop_first -- whether to get k-1 dummies out of k categorical levels by removing the first level (default True).
    """

    def __init__(self, dummy_na=False, drop_first=True):
        self.dummy_na = dummy_na
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return pd.get_dummies(X, dummy_na=self.dummy_na, drop_first=self.drop_first).values


class ColumnSelector(object):
    """
    A feature selector for scikit-learn's Pipeline class that returns specified columns from a numpy array.
    cols -- list of the selected index columns.
    """

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, y=None):
        return X[:, self.cols]

    def fit(self, X, y=None):
        return self


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


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from datasets import boston
    from sklearn.pipeline import Pipeline

    """ CUSTOM TRANSFORMER WITH DECORATOR """

    # load boston dataset
    data, dcol = boston.load()
    X = data[dcol['lx']].values
    print('Original data:\n', X[:5, :])
    # custom transformer: selector

    @SKTransform
    def selector(x):
        return x[:, :2]
    # custom transformer: power 2

    @SKTransform
    def power2(x):
        return x**2

    # final pipeline
    ppl = Pipeline([selector, power2])
    X_transformed = ppl.transform(X)
    print('Transformed data:\n', X_transformed[:5, :])
