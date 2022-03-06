# -*- coding: utf-8 -*-
# @Author: jmquintana79
# @Date:   2018-09-19 20:08:12
# @Last Modified by:   jmquintana79
# @Last Modified time: 2018-09-19 21:01:42

"""
Basic Linear Regressors.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
sys.path.append('../')
from models.metrics import metrics_regression
import warnings
warnings.filterwarnings('ignore')



class LRFormula(BaseEstimator, RegressorMixin):
    """
    Simple Linear Regression by formula
    """

    def __init__(self, formula:str, colnames_X:list , colname_y:str):
        """
        Initializing the regressor
        """
        self.formula = formula
        self.colnames_X = colnames_X
        self.colname_y = colname_y

    def fit(self, X, y):
        """
        Fit the model.
        """
        # validation: check that X and y have correct shape
        X, y = check_X_y(X, y)
        # set as attributes
        self.X_ = X
        self.y_ = y
        # validation: check that X and colnames_X have correct shape
        assert (len(self.colnames_X) == X.shape[1]), "colnames no match with the data shape."
        # data to df
        dfxy = DataFrame(np.c_[X, y], columns = self.colnames_X + [self.colname_y])
        # fit the model
        try:
            self.linreg = smf.ols(formula=self.formula, data=dfxy).fit()
        except Exception as e:
            print(str(e))
        # return
        return self


    def predict(self, X, y=None):
        """
        Calculate prediction.
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # data to df
        dfx = DataFrame(X, columns = self.colnames_X)
        # prediction
        return self.linreg.predict(dfx).values.ravel()

    def score(self, X, y=None):
        # validation: check that X and y have correct shape
        X, y = check_X_y(X, y)
        # calculate and retur
        return r2_score(y, self.predict(X))

    def metrics(self, X, y):
        # validation: check that X and y have correct shape
        X, y = check_X_y(X, y)
        # calculate and return metrics
        return metrics_regression(y, self.predict(X))

    def summary(self):
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # display summary
        print(self.linreg.summary())
        return None


if __name__ == '__main__':
    from datasets import solar
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    # load data
    data, dcol = solar.load()
    # rename columns
    data.rename(columns = {'TMP267':'tmp', 'DSWRF267':'ghi', 'y':'power'}, inplace=True)
    # filter rows
    data = data.dropna()
    # collect per hour
    ihour = 11
    hourly = data[data.hour==ihour]
    # data preparation
    hourly['lnghi'] = np.log(hourly['ghi'].values)

    # normalization
    scaler = StandardScaler()
    ihourly = scaler.fit_transform(hourly[['power','ghi','tmp', 'lnghi']])
    idata = pd.DataFrame(ihourly, columns = ['power','ghi','tmp', 'lnghi'])

    ## simulation
    colnames_X = ['ghi','tmp', 'lnghi']
    colname_y = 'power'
    formula = 'power ~ (ghi + lnghi) * (tmp + ghi)'
    X = idata[colnames_X]
    y = idata[colname_y].values
    # create regressor object
    clf = LRFormula(formula, colnames_X, colname_y)
    clf.fit(X,y)
    print('bias: %s'%clf.score(X,y))
    # cross validation score
    clf = LRFormula(formula, colnames_X, colname_y)
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring='r2')
    print('bias: avg = %.3f   std = %.3f'%(np.mean(scores), np.std(scores)))

    quit('bye')
