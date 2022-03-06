# -*- coding: utf-8 -*-
# @Author: jmquintana79
# @Date:   2018-09-19 21:25:14
# @Last Modified by:   jmquintana79
# @Last Modified time: 2018-09-19 22:01:22

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

"""
Templates to create own models which could use Scikit-Learn tools. So, it is a wrapper
between scikit and others.
"""

class Regressor(BaseEstimator, RegressorMixin):

    def __init__(self, param1, param2):
        # attributes
        self.param1 = param1
        self.param2 = param2
        # validation
        # assert (), ""

    def fit(self, X, y=None):
        # validation: check that X and y have correct shape
        X, y = check_X_y(X, y)
        # set as attributes
        self.X_ = X
        self.y_ = y
        # validation
        # assert (), ""

        """ fit the model """

        # return
        return self


    def predict(self, X, y=None):
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)

        """ prediction and return y_hat """

        return y_hat

    def score(self, X, y=None):
        # validation: check that X and y have correct shape
        X, y = check_X_y(X, y)
        # calculate and return
        return r2_score(y, self.predict(X))
