# -*- coding: utf-8 -*-
# @Author: Juan Quintana
# @Date:   2018-09-26 10:01:02
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-26 16:04:24

"""
XGBOOST Regressor with Bayesian tuning: OPTION 3

In this case it will be used hyperopt-sklearn and his native algorithm
"xgboost_regression".

NOTE: scikit-learn tools is not working for this estimator.

Reference: https://github.com/hyperopt/hyperopt-sklearn
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
sys.path.append('../../')
from datasets import solar
from tools.reader import get_dcol
from preprocessing.scalers.normalization import Scaler
from models.metrics import metrics_regression
from tools.timer import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
import os
os.environ['OMP_NUM_THREADS'] = str(2)


def main():
    # init timer
    t = Timer()
    t.add('test')

    """ DATA PREPARATION """

    # load data
    data, dcol = solar.load()
    # select data
    ly = ['y']
    lx = ['doy', 'hour', 'LCDC267', 'MCDC267', 'HCDC267', 'TCDC267', 'logAPCP267', 'RH267', 'TMP267', 'DSWRF267']
    data = data[lx + ly]
    dcol = get_dcol(data, ltarget=ly)
    # select one hour data
    hour = 11
    idata = data[data.hour == hour]
    idata.drop('hour', axis=1, inplace=True)
    idcol = get_dcol(idata, ltarget=['y'])
    # clean
    del(data)
    del(dcol)

    # filtering outliers (ghi vs power)
    from preprocessing.outliers import median2D
    isoutlier = median2D.launch(idata['DSWRF267'].values, idata.y.values, percent=20.)
    idata['isoutlier'] = isoutlier
    idata = idata[idata.isoutlier == False]
    idata.drop('isoutlier', axis=1, inplace=True)

    # prepare data
    X = idata[idcol['lx']].values
    scaler = Scaler()
    y = scaler.fit_transform(idata[idcol['ly']].values).ravel()
    print('Prepared data: X: %s  y: %s' % (X.shape, y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('Prepared data: X_train: %s  y_train: %s' % (X_train.shape, y_train.shape))
    print('Prepared data: X_test: %s  y_test: %s' % (X_test.shape, y_test.shape))
    # replace training dataset
    X = X_train
    y = y_train

    """ ESTIMATOR WITH BAYESIAN TUNING """

    from hpsklearn import HyperoptEstimator, xgboost_regression
    from hyperopt import tpe
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    clf = HyperoptEstimator(regressor=xgboost_regression('my_clf'),
                            preprocessing=[],
                            algo=tpe.suggest,
                            max_evals=250,
                            trial_timeout=300)

    clf.fit(X, y)
    print(clf.best_model())
    y_hat = clf.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('test')
    print('\nBayesian tuning -test:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' %
          (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))
    # training
    y_hat = clf.predict(X)
    dscores = metrics_regression(y, y_hat, X.shape[1])
    print('Bayesian tuning - train:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' %
          (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))


if __name__ == '__main__':
    main()
