# -*- coding: utf-8 -*-
# @Author: jmquintana79
# @Date:   2018-09-22 11:58:53
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-26 16:12:20

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
sys.path.append('../')
from datasets import solar
from tools.reader import get_dcol
from tools.timer import *
from preprocessing.scalers.normalization import Scaler
from models.metrics import metrics_regression
# from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def main():
    # init timer
    t = Timer()

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

    """ ESTIMATOR WITHOUT TUNING """

    t.add('no_tuning')
    clf = XGBRegressor(nthreads=-1)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('no_tuning')
    print('Without tuning:     bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' % (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))

    """ ESTIMATOR WITH RANDOM TUNING """

    t.add('random_tuning')
    clf = XGBRegressor(nthreads=-1)
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    dparams = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }
    gs = RandomizedSearchCV(clf, dparams, cv=5, n_jobs=1, scoring='r2')
    gs.fit(X_train, y_train)
    y_hat = gs.best_estimator_.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('random_tuning')
    print('Random tuning:      bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' % (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))

    """ ESTIMATOR WITH EXHAUSTIVE TUNING """

    t.add('exhaustive_tuning')
    clf = XGBRegressor(nthreads=-1)
    dparams = {
        "n_estimators": [3, 10, 25, 40],
        "max_depth": [3, 10, 25, 40],
        "learning_rate": [0.05, 0.1, 0.25, 0.5],
        "gamma": np.arange(0, 11, 1),
    }
    gs = GridSearchCV(clf, param_grid=dparams, cv=5, n_jobs=1, scoring='r2')
    gs.fit(X_train, y_train)
    y_hat = gs.best_estimator_.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('exhaustive_tuning')
    print('Exhaustive tuning:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' % (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))

    """ ESTIMATOR WITH BAYESIAN TUNING """

    from hpsklearn import HyperoptEstimator, xgboost_regression
    from hyperopt import tpe
    import os
    os.environ['OMP_NUM_THREADS'] = str(2)
    t.add('bayesian_tuning')
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    clf = HyperoptEstimator(regressor=xgboost_regression('my_clf'),
                            preprocessing=[],
                            algo=tpe.suggest,
                            max_evals=250,
                            trial_timeout=300)

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('bayesian_tuning')
    print('Bayesian tuning:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' % (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))


if __name__ == '__main__':
    main()
