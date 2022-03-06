# -*- coding: utf-8 -*-
# @Author: Juan Quintana
# @Date:   2018-09-26 10:01:02
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-26 16:04:15

"""
XGBOOST Regressor with Bayesian tuning: OPTION 2

This option is using directly the XGBoost regressor. For this reason is possible
use scikit-learn tools to scoring.

This one is faster (3 times faster) but get a little bit worse performance.

Reference:
- https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt

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
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error

""" OBJECTVE FUNCTION TO BE OPTIMIZED """


def objective(params):
    params['learning_rate'] = float(params['learning_rate'])
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = float(params['min_child_weight'])
    params['subsample'] = float(params['subsample'])
    params['gamma'] = float(params['gamma'])
    params['colsample_bytree'] = float(params['colsample_bytree'])
    params['n_estimators'] = int(params['n_estimators'])
    params['reg_alpha'] = float(params['reg_alpha'])
    params['reg_lambda'] = float(params['reg_lambda'])
    params['objective'] = params['objective']
    params['eval_metric'] = params['eval_metric']
    params['nthread'] = params['nthread']
    params['booster'] = params['booster']
    params['tree_method'] = params['tree_method']
    params['silent'] = params['silent']

    global X, y, best

    clf = xgb.XGBRegressor(
        n_jobs=2,
        **params
    )

    y_hat = cross_val_predict(clf, X, y, method='predict', cv=5)
    score = metrics_regression(y, y_hat)['mae']
    print("############### Score: {0}".format(score))
    print("############### Prms: ", params)
    print('..........................')

    return score


""" PARAMETERS SPACE """

xgb_space = {
    'learning_rate': hp.quniform('eta', 0.005, 0.05, 0.005),
    'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.0, 1, 0.01),  # 0.0 o 0.5
    'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1.0, 0.05),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'reg_alpha':  hp.quniform('alpha', 0, 10, 1),
    'reg_lambda': hp.quniform('lambda', 1, 2, 0.1),
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'nthread': -1,
    'booster': 'gbtree',
    'tree_method': 'exact',  # 'auto'
    'silent': 1
}


if __name__ == '__main__':
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

    trials = Trials()
    best = fmin(objective,
                space=xgb_space,
                algo=tpe.suggest,
                max_evals=250,
                trials=trials)
    print(best)

    # launch prediction with this parameters
    from xgboost.sklearn import XGBRegressor
    clf = XGBRegressor(
        learning_rate=float(best['eta']),
        max_depth=int(best['max_depth']),
        min_child_weight=float(best['min_child_weight']),
        subsample=float(best['subsample']),
        gamma=float(best['gamma']),
        colsample_bytree=float(best['colsample_bytree']),
        n_estimators=int(best['n_estimators']),
        reg_alpha=float(best['alpha']),
        reg_lambda=float(best['lambda']),
        objective='reg:linear',
        eval_metric='mae',
        nthread=-1,
        booster='gbtree',
        tree_method='exact',
        silent=1
    )
    # test
    clf.fit(X, y)
    y_hat = clf.predict(X_test)
    dscores = metrics_regression(y_test, y_hat, X.shape[1])
    tf = t.since('test')
    print('\nBayesian tuning - test:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' %
          (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))
    # training
    y_hat = clf.predict(X)
    dscores = metrics_regression(y, y_hat, X.shape[1])
    print('Bayesian tuning - train:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' %
          (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))
    # validation
    y_hat = cross_val_predict(clf, X, y, method='predict', cv=5)
    dscores = metrics_regression(y, y_hat, X.shape[1])
    print('Bayesian tuning - val:  bias = %.3f  mae = %.3f  r2 = %.3f (time: %s)' %
          (dscores['bias'], dscores['mae'], dscores['r2'], format_duration(tf)))
