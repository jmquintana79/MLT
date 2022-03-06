# -*- coding: utf-8 -*-
# @Author: jmquintana79
# @Date:   2018-08-30 23:54:47
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-04 10:55:22

"""
Regression and Classification frameworks based on scikit-learn models with extra funcionalities.
"""
import sys
sys.path.append('../')
from backend.decorators import valida
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from models.metrics import metrics_regression


class Base():
    """
    Model class for scikit-learn estimators with the common funcionalities between regressors and classificators.
    """

    def __init__(self, estimator):
        """
        Create object.
        estimator -- scikit-learn estimator.
        """
        # set attributes
        self.clf = estimator
        self.name = estimator.__class__.__name__
        self.type = estimator._estimator_type
        # display estimator params
        print('[info] Create %s(%s):' % (self.name, self.type), self.get_params())

    def fit(self, X: 'array', y: 'array'):
        """
        Fit an estimator.
        X -- features data from training dataset.
        y -- target data from training dataset.
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # set as attributes
        self.X_ = X
        self.y_ = y
        # fit the model
        print('[info] Fitting...')
        self.clf.fit(X, y)

    def predict(self, X: 'array')->'array':
        """
        Predict using a fitted estimator.
        X -- features data from testing dataset.
        return -- prediction.
        """

        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # prediction
        return self.clf.predict(X)

    def valida(self, X: 'array', y: 'array')->dict:
        """
        Calculate metrics of a fitted estimator.
        X -- features data from testing dataset.
        y -- target data from testing dataset.
        return -- dictionary of metrics.
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # input validation
        X = check_array(X)
        # prediction
        y_hat = self.clf.predict(X)
        # calculate scores
        print('[info] Scoring...')
        return self.metrics(y, y_hat)

    def scores_cv(self, X: 'array', y: 'array', cv: int=5, njobs: int=-1, scoring: str=''):
        """
        Calculate model score to evaluate model performance using cv.
        X -- features data of whole dataset.
        y -- target data of whole dataset.
        cv -- number of folds to divide the whole period for recursive train / validation (default 5).
        njobs -- the number of CPUs to use to do the computation. -1 means 'all CPUs' (default -1).
        scoring -- scikit-learn score to be used (default ''=use default scorer according type of estimator).
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # input validation
        X = check_array(X)
        # display
        print('[info] CV scoring...')

        # fit grid
        if scoring == '':
            scoring = self.metric_cv()
        # cv score
        scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=njobs, scoring=scoring)
        # display
        print('[info] cv scores(%s): %.3f +/- %.3f ' % (scoring, np.mean(scores), np.std(scores)))

    def metrics_train_cv(self, X: 'array', y: 'array', cv: int=5, njobs: int=-1, method: str='predict')->dict:
        """
        Calculate train / cv metrics of a fitted model.
        X -- features data of whole dataset.
        y -- target data of whole dataset.
        cv -- number of folds to divide the whole period for recursive train / validation (default 5).
        njobs -- the number of CPUs to use to do the computation. -1 means 'all CPUs' (default -1).
        method -- invokes the passed method name of the passed estimator. For method=’predict_proba’,
        the columns correspond to the classes in sorted order (default 'predict').
        return -- dictionary with the train and cv metrics.
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # input validation
        X = check_array(X)
        # calculate train prediction
        y_hat_train = self.clf.fit(X, y).predict(X)
        # calculate cv prediction
        print('[info] Calculating train/cv metrics...')
        y_hat_cv = cross_val_predict(self.clf, X, y, cv=cv, n_jobs=njobs, method=method)
        # calculate scores
        dmetrics_train = self.metrics(y, y_hat_train, 'train')
        dmetrics_cv = self.metrics(y, y_hat_cv, 'cv')
        # return
        return {'train': dmetrics_train, 'cv': dmetrics_cv}

    def fit_exhaustive_tuning(self, X: 'array', y: 'array', dparam_grid: dict, cv: int=5, njobs: int=-1):
        """
        Fit an estimator with exhaustive tuning.
        X -- features data of whole dataset.
        y -- target data of whole dataset.
        dparam_grid -- dictionary of parameters to be tuned and theirs possible values to be tested.
        cv -- number of folds to divide the whole period for recursive train / validation (default 5).
        njobs -- the number of CPUs to use to do the computation. -1 means 'all CPUs' (default -1).
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # set as attributes
        self.X_ = X
        self.y_ = y
        # launch tuning
        print('[info] Fitting with exhauestive tuning...')
        gs = self._fit_gridsearchcv(X, y, dparam_grid, cv=cv, njobs=njobs)
        # results tuning
        best_score = abs(gs.best_score_) if 'neg' in self.metric_cv() else gs.best_score_
        print('[info] best score (%s): %s' % (self.metric_cv(), best_score))
        print('[info] best hyperparams:')
        for k, v in gs.best_params_.items():
            print('\t-"%s":' % k, v)
        # set best estimator
        self.clf = gs.best_estimator_

    def scores_cv_exhaustive_tuning(self, X: 'array', y: 'array', dparam_grid: dict, cv: int=5, njobs: int=-1, scoring: str=''):
        """
        Calculate the tuned model score to evaluate model performance.
        X -- features data of whole dataset.
        y -- target data of whole dataset.
        dparam_grid -- dictionary of parameters to be tuned and theirs possible values to be tested.
        cv -- number of folds to divide the whole period for recursive train / validation (default 5).
        njobs -- the number of CPUs to use to do the computation. -1 means 'all CPUs' (default -1).
        scoring -- scikit-learn score to be used (default ''=use default scorer according type of estimator).
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # input validation
        X = check_array(X)
        # display
        print('[info] CV scoring with exhaustive tuning...')

        # fit grid
        if scoring == '':
            scoring = self.metric_cv()
        gs = GridSearchCV(estimator=self.clf, param_grid=dparam_grid, cv=2, n_jobs=njobs, scoring=scoring)
        # cv score
        scores = cross_val_score(gs, X, y, cv=cv, n_jobs=njobs, scoring=scoring)
        # display
        print('[info] cv scores(%s): %.3f +/- %.3f ' % (scoring, np.mean(scores), np.std(scores)))

    # GridSearchCV fitting
    def _fit_gridsearchcv(self, X: 'array', y: 'array', dparam_grid: dict, cv: int=2, njobs: int=-1)->'object':
        """
        GridSearchCV fitting.
        X -- features data of whole dataset.
        y -- target data of whole dataset.
        dparam_grid -- dictionary of parameters to be tuned and theirs possible values to be tested.
        cv -- number of folds to divide the whole period for recursive train / validation (default 5).
        njobs -- the number of CPUs to use to do the computation. -1 means 'all CPUs' (default -1).
        scoring -- scikit-learn score to be used (default ''=use default scorer according type of estimator).
        return -- a fitted GridSearchCV object.
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # set as attributes
        self.X_ = X
        self.y_ = y
        # fit grid
        gs = GridSearchCV(estimator=self.clf, param_grid=dparam_grid, cv=cv, n_jobs=njobs, scoring=self.metric_cv())
        gs.fit(X, y)
        # return fitted gridsearchcv object
        return gs

    def get_params(self, deep: bool=True):
        """
        Return parameters of the estimator.
        """
        return self.clf.get_params()

    def set_params(self, **parameters):
        """
        Set parameters of the estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Regressor(Base):
    """
    Model class with especific funcionalities for regressors.
    """

    def metrics(self, y: 'array', y_hat: 'array', msg: str='')->dict:
        """
        Calculate basic metrics for regressors.
        y -- real data.
        y_hat -- predicted data.
        msg -- word or sentence to be included in the screen message.
        return -- dictionary of metrics.
        """
        dmetrics = metrics_regression(y, y_hat, self.X_.shape[1])
        print('[info] Metrics(%s): bias = %.3f  mae = %.3f   r2 = %.3f' % (msg, dmetrics['bias'], dmetrics['mae'], dmetrics['r2']))
        return dmetrics

    def metric_cv(self, scoring: str='neg_mean_absolute_error')->str:
        """
        Select the scikit-learn score to be used for regressors.
        scoring -- scikit-learn score to be used for regressors (default 'neg_mean_absolute_error').
        return -- the scikit-learn score to be used for regressors.
        """
        return scoring


class Classificator(Base):
    """
    Model class with especific funcionalities for classificators.
    """

    def metrics(self, y: 'array', y_hat: 'array', msg: str='')->dict:
        """
        Calculate basic metrics for regressors.
        y -- real data.
        y_hat -- predicted data.
        msg -- word or sentence to be included in the screen message.
        return -- dictionary of metrics.
        """
        return None

    def metric_cv(self, scoring: str='accuracy'):
        """
        Select the scikit-learn score to be used for classificator.
        scoring -- scikit-learn score to be used for classificator (default 'neg_mean_absolute_error').
        return -- the scikit-learn score to be used for classificator.
        """
        return scoring


if __name__ == '__main__':
    import sys
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # load dataset
    sys.path.append('../')
    from tools.datasets import *
    dataset = Data()
    dfdata, dcol = dataset.load('boston')
    X = dfdata[dcol['lx']].values
    y = dfdata[dcol['ly']].values

    # select estimator object
    estimator = LinearRegression()

    # SCORE ESTIMATOR (for model selection) #

    # calculate cv scores WITHOUT exhaustive tuning
    clf = Regressor(estimator)
    clf.scores_cv(X, y, scoring='r2')
    # calculate cv scores WITH exhaustive tuning
    clf = Regressor(estimator)
    dparams = {'fit_intercept': [True, False], 'normalize': [True, False]}
    clf.scores_cv_exhaustive_tuning(X, y, dparams, scoring='r2')

    # TRAIN/CV METRICS OF ESTIMATOR (for settings options selection) #

    # calculate train/cv metrics WITHOUT exhaustive tuning
    clf = Regressor(estimator)
    clf.fit(X, y)
    clf.metrics_train_cv(X, y, cv=5, njobs=-1, method='predict')
    # calculate train/cv metrics WITH exhaustive tuning
    clf = Regressor(estimator)
    dparams = {'fit_intercept': [True, False], 'normalize': [True, False]}
    clf.fit_exhaustive_tuning(X, y, dparams)
    clf.metrics_train_cv(X, y, cv=5, njobs=-1, method='predict')
