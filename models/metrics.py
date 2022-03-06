# -*- coding: utf-8 -*-
# @Author: jmquintana79
# @Date:   2018-08-31 00:51:44
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-08-31 11:34:44

import numpy as np


# metrics for regression model
def metrics_regression(y: 'array', yhat: 'array', k: int=None)->dict:
    """
    Metrics for regression model: BIAS, MAE, R2 / R2 adjusted. It is robust to NaN values.
    y -- array of real data (observation).
    yhat -- array of data to be scored (prediction).
    k -- number of predictors (features) for calculating R2 adjusted. If this one is None is returned R2 (default None).
    return -- dictionary of calculated metrics.
    """
    # concat arrays
    yy = np.c_[yhat, y]
    # drop rows with nan values
    yy = yy[~np.isnan(yy).any(axis=1)]
    # transfer cleaned data
    fcst = yy[:, 0]
    obs = yy[:, 1]
    # calculate general error
    error = fcst - obs
    # bias
    bias = np.mean(error)
    # mae
    mae = np.mean(np.abs(error))
    # R2 / R2 ajusted
    SS_Residual = sum((obs-fcst)**2)
    SS_Total = sum((obs-np.mean(obs))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    if k is None:
        r2 = r_squared
    else:
        adj_r_squared = 1 - (1-r_squared)*(len(obs)-1)/(len(obs)-k-1)
        r2 = adj_r_squared
    # return
    return {'bias': bias, 'mae': mae, 'r2': r2}
