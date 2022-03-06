# -*- coding: utf-8 -*-
# @Author: Juan Quintana
# @Date:   2018-09-11 13:38:00
# @Last Modified by:   Juan Quintana
# @Last Modified time: 2018-09-12 11:57:03

"""
Basic scikit-learn pipeline to prepare the features according
type of each column to be used in a ML algorithm:
 - boolean: no transformation.
 - numerical (int, float): standarization.
 - categorical: Hot Encoder (get dummies).
"""

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from pipelines.custom import *
import numpy as np

# only for numerical and boolean data
num_pipeline = Pipeline([
    ('features', FeatureUnion(n_jobs=-1, transformer_list=[
        # bool
        ('boolean', Pipeline([('selector', TypeSelector('bool'))])),
        # numericals
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
        ])),
    ])),
])


# only for categorical data
cat_pipeline = Pipeline([
    ('selector', TypeSelector('category')),
    ('encoder', MyOneHotEncoder()),
])


# for numerical and categorical data
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
