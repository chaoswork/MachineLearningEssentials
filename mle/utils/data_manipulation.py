#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def split_func(x, threshold):
    assert type(x) == type(threshold), "feature and threshold type dismatch"
    if isinstance(threshold, int) or isinstance(threshold, float):
        return x >= threshold
    return x == threshold


def divide_on_feature(X, feature_i, threshold):

    X_true = np.array([x for x in X if split_func(x[feature_i], threshold)])
    X_false = np.array([x for x in X if not split_func(x[feature_i], threshold)])

    return np.array([X_true, X_false])
