#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def calculate_variance(X, axis=0, ddof=0):
    """
    can use np.var(X, axis=axis, ddof=ddof)
    ddof: Delta Degrees of Freedom, 贝塞尔修正,
    https://dfrieds.com/math/bessels-correction
    """
    mean = np.ones(np.shape(X)) * X.mean(axis)
    n_samples = np.shape(X)[axis]
    n_samples -= ddof
    return (np.sum((X - mean) * (X - mean), axis=axis) / n_samples)
