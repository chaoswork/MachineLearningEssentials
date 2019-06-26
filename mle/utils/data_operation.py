#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
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


def calculate_entropy(y):
    """
    $$H(label)=-\sum_{i}^{|label|}p_{i}log_{2}(p_{i})$$
    """
    labels = y.flatten().tolist()
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0
    for label in counts:
        p = float(counts[label]) / len(labels)
        entropy += -p * math.log(p, 2)
    return entropy



