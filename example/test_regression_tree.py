#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

sys.path.append('.')
from mle.supervised_learning.decision_tree import RegressionTree
import pandas as pd


dataset = pd.read_csv(
    "data/uci/bike/day.csv",
    usecols=['season', 'holiday', 'weekday', 'workingday', 'weathersit', 'cnt'])
print(dataset.sample(frac=1).head())


def train_test_split(dataset):
    # We drop the index respectively relabel the index
    training_data = dataset.iloc[:int(0.7 * len(dataset))].reset_index(drop=True)
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[int(0.7 * len(dataset)):].reset_index(drop=True)
    return training_data, testing_data


training_data, testing_data = train_test_split(dataset)
train_features = training_data.iloc[:, :-1].values
test_features = testing_data.iloc[:, :-1].values
train_targets = training_data.iloc[:, -1].values
test_targets = testing_data.iloc[:, -1].values
reg_tree = RegressionTree(min_data_in_leaf=0)
reg_tree.fit(train_features, train_targets)
print(reg_tree)
