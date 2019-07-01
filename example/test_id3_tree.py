#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train and test data reference: 
https://www.python-course.eu/Decision_Trees.php
http://archive.ics.uci.edu/ml/datasets/zoo
"""
import sys
import numpy as np

sys.path.append('.')
from mle.supervised_learning.decision_tree import ID3ClassificationTree


import pandas as pd

# Import the dataset and define the feature as well as the target datasets / columns#
# Import all columns omitting the fist which consists the names of the animals
dataset = pd.read_csv('data/uci/zoo/zoo.data',
                      names=['animal_name', 'hair', 'feathers', 'eggs', 'milk',
                             'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
                             'breathes', 'venomous', 'fins', 'legs', 'tail',
                             'domestic', 'catsize', 'class', ])


dataset = dataset.drop('animal_name', axis=1)


def train_test_split(dataset):
    # We drop the index respectively relabel the index
    training_data = dataset.iloc[:80].reset_index(drop=True)
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


train_features = dataset.iloc[:80, :-1].values
test_features = dataset.iloc[80:, :-1].values
train_targets = dataset.iloc[:80, -1].values
test_targets = dataset.iloc[80:, -1].values
# print(train_features)
# print(train_targets)
# print(train_targets.tolist(), len(train_targets.tolist()))
id3 = ID3ClassificationTree(decision_type='is', min_data_in_leaf=0)
id3.fit(train_features, train_targets)
id3.print_tree()
# print (id3.root.feature_i, id3.root.threshold, id3.root.leaf_value, id3.root.child_trees)

predicted = id3.predict(test_features)
print (predicted, len(predicted))
print (test_targets, len(test_targets))
print (float(np.sum(predicted == test_targets)) / len(predicted))