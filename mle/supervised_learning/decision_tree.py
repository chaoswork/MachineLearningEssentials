#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decision Tree
reference: https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/
supervised_learning/decision_tree.py
"""
import numpy as np
from ..utils import divide_on_feature
from ..utils import split_func


class DecisionNode(object):

    """
    Decision Node or Leaf Node
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_i=None, threshold=None, leaf_value=None,
                 true_branch=None, false_branch=None):
        assert threshold and leaf_value, "Leaf or Decision, choose one"
        # Decision Node
        self.feature_i = feature_i
        self.threshold = threshold
        self.true_branch = true_branch
        self.false_branch = false_branch

        # Leaf Node
        self.leaf_value = leaf_value


class DecisionTree(object):

    """
    A super class for CART(Classification And Regression Tree)
    Parameters:
    -----------
    min_data_in_leaf: int
        minimal number of data in one leaf. Can be used to deal with over-fitting
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree. max_depth <= 0 means no limit
    num_leaves: int
        max number of leaves in one tree
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, min_data_in_leaf=20, min_impurity=1e-7, max_depth=-1,
                 num_leaves=31, loss=None):
        self.root = None
        self.min_data_in_leaf = min_data_in_leaf
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss
        self._impurity_calc_func = None  # 分裂指标计算
        self._leaf_value_calc_func = None

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0    #
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # shape (n, ) -> (n, 1)
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        # TODO: add num_leaves limit, maybe another parameter: level-wise/leaf-wise(Best-First)
        # https://lightgbm.readthedocs.io/en/latest/Features.html#references
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.2862&rep=rep1&type=pdf
        if n_samples >= self.min_data_in_leaf and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # calculate the impurity
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1) and len(Xy2):
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        _impurity = self._impurity_calc_func(y, y1, y2)

                        if largest_impurity < _impurity:
                            largest_impurity = _impurity
                            best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                            best_sets = {
                                'leftX': Xy1[:, :n_features],
                                'lefty': Xy1[:, n_features:],
                                'rightX': Xy2[:, :n_features],
                                'righty': Xy2[:, n_features:]
                            }
        # Build left and right tree
        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(
                best_sets['leftX'], best_sets['lefty'], current_depth + 1)
            false_branch = self._build_tree(
                best_sets['rightX'], best_sets['righty'], current_depth + 1)

            return DecisionNode(feature_i=best_criteria['feature_i'],
                                threshold=best_criteria['threshold'],
                                true_branch=true_branch,
                                false_branch=false_branch)

        # Now this is a leaf node
        leaf_value = self._leaf_value_calc_func(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # if leaf, return value
        if tree.leaf_value:
            return tree.leaf_value

        # now this node is a decision node
        branch = tree.false_branch
        feature_value = x[tree.feature_i]
        if split_func(feature_value, tree.threshold):
            branch = tree.true_branch
        # recursive search
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(x) for x in X]
        return y_pred





