#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decision Tree
reference:
1. https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/
   supervised_learning/decision_tree.py

2. https://github.com/rasbt/python-machine-learning-book/blob/master/faq/decision-tree-binary.md
For practical reasons (combinatorial explosion) most libraries implement decision trees
with binary splits. The nice thing is that they are NP-complete
(Hyafil, Laurent, and Ronald L. Rivest. "Constructing optimal binary decision trees is
NP-complete." Information Processing Letters 5.1 (1976): 15-17.)
"""
import math
import numpy as np
from ..utils import calculate_variance
from ..utils import calculate_entropy
from ..utils import get_element_count
from ..utils import calculate_info_gain


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
    child_trees: list of DecisionNode
        Next decision node for samples where features match
    """

    def __init__(self, feature_i=None, threshold=None, leaf_value=None,
                 child_trees=None, decision_type='no_lesser'):
        # Decision Node
        self.feature_i = feature_i
        self.threshold = threshold
        self.decision_type = decision_type
        self.child_trees = child_trees

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

    def __init__(self, decision_type='is', min_data_in_leaf=20, min_impurity=1e-7, max_depth=-1,
                 num_leaves=31, loss=None):
        self.decision_type = decision_type  # no_lesser/is
        self.root = None
        self.min_data_in_leaf = min_data_in_leaf
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss
        self._impurity_calc_func = None  # 不纯度
        self._leaf_value_calc_func = None
        self._feature_split_iter = None  # 特征分裂的迭代器

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None

    def _get_child_tree(self, x, threshold):
        if threshold is None:
            return x
        if self.decision_type == 'no_lesser':
            return x >= threshold
        return x == threshold

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0    #
        best_criteria = None    # Feature index and threshold
        best_sets_X = None
        best_sets_y = None

        # shape (n, ) -> (n, 1)
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        n_samples, n_features = np.shape(X)
        # TODO: add num_leaves limit, maybe another parameter: level-wise/leaf-wise(Best-First)
        # https://lightgbm.readthedocs.io/en/latest/Features.html#references
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.2862&rep=rep1&type=pdf
        if n_samples >= self.min_data_in_leaf and (
                self.max_depth == -1 or current_depth <= self.max_depth):
            # Calculate the impurity for each feature
            for (X_split_list, y_split_list, split_strategy) in self._feature_split_iter(X, y):
                _impurity = self._impurity_calc_func(y, y_split_list)
                # print ('impurity = ', _impurity)
                if largest_impurity < _impurity:
                    largest_impurity = _impurity
                    best_criteria = split_strategy
                    best_sets_X = X_split_list
                    best_sets_y = y_split_list

        # Build child tree
        if largest_impurity > self.min_impurity:
            sub_tree_dict = {}
            for i in range(len(best_sets_X)):
                sub_tree = self._build_tree(
                    best_sets_X[i], best_sets_y[i], current_depth + 1)
                child_key = self._get_child_tree(best_sets_X[i][0][best_criteria['feature_i']],
                                                 best_criteria['threshold'])

                sub_tree_dict[child_key] = (sub_tree, i)  # index for print tree

            return DecisionNode(feature_i=best_criteria['feature_i'],
                                threshold=best_criteria['threshold'],
                                decision_type=self.decision_type,
                                child_trees=sub_tree_dict)

        # Now this is a leaf node
        leaf_value = self._leaf_value_calc_func(y)
        return DecisionNode(leaf_value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # if leaf, return value
        if tree.leaf_value:
            return tree.leaf_value

        # now this node is a decision node
        child_key = self._get_child_tree(x[tree.feature_i], tree.threshold)
        if child_key not in tree.child_trees:
            return None
        branch = tree.child_trees[child_key][0]
        # recursive search
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(x) for x in X]
        return y_pred

    def __str__(self, tree=None, indent=""):
        """ Recursively print the decision tree """
        trees_str = ""
        if not tree:
            tree = self.root
            trees_str += "def predict(feature):\n"
            indent = "    "

        # If we're at leaf => print the label
        if tree.leaf_value is not None:
            return "\n{}return {}".format(indent, tree.leaf_value)
        # Go deeper down the tree
        else:
            if self.decision_type == 'no_lesser':
                operator = '>='
            elif self.decision_type == 'is':
                operator = '=='
            for (child_key, (tree_branch, index)) in sorted(
                    tree.child_trees.items(), key=lambda x: x[1][1]):
                threshold = tree.threshold
                if threshold is None:
                    threshold = child_key

                # index = 0 -> if, others -> elif
                cond = 'elif'
                if index == 0:
                    cond = 'if'
                elif index == len(tree.child_trees) - 1:
                    if self.decision_type == 'no_lesser':
                        cond = 'else:#'

                trees_str += "\n{}{} feature[{}] {} {}:".format(
                    indent, cond, tree.feature_i, operator, threshold).split('#')[0]
                sub_tree = self.__str__(tree_branch, indent + "    ")
                trees_str += sub_tree
            return trees_str


class RegressionTree(DecisionTree):

    """
    Least Square Regression Tree
    """
    def _calculate_variance_reduction(self, y, y_split_list):
        assert len(y_split_list) == 2, "only support binary regression tree now"
        total_variance = y.shape[0] * calculate_variance(y)
        y1 = y_split_list[0]
        y2 = y_split_list[1]
        y1_variance = y1.shape[0] * calculate_variance(y1)
        y2_variance = y2.shape[0] * calculate_variance(y2)

        # is variance_reduction always bigger than zero ?
        # how to prove it ?
        variance_reduction = total_variance - (y1_variance + y2_variance)

        return variance_reduction

    def _split_iter(self, X, y):
        n_samples, n_features = np.shape(X)
        Xy = np.concatenate((X, y), axis=1)
        for feature_i in range(n_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                split_strategy = {'feature_i': feature_i, 'threshold': threshold}
                Xy_1 = np.array([x for x in Xy if x[feature_i] >= threshold])
                Xy_2 = np.array([x for x in Xy if x[feature_i] < threshold])
                if len(Xy_1) and len(Xy_2):
                    X_split_list = [Xy_1[:, :n_features], Xy_2[:, :n_features]]
                    y_split_list = [Xy_1[:, n_features:], Xy_2[:, n_features:]]

                    yield (X_split_list, y_split_list, split_strategy)

    def _mean_of_y(self, y):
        return np.mean(y, axis=0)

    def fit(self, X, y):
        self.decision_type = 'no_lesser'
        self._leaf_value_calc_func = self._mean_of_y
        self._impurity_calc_func = self._calculate_variance_reduction
        self._feature_split_iter = self._split_iter
        super(RegressionTree, self).fit(X, y)


class ID3ClassificationTree(DecisionTree):
    """
    ID3 Classification Tree by Information Gain
    """

    def _split_iter(self, X, y):
        n_samples, n_features = np.shape(X)
        Xy = np.concatenate((X, y), axis=1)
        for feature_i in range(n_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)
            X_split_list = []
            y_split_list = []
            split_strategy = {'feature_i': feature_i, 'threshold': None}
            for x_value in unique_values:
                Xy_select = np.array([x for x in Xy if x[feature_i] == x_value])
                X_split_list.append(Xy_select[:, :n_features])
                y_split_list.append(Xy_select[:, n_features:])

            yield (X_split_list, y_split_list, split_strategy)

    def _majority_vote(self, y):
        counts, y_len = get_element_count(y)
        most_common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        return most_common

    def fit(self, X, y):
        self.decision_type = 'is'
        self._leaf_value_calc_func = self._majority_vote
        self._impurity_calc_func = calculate_info_gain
        self._feature_split_iter = self._split_iter
        super(ID3ClassificationTree, self).fit(X, y)


class C45ClassificationTree(DecisionTree):
    """
    C4.5 Classification Tree by Information Gain
    """

    def _calculate_information_gain_ratio(self, y, y1, y2):
        p = len(y1) / y
        entropy = calculate_entropy(y)
        y1_entropy = calculate_entropy(y1)
        y2_entropy = calculate_entropy(y2)
        info_gain = entropy - (p * y1_entropy + (1 - p) * y2_entropy)
        info_gain_ratio = info_gain / (-p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2))
        # TODO C4.5 / ID3 is not a binary tree, so the frame need to update

        return info_gain_ratio

    def _majority_vote(self, y):
        counts, y_len = get_element_count(y)
        most_common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        return most_common

    def fit(self, X, y):
        self._leaf_value_calc_func = self._majority_vote
        self._impurity_calc_func = self._calculate_information_gain_ratio
        super(ID3ClassificationTree, self).fit(X, y)


