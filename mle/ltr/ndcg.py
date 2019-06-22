#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

"""
reference: https://github.com/jma127/pyltr/blob/master/pyltr/metrics/dcg.py
"""


class DCG(object):

    """
    Discounted Cumulative Gain, ranks are 0-based
    DCG_{k}=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{log_{2}(i+1)}, ranks are 1-based here
    rel_i the graded relevance of the result at position i
    《A Theoretical Analysis of Normalized Discounted Cumulative Gain (NDCG) Ranking Measures》
    """

    def __init__(self, k,
                 gain_fn=lambda x: math.pow(2, x) - 1,
                 rank_discount_fn=lambda rank: math.log(rank + 2, 2)  # rank begin with 0
                 ):
        """init function"""
        self.k = k
        self._gain_fn = gain_fn
        self._rank_discount_fn = rank_discount_fn
        self._discounts = self._make_discounts(256)

    def evaluate(self, targets):
        return sum(self._gain_fn(t) * self._get_discount(i)
                   for i, t in enumerate(targets) if i < self.k)

    def _make_discounts(self, n):
        return [1.0 / self._rank_discount_fn(i) for i in range(n)]

    def _get_discount(self, i):
        if i >= self.k:
            return 0.0
        while i >= len(self._discounts):
            self._grow_discounts()
        return self._discounts[i]

    def _grow_discounts(self):
        self._discounts = self._make_discounts(len(self._discounts) * 2)


class NDCG(object):

    """
    Normalized Discounted Cumulative Gain, ranks are 0-based
    NDCG_{k}=\frac{DCG_{k}}{IDCG_{k}}
    rel_i the graded relevance of the result at position i
    IDCG is ideal discounted cumulative gain
    IDCG_{k}=\sum_{i=1}^{|REL|}\frac{2^{rel_i}-1}{log_{2}(i+1)}, |REL| represents the list of
    relevant documents (ordered by their relevance) in the corpus up to position p
    """

    def __init__(self, k,
                 gain_fn=lambda x: math.pow(2, x) - 1,
                 rank_discount_fn=lambda rank: math.log(rank + 2, 2)  # rank begin with 0
                 ):
        """init function"""
        self.k = k
        self._dcg = DCG(k=k, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

    def evaluate(self, targets, all_rel=None):
        if all_rel is None:
            all_rel = targets
        return self._dcg.evaluate(targets) / max(1e-6, self._inverse_max_dcg(all_rel))

    def _inverse_max_dcg(self, targets):
        sorted_targets = sorted(targets, reverse=True)
        return self._dcg.evaluate(sorted_targets)


if __name__ == "__main__":
    check_list = [3, 2, 3, 0, 1, 2]
    k = len(check_list)
    dcg = DCG(k, gain_fn=lambda x: x)
    ndcg = NDCG(k, gain_fn=lambda x: x)
    print dcg.evaluate(check_list)
    print ndcg.evaluate(check_list, all_rel=check_list + [3, 2])

