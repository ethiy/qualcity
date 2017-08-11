# -*- coding: <utf-8> -*-

import numpy as np
import networkx as nx

from graph import compute_weight

class DirectProductKernel(object):
    """
    Direct Product graph kernel.
    """
    def __init__(self, g1, g2):
        super(DirectProductKernel, self).__init__()
        self.g1 = g1
        self.g2 = g2

    def compare(self, attribute):
        product = nx.cartesian_product(self.g1, self.g2)
        compute_weight(product, attribute)
        return None
