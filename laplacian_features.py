# -*- coding: <utf-8> -*-

import networkx as nx
import numpy as np

from graph_util import compute_weight

def laplacian_eigen_vectors(graph, attribute, operation):
    compute_weight(graph, attribute, operation)
    return np.linalg.eig(nx.laplacian_matrix(graph).todense())[1]
