# -*- coding: <utf-8> -*-

import networkx as nx
import numpy as np

from graph_util import compute_weight


def laplacian_eigen_vectors(graph, attribute, operation):
    compute_weight(graph, attribute, operation)
    return np.linalg.eig(
                nx.normalized_laplacian_matrix(
                    graph,
                    weight='weight'
                ).todense()
            )[1]


def laplacian_features(graphs, attribute, operation):
    return None
