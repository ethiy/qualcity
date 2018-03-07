# -*- coding: <utf-8> -*-

import networkx as nx
import numpy as np

from qualcity.graph_util import compute_weight


def laplacian_eigen_vectors(graph, attribute, operation):
    compute_weight(graph, attribute, operation)
    return np.linalg.eig(
                nx.normalized_laplacian_matrix(
                    graph,
                    weight='weight'
                ).todense()
            )[1]


def laplacian_features(graphs, attribute, operation):
    lapeigs = [laplacian_eigen_vectors(
        graph, 'area', operation) for graph in graphs]
    size = reduce(
            lambda x, y: min(x, y),
            [lapeig.shape[1] for lapeig in lapeigs]
        )
    return [lapeig.flatten('F') for lapeig in lapeigs]
