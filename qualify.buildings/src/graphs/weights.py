# -*- coding: <utf-8> -*-

import networkx as nx

def compute_weight(graph, attribute):
    for u, v, data in graph.edges(data=True):
        data = graph.node[u][attribute] * graph.node[v][attribute]
    return graph
