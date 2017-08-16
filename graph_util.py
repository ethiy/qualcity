# -*- coding: <utf-8> -*-

import networkx as nx
import numpy as np

operations = ['multiplication', 'resistance', 'addition', 'scalar_product']

def compute_weight(graph, attribute, operation):
    if operation == 'multiplication':
        for u, v, data in graph.edges(data=True):
            data = graph.node[u][attribute] * graph.node[v][attribute]
    elif operation == 'resistance':
        for u, v, data in graph.edges(data=True):
            data = 1 / ((1 / graph.node[u][attribute]) * (1 / graph.node[v][attribute]))
    elif operation == 'addition':
        for u, v, data in graph.edges(data=True):
            data = graph.node[u][attribute] + graph.node[v][attribute]
    elif operation == 'scalar_product':
        for u, v, data in graph.edges(data=True):
            data = np.dot(graph.node[u][attribute], graph.node[v][attribute])
    else:
        raise AttributeError
    return graph
