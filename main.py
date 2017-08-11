#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from graphs import read
from laplacian_features import laplacian_eigen_vectors
from weisfeiler_lehman import GK_WL
from direct_product_kernel import DirectProductKernel

def main():
    root_path = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/dual_graphs'

    graphs = [read(graph_file) for graph_file in fnmatch.filter([os.path.join(root_path, file) for file in os.listdir(root_path)], '*.txt')]

    print [laplacian_eigen_vectors(graph, 'area') for graph in graphs]
    #nx.draw(G)
    #plt.show()

if __name__ == '__main__':
    main()
