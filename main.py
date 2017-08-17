#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

from graph_io import read
from graph_util import operations

from laplacian_features import laplacian_eigen_vectors


def main():
    root_path = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/dual_graphs'

    graphs = [read(graph_file) for graph_file in fnmatch.filter([os.path.join(root_path, file) for file in os.listdir(root_path)], '*.txt')]

    lapeigs = [laplacian_eigen_vectors(graph, 'area', operations[2]) for graph in graphs]
    size = reduce(lambda x, y: min(x, y), [lapeig.shape[1] for lapeig in lapeigs])
    features = [lapeig[:,-2:] for lapeig in lapeigs]


if __name__ == '__main__':
    main()
