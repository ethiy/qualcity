#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

from graph_io import read
from graph_util import operations

from laplacian_features import laplacian_features


def graph_files(directory):
    return fnmatch.filter(
        [os.path.join(directory, file) for file in os.listdir(directory)],
        '*.txt'
    )


def main():
    root_path = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/dual_graphs'

    graphs = [read(graph_file) for graph_file in graph_files(root_path)]

    print laplacian_features(graphs, 'area', operations[2])[1]


if __name__ == '__main__':
    main()
