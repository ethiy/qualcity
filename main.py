#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import sklearn.decomposition as skd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import graph_io
import labels_io


def graph_files(directory):
    return fnmatch.filter(
        [os.path.join(directory, file) for file in os.listdir(directory)],
        '*.txt'
    )


def main():
    root_path = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
        'dual_graphs'
    )

    features = np.vstack(
        [
            np.array(
                graph_io.feature_vector(graph_file)
            )
            for graph_file in graph_files(root_path)
        ]
    )
    reduced_features = skd.PCA(n_components=3).fit_transform(features)
    x, y, z = zip(*[list(couple) for couple in list(reduced_features)])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    main()
