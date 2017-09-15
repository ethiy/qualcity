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
    graph_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
        'dual_graphs'
    )
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )

    features = np.vstack(
        [
            np.array(
                graph_io.feature_vector(graph)
            )
            for graph in graph_files(graph_dir)
        ]
    )
    labels = [
        labels_io.errors_per_building(os.path.join(labels_dir, graph))
        for graph in fnmatch.filter(os.listdir(labels_dir), '*.shp')
    ]

    binary_labels = np.array(
        [
            label == ['None', 'None', 'None']
            for label in labels
        ]
    )

    print binary_labels

    reduced_features = skd.PCA(n_components=3).fit_transform(features)
    x, y, z = zip(*[list(couple) for couple in list(reduced_features)])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    main()
