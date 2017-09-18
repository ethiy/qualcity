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

    clean_building_labels = [
        couple[0]
        for couple
        in filter(
            lambda x: x[1],
            zip(
                range(len(labels)),
                binary_labels
            )
        )
    ]

    separated_features = [
        [
            features[index]
            for index in clean_building_labels
        ],
        [
            features[index]
            for index in range(len(features))
            if index not in clean_building_labels
        ]
    ]

    fig = plt.figure()
    ax = Axes3D(fig)

    for col, mark, feat in zip(['r', 'b'], ['o', '^'], separated_features):
        reduced_features = skd.PCA(n_components=3).fit_transform(feat)
        x, y, z = zip(*[list(couple) for couple in list(reduced_features)])
        ax.scatter(x, y, z, c=col, marker=mark)

    plt.show()


if __name__ == '__main__':
    main()
