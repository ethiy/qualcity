#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import operator

import sklearn.decomposition as skdecomp
import sklearn.ensemble as skens
import sklearn.tree as sktree
import sklearn.model_selection as skmodsel

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import graph_io
import labels_io


CLASSES = {
    0: 'None',
    1: 'Unqualified',
    2: 'Building',
    3: 'Facet'
}

INV_CLASSES = {v: k for k, v in CLASSES.iteritems()}


def graph_files(directory):
    return fnmatch.filter(
        [os.path.join(directory, file) for file in os.listdir(directory)],
        '*.txt'
    )


def vizualize_tree(idx, tree):
    sktree.export_graphviz(
        tree,
        out_file=(
            './ressources/output/randomforest/trees/'
            +
            'tree-' + str(idx) + '.dot'
        ),
        filled=True,
        rounded=True
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

    labels_classes = [
        INV_CLASSES[
            labels_io.error_classes(
                os.path.join(labels_dir, graph),
                5
            )
        ]
        for graph in fnmatch.filter(os.listdir(labels_dir), '*.shp')
    ]

    features_per_errors = [
        [
            features[idx]
            for idx, _
            in filter(
                lambda x: x[1] == cat,
                zip(
                    range(len(labels_classes)),
                    labels_classes
                )
            )
        ]
        for cat in CLASSES.keys()
    ]

    classifier = skens.RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        max_depth=6,
        oob_score=True,
        n_jobs=-1
    )
    classifier.fit(features, [int(label != 0) for label in labels_classes])
    feature_importance = zip(
        range(len(classifier.feature_importances_)),
        classifier.feature_importances_
    )
    feature_importance.sort(key=operator.itemgetter(1), reverse=True)
    print feature_importance

    map(
        lambda couple: vizualize_tree(couple[0], couple[1]),
        zip(
            range(len(classifier.estimators_)),
            classifier.estimators_
        )
    )

    print "Multiclass random forest"
    print skmodsel.cross_validate(classifier, features, labels_classes, cv=10)
    print "Binary random forest"
    print skmodsel.cross_validate(
        classifier,
        features,
        [int(label != 0) for label in labels_classes],
        cv=10
    )

    fig = plt.figure()
    ax = Axes3D(fig)

    for (col, mark, label, feat) in zip(
        ['g', 'r', 'b', 'm'],
        ['o', '^', ',', 'd'],
        CLASSES.values(),
        features_per_errors
    ):
        reduced_features = skdecomp.PCA(n_components=3).fit_transform(feat)
        x, y, z = zip(*[list(couple) for couple in list(reduced_features)])
        ax.scatter(x, y, z, label=label, c=col, marker=mark)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
