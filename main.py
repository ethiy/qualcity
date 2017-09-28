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


def geometric_features(directory):
    return {
        os.path.splitext(graph)[0]: np.array(
            graph_io.feature_vector(
                os.path.join(directory, graph)
            )
        )
        for graph in fnmatch.filter(
            os.listdir(directory),
            '*.txt'
        )
    }


def labels_map(directory):
    return {
        os.path.splitext(shape)[0]: INV_CLASSES[
                labels_io.error_classes(
                    os.path.join(directory, shape),
                    5
                )
            ]
        for shape in fnmatch.filter(
            os.listdir(directory),
            '*.shp'
        )
    }


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

    features = [
        feature
        for _, feature in sorted(
            geometric_features(graph_dir).iteritems(),
            key=operator.itemgetter(0)
        )
    ]
    labels = [
        label
        for _, label in sorted(
            labels_map(labels_dir).iteritems(),
            key=operator.itemgetter(0)
        )
    ]

    features_per_errors = [
        [
            features[idx]
            for idx, _
            in filter(
                lambda (_, label): label == cat,
                enumerate(labels)
            )
        ]
        for cat in CLASSES.keys()
        if cat != 1
    ]

    qualified_features = [
        features[idx]
        for idx, _
        in filter(
            lambda (_, label): label != 1,
            enumerate(labels)
        )
    ]
    qualified_labels = filter(
        lambda label: label != 1,
        labels
    )

    classifier = skens.RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        max_depth=6,
        oob_score=True,
        n_jobs=-1
    )
    classifier.fit(
        qualified_features,
        [int(label != 0) for label in qualified_labels]
    )
    feature_importance = zip(
        range(len(classifier.feature_importances_)),
        classifier.feature_importances_
    )
    feature_importance.sort(key=operator.itemgetter(1), reverse=True)
    print feature_importance

    map(
        lambda (idx, tree): vizualize_tree(idx, tree),
        zip(
            range(len(classifier.estimators_)),
            classifier.estimators_
        )
    )

    print "Multiclass random forest"
    print skmodsel.cross_validate(
        classifier,
        qualified_features,
        qualified_labels,
        cv=10
    )
    print "Binary random forest"
    print skmodsel.cross_validate(
        classifier,
        qualified_features,
        [int(label != 0) for label in qualified_labels],
        cv=10
    )

    fig = plt.figure()
    ax = Axes3D(fig)

    for (col, mark, label, feat) in zip(
        ['g', 'r', 'b'],
        ['o', '^', ','],
        [CLASSES.values()[0]] + CLASSES.values()[2:],
        features_per_errors
    ):
        reduced_features = skdecomp.PCA(n_components=3).fit_transform(feat)
        x, y, z = zip(*[list(couple) for couple in list(reduced_features)])
        ax.scatter(x, y, z, label=label, c=col, marker=mark)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
