#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import time

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

import geometry_io
import utils
import labels_io


CLASSES = {
    0: 'None',
    1: 'Unqualified',
    2: 'Building',
    3: 'Facet'
}

INV_CLASSES = {v: k for k, v in CLASSES.iteritems()}


def building_features(directory):
    return {
        os.path.splitext(building)[0]: np.array(
            geometry_io.geometric_features(
                os.path.join(directory, building),
                ['degree', 'area', 'centroid_bis', 'angle', 'angle_bis']
            )
        )
        for building in fnmatch.filter(
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


def random_forests_stats(depths, number_of_estimators, features, labels, cv):
    depth_vs_nbr = [
        [
            skmodsel.cross_validate(
                skens.RandomForestClassifier(
                    n_estimators=size,
                    class_weight="balanced",
                    max_depth=depth,
                    oob_score=True,
                    n_jobs=-1
                ),
                features,
                labels,
                cv=cv
            )['test_score']
            for depth in depths
        ]
        for size in number_of_estimators
    ]

    min_depth_vs_nbr = np.array(
        [
            [
                min(values)
                for values in row
            ]
            for row in depth_vs_nbr
        ]
    )
    max_depth_vs_nbr = np.array(
        [
            [
                max(values)
                for values in row
            ]
            for row in depth_vs_nbr
        ]
    )
    median_depth_vs_nbr = np.array(
        [
            [
                utils.median(values)
                for values in row
            ]
            for row in depth_vs_nbr
        ]
    )
    return (
            min_depth_vs_nbr,
            max_depth_vs_nbr,
            median_depth_vs_nbr
        )


def plot_rf_stats(min_rf, max_rf, median_rf):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey='row')
    min_plt = ax1.imshow(min_rf, cmap='jet')
    ax1.set_title('minimum')
    f.colorbar(min_plt, ax=ax1)
    max_plt = ax2.imshow(max_rf, cmap='jet')
    f.colorbar(max_plt, ax=ax2)
    ax2.set_title('maximum')
    median_plt = ax3.imshow(median_rf, cmap='jet')
    f.colorbar(median_plt, ax=ax3)
    ax3.set_title('median')
    f.show()


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
    raster_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
        'export-3DS/rasters'
    )
    graph_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
        'export-3DS/dual_graphs'
    )
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )

    features = [
        feature
        for _, feature in sorted(
            building_features(graph_dir).iteritems(),
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
        n_estimators=1000,
        class_weight="balanced",
        max_depth=4,
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

    start = time.time()
    print "Multiclass random forest"
    print skmodsel.cross_validate(
        classifier,
        qualified_features,
        qualified_labels,
        cv=10
    )
    print "Time taken =", time.time() - start, 'sec'
    start = time.time()
    print "Binary random forest"
    print skmodsel.cross_validate(
        classifier,
        qualified_features,
        [int(label != 0) for label in qualified_labels],
        cv=10
    )
    print "Time taken =", time.time() - start, 'sec'

    # start = time.time()
    # (min_rf, max_rf, median_rf) = random_forests_stats(
    #     range(1, 11),
    #     [25 * sz for sz in range(1, 41)],
    #     qualified_features,
    #     qualified_labels,
    #     10
    # )
    # print "Time taken =", time.time() - start, 'sec'

    # plot_rf_stats(min_rf, max_rf, median_rf)

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
