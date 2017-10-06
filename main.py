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
import altimetric_difference
import utils
import labels_io


CLASSES = {
    0: 'None',
    1: 'Unqualified',
    2: 'Building',
    3: 'Facet'
}

INV_CLASSES = {v: k for k, v in CLASSES.iteritems()}


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


def detect_unqualified_buildings(classifier, features, binary_labels, cv):
    start = time.time()
    print "Binary classification detecting unqualified buildings"
    print skmodsel.cross_validate(
        classifier,
        features,
        binary_labels,
        cv=cv,
        scoring=[
            'accuracy',
            'f1_weighted',
            'average_precision',
            'precision',
            'recall'
        ]
    )
    print "Time taken =", time.time() - start, 'sec'


def coarse_classification_buildings(classifier, features, binary_labels, cv):
    start = time.time()
    print "Coarse Classification"
    print skmodsel.cross_validate(
        classifier,
        features,
        binary_labels,
        cv=cv,
        scoring=[
            'accuracy',
            'precision_weighted',
            'recall_weighted'
        ]
    )
    print "Time taken =", time.time() - start, 'sec'


def feature_impotance_eval(classifier, features, labels):
    classifier.fit(features, labels)
    feature_importance = zip(
        range(len(classifier.feature_importances_)),
        classifier.feature_importances_
    )
    feature_importance.sort(key=operator.itemgetter(1), reverse=True)
    return feature_importance


def visualize_feature(ax, color, marker, label, features, dims):
    if dims is None:
        reduced = skdecomp.PCA(n_components=3).fit_transform(features)
    elif type(dims) is list and len(dims) == 3:
        reduced = features[:, dims]
    else:
        LookupError
    x, y, z = zip(
        *[
            list(couple)
            for couple
            in list(
                reduced
            )
        ]
    )
    ax.scatter(x, y, z, label=label, c=color, marker=marker)


def visualize_features(features, labels, dims=None):
    fig = plt.figure()
    ax = Axes3D(fig)

    features_per_errors = [
        np.array(
            [
                features[idx]
                for idx, _
                in filter(
                    lambda (_, label): label == cat,
                    enumerate(labels)
                )
            ]
        )
        for cat in CLASSES.keys()
        if cat != 1
    ]

    map(
        lambda (color, marker, label, features): visualize_feature(
            ax,
            color,
            marker,
            label,
            features,
            dims
        ),
        zip(
            ['g', 'r', 'b'],
            ['o', '^', ','],
            [CLASSES.values()[0]] + CLASSES.values()[2:],
            features_per_errors
        )
    )

    ax.legend()
    plt.show()


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

    altimetric_features = [
        feature
        for _, feature in sorted(
            altimetric_difference.histogram_features(
                raster_dir,
                labels_dir,
                altimetric_difference.DSM_DIR,
                100,
                100
            ).iteritems(),
            key=operator.itemgetter(0)
        )
    ]
    geometric_features = [
        feature
        for _, feature in sorted(
            geometry_io.geometric_features(
                graph_dir,
                ['degree', 'area', 'centroid_bis', 'angle', 'angle_bis']
            ).iteritems(),
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

    classifier = skens.RandomForestClassifier(
        n_estimators=1000,
        class_weight="balanced",
        max_depth=4,
        oob_score=True
    )

    detect_unqualified_buildings(
        classifier,
        geometric_features,
        [int(label == 1) for label in labels],
        10
    )

    qualified_geometric_features = [
        geometric_features[idx]
        for idx, _
        in filter(
            lambda (_, label): label != 1,
            enumerate(labels)
        )
    ]

    qualified_features = [
        np.hstack(
            feats
        )
        for feats
        in zip(
            qualified_geometric_features,
            altimetric_features
        )
    ]

    qualified_labels = filter(
        lambda label: label != 1,
        labels
    )

    coarse_classification_buildings(
        classifier,
        qualified_features,
        qualified_labels,
        10
    )

    print feature_impotance_eval(
        classifier,
        qualified_features,
        qualified_labels
    )

    visualize_features(
        qualified_features,
        qualified_labels,
        [
            idx
            for idx, _
            in feature_impotance_eval(
                classifier,
                qualified_features,
                qualified_labels
            )[:3]
        ]
    )

    start = time.time()
    (min_rf, max_rf, median_rf) = random_forests_stats(
        range(1, 21),
        [25 * sz for sz in range(1, 101)],
        qualified_features,
        qualified_labels,
        10
    )
    print "Time taken =", time.time() - start, 'sec'

    plot_rf_stats(min_rf, max_rf, median_rf)


if __name__ == '__main__':
    main()
