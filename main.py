#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import time

import os
import fnmatch

import operator

import sklearn.decomposition
import sklearn.ensemble
import sklearn.tree
import sklearn.model_selection
import sklearn.svm

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def labels_stats(labels):
    return [
        len(
            filter(
                lambda label: label == cat,
                labels
            )
        ) / float(len(labels)) * 100
        for cat in set(labels)
    ]


def random_forests_stats(depths, number_of_estimators, features, labels, cv):
    depth_vs_nbr = [
        [
            sklearn.model_selection.cross_validate(
                sklearn.ensemble.RandomForestClassifier(
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


def plot_rf_stats(min_rf, max_rf, median_rf, figure):
    ax1 = figure.add_subplot(131)
    min_plt = ax1.imshow(min_rf, cmap='jet')
    ax1.set_title('minimum')
    figure.colorbar(min_plt, ax=ax1)
    ax2 = figure.add_subplot(132)
    max_plt = ax2.imshow(max_rf, cmap='jet')
    figure.colorbar(max_plt, ax=ax2)
    ax2.set_title('maximum')
    ax3 = figure.add_subplot(133)
    median_plt = ax3.imshow(median_rf, cmap='jet')
    figure.colorbar(median_plt, ax=ax3)
    ax3.set_title('median')


def vizualize_tree(idx, tree):
    sklearn.tree.export_graphviz(
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
    print sklearn.model_selection.cross_validate(
        classifier,
        features,
        binary_labels,
        cv=cv
    )
    print "Time taken =", time.time() - start, 'sec'


def coarse_classification_buildings(classifier, features, binary_labels, cv):
    start = time.time()
    print "Coarse Classification"
    print sklearn.model_selection.cross_validate(
        classifier,
        features,
        binary_labels,
        cv=cv
    )
    print "Time taken =", time.time() - start, 'sec'


def feature_impotance_eval(classifier, features, labels, geom_attribs):
    classifier.fit(features, labels)
    feature_importance = enumerate(classifier.feature_importances_)
    geom_attr_map = dict(geometry_io.features_anntotations(geom_attribs))
    feature_importance = map(
        lambda (idx, impor):  (geom_attr_map[idx], impor)
        if idx < len(geom_attribs) * 4
        else (idx, impor),
        feature_importance
    )
    feature_importance.sort(key=operator.itemgetter(1), reverse=True)

    return filter(
        lambda (_, impor): impor > 0,
        feature_importance
    )


def evaluate_rf(classifier, qualified_features, qualified_labels):
    feat_import = feature_impotance_eval(
        classifier,
        qualified_features,
        qualified_labels,
        ['degree', 'area', 'centroid_bis', 'angle', 'angle_bis']
    )

    print feat_import

    camembert_fig = plt.figure(2)
    ax = camembert_fig.add_subplot(111)
    ax.pie(
        map(operator.itemgetter(1), feat_import),
        labels=map(operator.itemgetter(0), feat_import),
        explode=len(feat_import) * [.1],
        startangle=90
    )
    ax.axis('equal')
    camembert_fig.show()

    start = time.time()
    (min_rf, max_rf, median_rf) = random_forests_stats(
        range(1, 11),
        [25 * sz for sz in range(1, 101)],
        qualified_features,
        qualified_labels,
        10
    )
    print "Time taken =", time.time() - start, 'sec'

    np.dstack((min_rf, max_rf, median_rf)).tofile(
        './ressources/output/randomforest/rf_stats_alt_geom.csv',
        sep=',',
        format='%10.5f'
    )

    fig_stat = plt.figure(3)
    plot_rf_stats(min_rf, max_rf, median_rf, fig_stat)
    fig_stat.show()


def visualize_feature(ax, color, marker, label, features, dims):
    if dims is None:
        reduced = sklearn.decomposition.PCA(n_components=3).fit_transform(
            features
        )
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


def error_detection(labels, features, classifier):
    print "Label statistics: ", labels_stats(labels)

    detect_unqualified_buildings(
        classifier,
        features[0],
        [int(label == 1) for label in labels],
        10
    )

    filtered_whole_features = [
        features[0][idx]
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
            filtered_whole_features,
            features[1]
        )
    ]

    qualified_labels = filter(
        lambda label: label != 1,
        labels
    )

    viz_fig = plt.figure(1)
    visualize_features(
        qualified_features,
        qualified_labels,
        viz_fig
    )
    viz_fig.show()

    print "Qualified label statistics: ", labels_stats(qualified_labels)

    coarse_classification_buildings(
        classifier,
        qualified_features,
        qualified_labels,
        10
    )

    {
        sklearn.ensemble.forest.RandomForestClassifier: evaluate_rf(
            classifier,
            qualified_features,
            qualified_labels
        ),
        sklearn.svm.classes.SVC: None
    }[type(classifier)]


def visualize_features(features, labels, figure, dims=None):
    ax = Axes3D(figure)

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

    error_detection(
        labels,
        (geometric_features, altimetric_features),
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=1000,
            class_weight="balanced",
            max_depth=4,
            oob_score=True
        )
    )
    plt.show()


if __name__ == '__main__':
    main()
