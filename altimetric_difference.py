#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import time

import math
import operator

import numpy as np

import sklearn.cluster
import sklearn.model_selection

import gdal
import gdalconst

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import labels_io


DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'


def bounding_box(dsm):
    dataset = gdal.Open(dsm, gdalconst.GA_ReadOnly)
    Ox, px, _, Oy, _, py = dataset.GetGeoTransform()
    return (
        (Ox, Oy),
        (
            Ox + px * dataset.RasterXSize,
            Oy + py * dataset.RasterYSize
        ),
        (px, py)
    )


def intersection(dsm, building):
    dsm_bb = bounding_box(dsm)
    intersection = overlap(
        dsm_bb,
        bounding_box(building)
    )
    if (
        intersection[1][0] < intersection[0][0]
        or intersection[1][1] > intersection[0][1]
    ):
        return None
    else:
        return (
            (
                int(
                    math.floor(
                        (intersection[0][0] - dsm_bb[0][0]) / dsm_bb[2][0]
                    )
                ),
                int(
                    math.floor(
                        (intersection[0][1] - dsm_bb[0][1]) / dsm_bb[2][1]
                    )
                )
            ),
            (
                int(
                    math.floor(
                        (intersection[1][0] - dsm_bb[0][0]) / dsm_bb[2][0]
                    )
                ),
                int(
                    math.floor(
                        (intersection[1][1] - dsm_bb[0][1]) / dsm_bb[2][1]
                    )
                )
            )
        )


def overlap(bb_1, bb_2):
    return (
        (
            max(bb_2[0][0], bb_1[0][0]),
            min(bb_2[0][1], bb_1[0][1])
        ),
        (
            min(bb_2[1][0], bb_1[1][0]),
            max(bb_2[1][1], bb_1[1][1])
        )
    )


def crop(filename, roi):
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray(
        roi[0][0],
        roi[0][1],
        roi[1][0] - roi[0][0],
        roi[1][1] - roi[0][1]
    ).astype(np.float)


def read(filename):
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray().astype(np.float)


def get_dsms(dsm_dir):
    return fnmatch.filter(
        [os.path.join(dsm_dir, filename) for filename in os.listdir(dsm_dir)],
        '*.geotiff'
    )


def building_intersections(filename, dsms):
    return {
        dsm: intersection(dsm, filename)
        for dsm in dsms
        if intersection(dsm, filename) is not None
    }


def find_building(filename, dsms):
    crops = [
        crop(dsm, bb)
        for dsm, bb in building_intersections(filename, dsms).iteritems()
    ]
    idx, _ = max(
        enumerate([crp.shape for crp in crops]),
        key=operator.itemgetter(1)
    )
    return crops[idx]


def altimetric_difference(filename, dsm_dir):
    return find_building(filename, get_dsms(dsm_dir)) - read(filename)


def histogram_bins(diffs, near_zero_gran, big_gran):
    sorted_diffs = sorted(
        np.hstack(
            tuple([diff.flatten() for diff in diffs])
        )
    )
    min_max, max_min, _ = max(
        [
            (a, b, a - b)
            for a, b in zip(
                sorted_diffs[1:],
                sorted_diffs[:-1]
            )
        ],
        key=operator.itemgetter(2)
    )

    return np.hstack(
        (
            np.linspace(
                math.floor(sorted_diffs[0]),
                math.ceil(max_min),
                near_zero_gran
            ),
            np.linspace(
                math.floor(min_max),
                math.ceil(sorted_diffs[-1]),
                big_gran
            )
        )
    )


def qualified_building_diffs(raster_dir, labels_dir, dsm_dir):
    rasters = fnmatch.filter(
        os.listdir(raster_dir),
        '*.tiff'
    )

    labels = {
        os.path.splitext(graph)[0]: labels_io.error_classes(
            os.path.join(labels_dir, graph),
            5
        )
        for graph in fnmatch.filter(os.listdir(labels_dir), '*.shp')
    }

    return {
        raster: altimetric_difference(
            os.path.join(raster_dir, raster),
            dsm_dir
        )
        for raster in rasters
        if labels[
            os.path.splitext(raster)[0]
        ] != 'Unqualified'
    }


def histograms(raster_dir, labels_dir, dsm_dir, low_gran, big_gran):
    diffs = qualified_building_diffs(raster_dir, labels_dir, dsm_dir)
    bins = histogram_bins(diffs.values(), low_gran, big_gran)
    return {
        raster: np.histogram(
            diff.flatten(),
            bins
        )
        for raster, diff in diffs.iteritems()
    }


def histogram_features(raster_dir, labels_dir, dsm_dir, low_gran, big_gran):
    return {
        os.path.splitext(raster)[0]: histogram
        for raster, (histogram, _)
        in histograms(
            raster_dir,
            labels_dir,
            dsm_dir,
            low_gran,
            big_gran
        ).iteritems()
    }


def visualize_feature(ax, color, features):
    if features.shape[0] > 1:
        x, y, z = zip(
            *[
                list(couple)
                for couple
                in list(
                    sklearn.decomposition.PCA(n_components=3).fit_transform(
                        features
                    )
                )
            ]
        )
        ax.scatter(x, y, z, c=color)


def main():
    raster_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
        'export-3DS/rasters'
    )
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )

    hists = histograms(raster_dir, labels_dir, DSM_DIR, 100, 100)

    plt.clf()
    map(
        lambda ((hist, bins), color): plt.step(bins[1:], hist, c=color),
        zip(
            hists.values(),
            plt.cm.rainbow(np.linspace(0, 1, len(hists)))
        )
    )

    labels = [
        label
        for _, label in sorted(
            labels_io.labels_map(labels_dir).iteritems(),
            key=operator.itemgetter(0)
        )
        if label != 1
    ]

    altimetric_features = [
        feature
        for _, feature in sorted(
            histogram_features(
                raster_dir,
                labels_dir,
                DSM_DIR,
                100,
                100
            ).iteritems(),
            key=operator.itemgetter(0)
        )
    ]

    centers = 7
    start = time.time()
    print 'Spectral Clustering: ', centers
    cluster_labels = sklearn.cluster.SpectralClustering(
        n_clusters=centers,
        n_jobs=1
    ).fit_predict(
        altimetric_features
    )

    print 'Time taken = ', time.time() - start, 'sec'

    figure = plt.figure()
    ax = Axes3D(figure)

    features_per_clusters = [
        np.array(
            [
                altimetric_features[idx]
                for idx, _
                in filter(
                    lambda (_, label): label == cluster,
                    enumerate(cluster_labels)
                )
            ]
        )
        for cluster in set(cluster_labels)
    ]

    for cl in features_per_clusters:
        print cl.shape

    map(
        lambda (color, features): visualize_feature(
            ax,
            color,
            features
        ),
        zip(
            plt.cm.rainbow(np.linspace(0, 1, centers)),
            features_per_clusters
        )
    )
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
