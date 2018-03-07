#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import time

import math
import operator

import numpy as np

import gdal
import gdalconst

import matplotlib.pyplot as plt


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
        for dsm, bb in building_intersections(filename, dsms).items()
    ]
    idx, _ = max(
        enumerate([crp.shape for crp in crops]),
        key=operator.itemgetter(1)
    )
    return crops[idx]


def altimetric_difference(filename, dsm_dir):
    building_dsm = find_building(filename, get_dsms(dsm_dir))
    model_dsm = read(filename)
    if building_dsm.shape == model_dsm.shape:
        return find_building(filename, get_dsms(dsm_dir)) - read(filename)
    else:
        return


def histogram_bins(diffs, near_zero_gran, big_gran):
    sorted_diffs = sorted(
        np.hstack(
            tuple([diff.flatten() for diff in diffs if diff is not None])
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


def qualified_building_diffs(raster_dir, dsm_dir):
    rasters = fnmatch.filter(
        os.listdir(raster_dir),
        '*.tiff'
    )

    return {
        raster: altimetric_difference(
            os.path.join(raster_dir, raster),
            dsm_dir
        )
        for raster in rasters
    }


def histograms(raster_dir, dsm_dir, low_gran, big_gran):
    diffs = qualified_building_diffs(raster_dir, dsm_dir)
    bins = histogram_bins(diffs.values(), low_gran, big_gran)
    return {
        raster: np.histogram(
            diff.flatten(),
            bins
        )
        if diff is not None
        else (None, None)
        for raster, diff in diffs.items()
    }


def histogram_features(raster_dir, dsm_dir, low_gran, big_gran):
    return {
        os.path.splitext(raster)[0]: histogram
        for raster, (histogram, _)
        in histograms(
            raster_dir,
            dsm_dir,
            low_gran,
            big_gran
        ).items()
    }


def main():
    raster_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
        'export-3DS/rasters'
    )

    hists = histograms(raster_dir, DSM_DIR, 100, 100)
    print(hists)

    plt.clf()
    map(
        lambda hist, bins, color: plt.step(bins[1:], hist, c=color),
        zip(
            (
                zip(
                    *[
                        _hist
                        for _hist in hists.values()
                        if _hist[0] is not None
                    ]
                ),
                plt.cm.rainbow(np.linspace(0, 1, len(hists)))
            )
        )
    )
    plt.show()


if __name__ == '__main__':
    main()
