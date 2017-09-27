#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import math

import numpy as np

import gdal
import gdalconst

import matplotlib.pyplot as plt


DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'

RASTER_DIR = os.path.join(
    '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
    'export-3DS/rasters'
)


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
    return [
        crop(dsm, bb)
        for dsm, bb in building_intersections(filename, dsms).iteritems()
    ][0]


def altimetric_difference(filename, dsm_dir):
    return find_building(filename, get_dsms(dsm_dir)) - read(filename)


def main():
    projected_building = read(
        os.path.join(
            RASTER_DIR,
            '23584.tiff'
        )
    )
    dsm_building = find_building(
        os.path.join(
            RASTER_DIR,
            '23584.tiff'
        ),
        get_dsms(DSM_DIR)
    )
    diff = altimetric_difference(
        os.path.join(
            RASTER_DIR,
            '23584.tiff'
        ),
        DSM_DIR
    )

    hist, bins = np.histogram(diff, bins=50)

    plt.close('all')
    figure = plt.figure()
    figure.add_subplot(221)
    plt.imshow(projected_building, cmap='viridis')
    plt.colorbar()
    figure.add_subplot(222)
    plt.imshow(dsm_building)
    plt.colorbar()
    figure.add_subplot(223)
    plt.imshow(diff, cmap='viridis')
    plt.colorbar()
    figure.add_subplot(224)
    plt.plot(bins[1:], hist)
    plt.show()


if __name__ == '__main__':
    main()
