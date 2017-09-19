#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os

import math

import numpy as np

import gdal
import gdalconst

import matplotlib.pyplot as plt


DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'

RASTER_DIR = os.path.join(
    '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1247-13705',
    'export-3DS'
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
    building_bb = bounding_box(building)
    intersection = overlap(dsm_bb, building_bb)
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
    return


def find_building(filename, dsms):
    return


def altimetric_difference(filename, dsm_dir):
    return find_building(filename, get_dsms(dsm_dir)) - read(filename)


def main():
    projected_building = read(
        os.path.join(
            RASTER_DIR,
            'Export_Matis_EXPORT_1247-13705_T748_projected_xy.tiff'
        )
    )
    print projected_building.shape
    print bounding_box(
        os.path.join(
            RASTER_DIR,
            'Export_Matis_EXPORT_1247-13705_T748_projected_xy.tiff'
        )
    )
    plt.imshow(projected_building)
    plt.show()


if __name__ == '__main__':
    main()
