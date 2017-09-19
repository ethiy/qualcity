#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import math

import numpy as np

import gdal
import gdalconst


def bounding_box(dsm):
    dataset = gdal.Open(dsm, gdalconst.GA_ReadOnly)
    geotrans = dataset.GetGeoTransform()
    return (
        (
            geotrans[0],
            geotrans[3]
        ),
        (
            geotrans[0] + geotrans[1] * dataset.RasterXSize,
            geotrans[3] + geotrans[5] * dataset.RasterYSize
        ),
        (
            0.06,
            -0.06
        )
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


def main():
    return


if __name__ == '__main__':
    main()
