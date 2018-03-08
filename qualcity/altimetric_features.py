#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator

import numpy as np

import gdal
import gdalconst

alti_logger = logging.getLogger(__name__)

DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'


def bounding_box(dsm):
    alti_logger.info('Getting %s bounding_box', dsm)
    dataset = gdal.Open(dsm, gdalconst.GA_ReadOnly)
    alti_logger.info('Getting geo-transform from %s', dsm)
    Ox, px, _, Oy, _, py = dataset.GetGeoTransform()
    alti_logger.debug(
        'Geo-transform -> origin: %s, pixel resolution: %s',
        (Ox, Oy),
        (px, py)
    )
    return (
        (Ox, Oy),
        (
            Ox + px * dataset.RasterXSize,
            Oy + py * dataset.RasterYSize
        ),
        (px, py)
    )


def intersection(dsm, building):
    alti_logger.info('Intersection of %s and %s', dsm, building)
    dsm_bb = bounding_box(dsm)
    alti_logger.debug('%s bounding box: %s', dsm, dsm_bb)
    intersection = overlap(
        dsm_bb,
        bounding_box(building)
    )
    alti_logger.debug('%s overlap with %s -> %s', dsm, building, intersection)
    if (
        intersection[1][0] < intersection[0][0]
        or intersection[1][1] > intersection[0][1]
    ):
        alti_logger.debug('No overlap between %s, %s', dsm, building)
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
    alti_logger.info('Overlap between %s and %s', bb_1, bb_2)
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
    alti_logger.info('Croping region of interest %s from %s', roi, filename)
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray(
        roi[0][0],
        roi[0][1],
        roi[1][0] - roi[0][0],
        roi[1][1] - roi[0][1]
    ).astype(np.float)


def read(filename):
    alti_logger.info('Reading %s', filename)
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray().astype(np.float)


def get_dsms(dsm_dir):
    alti_logger.info('Getting Stored DSMs in %s', dsm_dir)
    return fnmatch.filter(
        [os.path.join(dsm_dir, filename) for filename in os.listdir(dsm_dir)],
        '*.geotiff'
    )


def building_intersections(filename, dsms):
    alti_logger.info(
        'Getting intersections between DSMs in %s and %s',
        dsms,
        filename
    )
    return [
        crop(
            dsm,
            intersection(dsm, filename)
        )
        for dsm in dsms
        if intersection(dsm, filename) is not None
    ]


def find_building(filename, dsms):
    alti_logger.info(
        'Getting max intersecting DSM in %s with %s',
        dsms,
        filename
    )
    return max(
        building_intersections(filename, dsms),
        key=lambda crop: crop.shape
    )


def altimetric_difference(filename, dsm_dir):
    alti_logger.info(
        'Getting altimetric residual for %s with max instersecting DSM in %s',
        filename,
        dsm_dir
    )
    building_dsm = find_building(filename, get_dsms(dsm_dir))
    alti_logger.debug(
        'Max instersecting DSM in %s with %s -> %s ',
        dsm_dir,
        filename,
        building_dsm
    )
    model_dsm = read(filename)
    alti_logger.debug(
        'Building DSM from 3d model %s -> %s ',
        filename,
        model_dsm
    )
    if building_dsm.shape == model_dsm.shape:
        return building_dsm - model_dsm
    else:
        alti_logger.warn(
            '%s border building -> ToDo: try merging with georasters!',
            filename
        )
        return None


def histogram_bins(diffs, low_res, high_res):
    alti_logger.info(
        'Getting histogram bins for %s with %s low values resolution'
        + 'and %s high values resolution',
        diffs,
        low_res,
        high_res
    )
    sorted_diffs = sorted(
        np.hstack(
            tuple([diff.flatten() for diff in diffs if diff is not None])
        )
    )
    alti_logger.debug(
        'Sorted residuals: %s',
        sorted_diffs
    )
    min_max, max_min = max(
        zip(
            sorted_diffs[1:],
            sorted_diffs[:-1]
        ),
        key=lambda x: operator.sub(*x)
    )
    alti_logger.debug(
        'Low values boundary: %s, and High values boundary: %s',
        min_max,
        max_min
    )
    return np.hstack(
        (
            np.linspace(
                math.floor(sorted_diffs[0]),
                math.ceil(max_min),
                low_res
            ),
            np.linspace(
                math.floor(min_max),
                math.ceil(sorted_diffs[-1]),
                high_res
            )
        )
    )


def qualifiable_building_diffs(raster_dir, dsm_dir, ext='*.tiff'):
    alti_logger.info(
        'Getting residuals for all buildings in %s wrt DSMs in %s '
        + 'with extension %s',
        raster_dir,
        dsm_dir,
        ext
    )
    return {
        raster: altimetric_difference(
            os.path.join(raster_dir, raster),
            dsm_dir
        )
        for raster in fnmatch.filter(
            os.listdir(raster_dir),
            ext
        )
    }


def histograms(raster_dir, dsm_dir, ext='*.tiff', low_res=5, high_res=5):
    alti_logger.info(
        'Computing histograms for residuals of all buildings in %s wrt DSMs '
        + 'in %s with extension %s with %s low values resolution'
        + 'and %s high values resolution',
        raster_dir,
        dsm_dir,
        ext,
        low_res,
        high_res
    )
    diffs = qualifiable_building_diffs(raster_dir, dsm_dir, ext)
    alti_logger.debug(
        'Residuals of Qualifiable buildings in %s wrt DSMs in %s '
        + 'with extension %s',
        raster_dir,
        dsm_dir,
        ext
    )
    bins = histogram_bins(diffs.values(), low_res, high_res)
    alti_logger.debug(
        'Histogram bins for %s with %s low values resolution'
        + 'and %s high values resolution',
        diffs.values(),
        low_res,
        high_res
    )
    return {
        raster: np.histogram(
            diff.flatten(),
            bins
        ) if diff is not None
        else (None, None)
        for raster, diff in diffs.items()
    }


def histogram_features(
    raster_dir,
    dsm_dir,
    ext='*.tiff',
    low_res=5,
    high_res=5
):
    alti_logger.info(
        'Computing histogram features for all buildings in %s wrt DSMs '
        + 'in %s with extension %s with %s low values resolution'
        + 'and %s high values resolution',
        raster_dir,
        dsm_dir,
        ext,
        low_res,
        high_res
    )

    return {
        os.path.splitext(raster)[0]: histogram
        for raster, (histogram, _)
        in histograms(
            raster_dir,
            dsm_dir,
            ext,
            low_res,
            high_res
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
