#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

import numpy as np

from .GeoRaster import overlap, bounding_box, GeoRaster

alti_logger = logging.getLogger(__name__)

DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'


def find_building(bbox, dsm_bboxes):
    alti_logger.info(
        'Getting %s corresponding DSM in %s',
        bbox,
        dsm_bboxes
    )
    return functools.reduce(
        lambda lhs, rhs: lhs.union(rhs),
        [
            GeoRaster.from_file(dsm_name, dtype=np.float).crop(bbox)
            for dsm_name, dsm_bbox in dsm_bboxes.items()
            if overlap(bbox, dsm_bbox)
        ]
    )


def altimetric_difference(filename, dsm_bboxes):
    alti_logger.info(
        'Getting altimetric residual for %s with max instersecting DSM in %s',
        filename,
        dsm_bboxes
    )
    model_dsm = GeoRaster.from_file(
        filename,
        dtype=np.float
    )
    alti_logger.debug(
        'Building DSM from 3d model %s -> %s ',
        filename,
        model_dsm
    )
    building_dsm = find_building(model_dsm.bbox, dsm_bboxes)
    alti_logger.debug(
        'Max instersecting DSM in %s with %s -> %s ',
        dsm_bboxes,
        filename,
        building_dsm
    )
    return (building_dsm - model_dsm).image


def histogram_bins(diffs, low_res, high_res):
    alti_logger.info(
        'Getting histogram bins for %s with %s low values resolution'
        + 'and %s high values resolution',
        diffs,
        low_res,
        high_res
    )
    sorted_diffs = sorted(
        np.hstack([diff.flatten() for diff in diffs])
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


def dsm_diff(raster_dir, dsm_dir, ext, dsm_ext):
    alti_logger.info(
        'Getting residuals for all buildings in %s (with extension %s) '
        + 'wrt DSMs in %s (with extension %s)',
        raster_dir,
        ext,
        dsm_dir,
        dsm_ext
    )
    alti_logger.info(
        'Computing all bounding boxes for DSMs in %s with extention %s',
        dsm_dir,
        dsm_ext
    )
    dsm_bboxes = {
        os.path.join(dsm_dir, dsm_name): bounding_box(
            os.path.join(dsm_dir, dsm_name)
        )
        for dsm_name in fnmatch.filter(
            os.listdir(dsm_dir),
            dsm_ext
        )
    }
    alti_logger.debug(dsm_bboxes)
    return {
        raster: altimetric_difference(
            os.path.join(raster_dir, raster),
            dsm_bboxes
        )
        for raster in fnmatch.filter(
            os.listdir(raster_dir),
            ext
        )
    }


def histograms(
    raster_dir,
    dsm_dir,
    ext='*.tiff',
    dsm_ext='*.getiff',
    low_res=5,
    high_res=5
):
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
    diffs = dsm_diff(raster_dir, dsm_dir, ext, dsm_ext)
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
    ext='tiff',
    dsm_ext='geotiff',
    low_res=5,
    high_res=5
):
    alti_logger.info(
        'Computing histogram features for all buildings in %s with extension '
        + '%s wrt DSMs in %s with extension %s with %s low values resolution'
        + 'and %s high values resolution',
        raster_dir,
        dsm_ext,
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
            '*.' + ext,
            '*.' + dsm_ext,
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
