#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

import numpy as np

import qualcity.GeoRaster

alti_logger = logging.getLogger(__name__)

DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'


def crop(bbox, dsm_path, margins):
    crop = qualcity.GeoRaster.GeoRaster.from_file(
        dsm_path,
        dtype=np.float
    ).crop(
        bbox,
        margins
    )
    return crop if crop.size() else None


def find_building(bbox, dsm_dir, ext, margins=(0, 0)):
    alti_logger.info(
        'Getting %s corresponding DSM in %s with extention %s',
        bbox,
        dsm_dir,
        ext
    )
    dsm_crops = [
        crop(bbox, os.path.join(dsm_dir, dsm_name), margins)
        for dsm_name in fnmatch.filter(
            os.listdir(dsm_dir),
            ext
        )
    ]
    if len(dsm_crops) > 1:
        print(bbox)
        print([dsm_crp.shape() for dsm_crp in dsm_crops if dsm_crp])
    return functools.reduce(
        operator.add,
        [dsm_crp for dsm_crp in dsm_crops if dsm_crp]
    )


def altimetric_difference(filename, dsm_dir, ext):
    alti_logger.info(
        'Getting altimetric residual for %s with max instersecting DSM in %s',
        filename,
        dsm_dir
    )
    print(filename)
    model_dsm = qualcity.GeoRaster.GeoRaster.from_file(
        filename,
        dtype=np.float
    )
    alti_logger.debug(
        'Building DSM from 3d model %s -> %s ',
        filename,
        model_dsm
    )
    building_dsm = find_building(model_dsm.bbox(), dsm_dir, ext)
    alti_logger.debug(
        'Max instersecting DSM in %s with %s -> %s ',
        dsm_dir,
        filename,
        building_dsm
    )
    if building_dsm.shape() == model_dsm.shape():
        return building_dsm.image - model_dsm.image
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
            dsm_dir,
            ext='*.geotiff'
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
