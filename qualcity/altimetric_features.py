# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

from tqdm import tqdm

import numpy as np

from . import GeoRaster

alti_logger = logging.getLogger(__name__)

DSM_DIR = '/home/ethiy/Data/Elancourt/DSM'


def find_building(bbox, dsm_bboxes):
    alti_logger.info(
        'Find instersecting DSM in %s with  %s',
        dsm_bboxes,
        bbox
    )
    return functools.reduce(
        lambda lhs, rhs: lhs.union(rhs),
        [
            GeoRaster.GeoRaster.from_file(dsm_name, dtype=np.float).crop(bbox)
            for dsm_name, dsm_bbox in dsm_bboxes.items()
            if GeoRaster.overlap(bbox, dsm_bbox)
        ]
    )


def dsm_residual(model_name, margins, dsm_bboxes):
    alti_logger.info(
        'Getting dsm residual for %s with (margins %s) with instersecting '
        + 'DSM in %s',
        model_name,
        margins,
        dsm_bboxes
    )
    return (
        find_building(
            GeoRaster.geo_info(model_name, margins)[0],
            dsm_bboxes
        )
        -
        GeoRaster.GeoRaster.from_file(
            model_name,
            dtype=np.float
        )
    ).image


def partition(residuals, low_res, high_res):
    alti_logger.info(
        'Getting histogram bins for %s with %s low values resolution'
        + 'and %s high values resolution',
        residuals,
        low_res,
        high_res
    )
    sorted_residuals = sorted(
        np.hstack([diff.flatten() for diff in residuals])
    )
    alti_logger.debug(
        'Sorted residuals: %s',
        sorted_residuals
    )
    min_max, max_min = max(
        zip(
            sorted_residuals[1:],
            sorted_residuals[:-1]
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
                math.floor(sorted_residuals[0]),
                math.ceil(max_min),
                low_res
            ),
            np.linspace(
                math.floor(min_max),
                math.ceil(sorted_residuals[-1]),
                high_res
            )
        )
    )


def histograms(
    model_dir,
    dsm_bboxes,
    margins=(0, 0),
    model_ext='*.tiff',
    low_res=5,
    high_res=5
):
    alti_logger.info(
        'Computing histograms for residuals of all buildings in %s wrt DSMs '
        + 'in %s (with extension %s and margins %s) with %s low values '
        + 'resolution and %s high values resolution',
        model_dir,
        dsm_bboxes,
        model_ext,
        margins,
        low_res,
        high_res
    )
    residuals = {
        model_name: dsm_residual(
            os.path.join(model_dir, model_name),
            margins,
            dsm_bboxes
        )
        for model_name in fnmatch.filter(
            os.listdir(model_dir),
            model_ext
        )
    }
    alti_logger.debug(
        'Residuals of Qualifiable buildings in %s wrt DSMs in %s '
        + '(with extension %s and margins %s)',
        model_dir,
        dsm_bboxes,
        model_ext,
        margins
    )
    bins = partition(residuals.values(), low_res, high_res)
    alti_logger.debug(
        'Histogram bins for %s with %s low values resolution'
        + 'and %s high values resolution',
        residuals.values(),
        low_res,
        high_res
    )
    return {
        model_name: np.histogram(
            residual.flatten(),
            bins
        ) if residual is not None
        else (None, None)
        for model_name, residual in residuals.items()
    }


def histogram_features(
    model_dir,
    dsm_dir,
    margins=(0, 0),
    model_ext='tiff',
    dsm_ext='geotiff',
    low_res=5,
    high_res=5
):
    alti_logger.info(
        'Computing histogram features for all buildings in %s with extension '
        + '%s and margins %s wrt DSMs in %s (with extension %s) with %s low '
        + 'values resolution and %s high values resolution',
        model_dir,
        dsm_ext,
        margins,
        dsm_dir,
        model_ext,
        low_res,
        high_res
    )
    dsm_bboxes = {
        os.path.join(dsm_dir, dsm_name): GeoRaster.geo_info(
            os.path.join(dsm_dir, dsm_name)
        )[0]
        for dsm_name in fnmatch.filter(
            os.listdir(dsm_dir),
            '*.' + dsm_ext
        )
    }
    alti_logger.debug(
        'All bounding boxes for DSMs in %s (with extension %s): %s',
        dsm_dir,
        dsm_ext,
        dsm_bboxes
    )
    return {
        os.path.splitext(model_name)[0]: histogram
        for model_name, (histogram, _)
        in histograms(
            model_dir,
            dsm_bboxes,
            margins,
            '*.' + model_ext,
            low_res,
            high_res
        ).items()
    }
