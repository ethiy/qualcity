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


def partition(residuals, resolution):
    alti_logger.info(
        'Getting histogram bins for %s with %s resolution',
        residuals,
        resolution
    )
    min_resid = min(
        [diff.min() for diff in residuals]
    )
    max_resid = max(
        [diff.max() for diff in residuals]
    )
    alti_logger.debug(
        'Low boundary: %s, and High boundary: %s',
        min_resid,
        max_resid
    )
    return np.linspace(
        math.floor(min_resid),
        math.ceil(max_resid),
        resolution
    )


def histograms(
    buildings,
    model_dir,
    dsm_bboxes,
    margins=(0, 0),
    model_ext='*.tiff',
    resolution=10
):
    alti_logger.info(
        'Computing histograms for residuals of buildings %s in %s wrt DSMs '
        + 'in %s (with extension %s and margins %s) with %s resolution',
        buildings,
        model_dir,
        dsm_bboxes,
        model_ext,
        margins,
        resolution
    )
    residuals = {
        model_name: dsm_residual(
            os.path.join(model_dir, model_name + model_ext),
            margins,
            dsm_bboxes
        )
        for model_name
        in tqdm(
            buildings,
            desc='  > Computing Residuals'
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
    bins = partition(residuals.values(), resolution)
    alti_logger.debug(
        'Histogram bins for %s with %s resolution',
        residuals.values(),
        resolution
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
    buildings,
    model_dir,
    dsm_dir,
    margins=(0, 0),
    model_ext='tiff',
    dsm_ext='geotiff',
    resolution=10
):
    alti_logger.info(
        'Computing histogram features for buildings %s in %s with extension '
        + '%s and margins %s wrt DSMs in %s (with extension %s) with %s resolution',
        buildings,
        model_dir,
        dsm_ext,
        margins,
        dsm_dir,
        model_ext,
        resolution
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
        in tqdm(
            histograms(
                buildings,
                model_dir,
                dsm_bboxes,
                margins,
                '.' + model_ext,
                resolution
            ).items(),
            desc='Altimetric features'
        )
    }
