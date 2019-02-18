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
from . import utils

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


def rmse(model_dir, model_ext, building, dsm_bboxes, margins):
    return np.sqrt(
        np.mean(
            np.square(
                dsm_residual(
                    os.path.join(model_dir, building + '.' + model_ext),
                    margins,
                    dsm_bboxes
                )
            )
        )
    )


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
    model_ext,
    dsm_bboxes,
    margins=(0, 0),
    resolution=10
):
    alti_logger.info(
        'Computing histogram features for buildings %s in %s with extension '
        + '%s and margins %s wrt DSMs in %s with %s resolution',
        buildings,
        model_dir,
        model_ext,
        margins,
        dsm_bboxes,
        resolution
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

def get_method(model_dir, model_ext, dsm_bboxes, margins, method, **method_args):
    alti_logger.info(
        'Getting the method %s for feature computation for buildings in %s wrt'
        + ' %s applying parameters %s',
        method,
        model_dir,
        dsm_bboxes,
        method_args,
    )
    if method == 'rmse':
        return lambda building: rmse(
            model_dir,
            model_ext,
            building,
            dsm_bboxes,
            margins
        )
    elif method == 'histogram': 
        return lambda building: histogram(
            model_dir,
            model_ext,
            building,
            dsm_bboxes,
            margins,
            **method_args
        )
    elif method == 'scattering':
        return lambda building: scatter(
            model_dir,
            model_ext,
            building,
            dsm_bboxes,
            margins,
            **method_args
        )
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(method)
        )


def compute_features(buildings, cache_dir, cache_args, model_dir, model_ext, dsm_bboxes, margins, method, **method_args):
    alti_logger.info(
        'Getting the features with method %s for buildings in %s wrt'
        + ' %s applying parameters %s',
        method,
        model_dir,
        dsm_bboxes,
        method_args,
    )
    cache_args.update(
        {
            'model_dir': model_dir,
            'model_ext': model_ext,
            'margins': margins,
            'method': method,
            'parameters': method_args
        }
    )
    cached_features = utils.fetch_features(
        buildings,
        'altimetric',
        cache_dir,
        **cache_args
    )
    if method == 'histogram_custom':
        features = histogram_features(
            [
                building
                for building in buildings
                if cached_features[building] is None
            ],
            model_dir,
            model_ext,
            dsm_bboxes,
            margins,
            **method_args
        )
    else:
        features = {
            building:
            np.concatenate(
                get_method(
                    model_dir,
                    model_ext,
                    dsm_bboxes,
                    margins,
                    method,
                    **method_args
                )(building).reshape(1, -1),
                axis=-1
            )
            for building in tqdm(
                [
                building
                for building in buildings
                if cached_features[building] is None
            ],
                desc='Altimetric features using ' + method
            )
        }
    utils.cache_features(
        cache_dir,
        'altimetric',
        cache_args,
        features
    )
    cached_features.update(features)
    return cached_features


def altimetric_features(
    buildings,
    cache_dir,
    model_dir,
    dsm_dir,
    dsm_ext='geotiff',
    model_ext='tiff',
    margins=(0, 0),
    **parameters
):
    alti_logger.info(
        'Computing altimetric features for buildings %s in %s with extension'
        + ' %s wrt Orthoimages in %s with extension %s applying parameters %s',
        buildings,
        model_dir,
        model_ext,
        dsm_dir,
        model_ext,
        parameters,
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
        method['method']: 
        compute_features(
            buildings,
            cache_dir,
            {
                'dsm_dir': dsm_dir,
                'dsm_ext': dsm_ext
            },
            model_dir,
            model_ext,
            dsm_bboxes,
            margins,
            method['method'],
            **method['parameters'] if 'parameters' in method.keys() else {},
        )
        for method in parameters['methods']
    }
