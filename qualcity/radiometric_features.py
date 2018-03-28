# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

import numpy as np

import matplotlib.pyplot as plt
import shapely.ops

from qualcity import GeoBuilding, GeoRaster

radio_logger = logging.getLogger(__name__)


def add_margins(bbox, resolution, margins):
    return (
        (
            bbox[0][0] - resolution[0] * margins[0],
            bbox[0][1] + resolution[1] * margins[1]
        ),
        (
            bbox[1][0] + resolution[0] * margins[0],
            bbox[1][1] - resolution[1] * margins[1]
        )
    )


def get_lines(line_string):
    for i in range(len(line_string) - 1):
        yield line_string[i: i + 2]


def norm(line):
    return math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1])


def normal(line):
    (x0, y0), (x1, y1) = line
    if (x0, y0) != (x1, y1):
        length = norm(line)
        return (
            (x1 - x0) / length,
            (y0 - y1) / length
        )
    else:
        raise RuntimeError('{} is a point not a line'.format(line))


def scalar_product(lvector, rvector):
    return functools.reduce(
        operator.add,
        [
            l_coef * r_coef
            for l_coef, r_coef in zip(lvector, rvector)
        ]
    )


def correlation(lvector, rvector):
    return (
        scalar_product(lvector, rvector) / (
            math.hypot(*lvector) * math.hypot(*rvector)
        )
        if math.hypot(*lvector) != 0 else math.inf
    )


def angle(lvector, rvector):
    return math.acos(correlation(lvector, rvector))


def find_building(building, ortho_infos, clip, margins):
    radio_logger.info(
        '%s %s corresponding Orthoimage in %s with margins %s',
        'Cliping' if clip else 'Cropping',
        building.bbox,
        ortho_infos,
        margins
    )
    if clip:
        _, resolutions = zip(*ortho_infos.values())
        radio_logger.debug(
            'All orthoimage resolutions: %s',
            resolutions
        )
        masks = {
            resolution: building.rasterize(
                resolution,
                dtype=np.uint8
            ).apply(
                lambda mask: np.stack((mask, ) * 3, -1),
                vectorize=False
            )
            for resolution in set(resolutions)
        }
        radio_logger.debug(
            'All rasterized masks corresponding to building: %s',
            masks
        )
        return functools.reduce(
            lambda lhs, rhs: (
                lhs[0].union(rhs[0]),
                lhs[1].union(rhs[1])
            ),
            [
                (
                    GeoRaster.GeoRaster.from_file(
                        ortho,
                        dtype=np.uint8
                    ).crop(
                        add_margins(
                            building.bbox,
                            ortho_res,
                            margins
                        )
                    ),
                    masks[ortho_res]
                )
                for ortho, (ortho_bbox, ortho_res) in ortho_infos.items()
                if GeoRaster.overlap(
                    add_margins(
                        building.bbox,
                        ortho_res,
                        margins
                    ),
                    ortho_bbox
                )
            ]
        )
    else:
        return functools.reduce(
            lambda lhs, rhs: lhs.union(rhs),
            [
                GeoRaster.GeoRaster.from_file(
                    ortho,
                    dtype=np.uint8
                ).crop(
                    add_margins(
                        building.bbox,
                        ortho_res,
                        margins
                    )
                )
                for ortho, (ortho_bbox, ortho_res) in ortho_infos.items()
                if GeoRaster.overlap(
                    add_margins(
                        building.bbox,
                        ortho_res,
                        margins
                    ),
                    ortho_bbox
                )
            ]
        )


def brute(
    vector_dir,
    building,
    ortho_infos,
    clip=True,
    margins=(0, 0),
    **hist_parameters
):
    radio_logger.info(
        'Compute brute features for building %s using orthoimages in %s by'
        + ' %s and adding margins %s and applying histogram parameters %s',
        building,
        ortho_infos,
        'cliping' if clip else 'cropping',
        margins,
        hist_parameters
    )
    if clip:
        ortho, mask = find_building(
            GeoBuilding.GeoBuilding.from_file(
                os.path.join(
                    vector_dir,
                    building
                )
            ),
            ortho_infos,
            clip,
            margins
        )
        radio_logger.debug(
            'Found orthoimage corresponding to %s and its mask: ortho = %s and'
            + 'mask = %s',
            building,
            ortho,
            mask
        )
        return [
            np.histogram(
                np.array(
                    [
                        pixel
                        for pixel, clipped in zip(
                            ortho.image[..., channel].flatten(),
                            mask.image[..., channel].flatten()
                        )
                        if clipped
                    ]
                ),
                **hist_parameters
            )[0]
            for channel in range(ortho.shape[-1])
        ]
    else:
        ortho = find_building(
            GeoBuilding.GeoBuilding.from_file(
                os.path.join(
                    vector_dir,
                    building
                )
            ),
            ortho_infos,
            clip,
            margins
        )
        radio_logger.debug(
            'Found orthoimage corresponding to %s: ortho = %s',
            building,
            ortho
        )
        return [
            np.histogram(
                ortho.image[..., channel].flatten(),
                **hist_parameters
            )[0]
            for channel in range(ortho.shape[-1])
        ]


def gradient(
    vector_dir,
    building,
    ortho_infos,
    normalized=False,
    resolution=10
):
    radio_logger.info(
        'Compute gradient features for building %s using orthoimages in %s '
        + ' with %s normalization and histogram resolution %s',
        building,
        ortho_infos,
        '' if normalized else 'no',
        resolution
    )
    print(building)
    vector_building = GeoBuilding.GeoBuilding.from_file(
        os.path.join(
            vector_dir,
            building
        )
    )
    ortho = find_building(
        vector_building,
        ortho_infos,
        clip=False,
        margins=(10, 10)
    )
    bins = [tick/resolution for tick in range(0, resolution + 1)]
    return functools.reduce(
        operator.add,
        [
            np.concatenate(
                [
                    np.array(
                        [
                            functools.reduce(operator.add, hist)
                            for hist in zip(
                                *[
                                    np.histogram(
                                        np.abs(np.array(channel)),
                                        bins=bins,
                                        range=(-1, 1)
                                    )[0] * norm(line) / line_string.length
                                    for channel in zip(
                                        *[
                                            [
                                                correlation(
                                                    channel_gradient,
                                                    normal(line)
                                                )
                                                for channel_gradient in zip(
                                                    (
                                                        ortho.image[i + 1, j]
                                                        +
                                                        ortho.image[i - 1, j]
                                                        -
                                                        2 * ortho.image[i, j]
                                                    ),
                                                    (
                                                        ortho.image[i, j + 1]
                                                        +
                                                        ortho.image[i, j - 1]
                                                        -
                                                        2 * ortho.image[i, j]
                                                    )
                                                )
                                            ]
                                            for i, j in ortho.intersection(
                                                shapely.geometry.LineString(
                                                    line
                                                )
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                    for line in get_lines(line_string.coords[:])
                ]
            )
            for line_string in vector_building.geometry.boundary
        ]
    )


def get_method(vector_dir, ortho_infos, method, **method_args):
    radio_logger.info(
        'Getting the method %s for feature computation for buildings in %s wrt'
        + ' %s applying parameters %s',
        method,
        vector_dir,
        ortho_infos,
        method_args,
    )
    if method == 'brute':
        return lambda building: brute(
            vector_dir,
            building,
            ortho_infos,
            **method_args
        )
    elif method == 'gradient':
        return lambda building: gradient(
            vector_dir,
            building,
            ortho_infos,
            **method_args
        )
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(method)
        )


def radiometric_features(
    vector_dir,
    ortho_dir,
    ortho_ext='geotiff',
    vector_ext='shp',
    **parameters
):
    radio_logger.info(
        'Computing radiometric features for all buildings in %s with extension'
        + ' %s wrt Orthoimages in %s with extension %s applying parameters %s',
        vector_dir,
        vector_ext,
        ortho_dir,
        ortho_ext,
        parameters,
    )
    ortho_infos = {
        os.path.join(ortho_dir, ortho_name): GeoRaster.geo_info(
            os.path.join(ortho_dir, ortho_name)
        )
        for ortho_name in fnmatch.filter(
            os.listdir(ortho_dir),
            '*' + ortho_ext
        )
    }
    radio_logger.debug(
        'Bounding boxes for OrthoImages in %s with extention %s: %s',
        ortho_dir,
        ortho_ext,
        ortho_infos
    )
    method = get_method(
        vector_dir,
        ortho_infos,
        parameters['method'],
        **parameters['parameters']
    )
    return {
        os.path.splitext(building)[0]:
        np.concatenate(
            method(building),
            axis=-1
        )
        for building in fnmatch.filter(
            os.listdir(vector_dir),
            '*.' + vector_ext
        )
    }
