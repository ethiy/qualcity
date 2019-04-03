# -*- coding: <utf-8> -*-

import os
import fnmatch

import ast

import logging

import math
import operator
import functools

import numpy as np

from tqdm import tqdm

import shapely.ops

import torch
from kymatio import Scattering2D

from . import GeoBuilding, GeoRaster
from . import utils

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
            (y0 - y1) / length,
            (x1 - x0) / length
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


def cosine_similarity(lvector, rvector):
    return (
        scalar_product(lvector, rvector) / (
            math.hypot(*lvector) * math.hypot(*rvector)
        )
        if math.hypot(*lvector) != 0 else math.inf
    )


def angle(lvector, rvector):
    return math.acos(cosine_similarity(lvector, rvector))


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

    
def image_gradient(image, line, column):
    return zip(
        (
            image[line + 1, column] + image[line - 1, column] - 2 * image[line, column]
        ),
        (
            image[line, column + 1] + image[line, column - 1] - 2 * image[line, column]
        )
    )


def distance_per_channel(vector, channels, metric):
    return [
        metric(
            channel,
            vector
        )
        for channel in channels
    ]


def discrepancies_per_channel(line, geo_image):
    return zip(
        *[
            distance_per_channel(
                normal(line),
                image_gradient(geo_image.image, i, j),
                cosine_similarity
            )
            for i, j in geo_image.intersection(
                shapely.geometry.LineString(
                    line
                )
            )
        ]
    )


def weighted_histogram(values, bins, weights, weight=(True, True, True)):
    return np.histogram(
        np.abs(np.array(values)),
        bins=bins,
        range=(-1, 1)
    )[0] * (
        (weights[0] * weight[0] + 1 - weight[0])
        /
        (weights[1] * weight[1] + 1 - weight[1])
        *
        (weights[2] * weight[2] + 1 - weight[2])
    )


def gradient(
    vector_dir,
    building,
    ortho_infos,
    weight=(True, True, True),
    resolution=10
):
    radio_logger.info(
        'Compute gradient features for building %s using orthoimages in %s '
        + ' with %s weighting and histogram resolution %s',
        building,
        ortho_infos,
        '' if sum(weight) else 'no',
        resolution
    )
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
        margins=(2, 2)
    )
    radio_logger.debug(
        'Cropped orthoimage corresponding to %s with margins %s: %s',
        building,
        (2, 2),
        ortho
    )
    bins = [tick/resolution for tick in range(0, resolution + 1)]
    radio_logger.debug('Histogram bins: %s', bins)
    return sum(
        [
            sum(
                [
                    np.concatenate(
                        [
                            weighted_histogram(
                                channel_discrepancies,
                                bins,
                                (
                                    norm(line),
                                    line_string.length,
                                    vector_building.area
                                ),
                                weight=(True, True, True)
                            )
                            for channel_discrepancies in discrepancies_per_channel(line, ortho)
                        ]
                    )
                    for line in get_lines(line_string.coords[:])
                ]
            )
            for line_string in vector_building.geometry.boundary
        ]
    )


def edges_pixels(ortho, building_vector):
    return [
        (i, j)
        for line_string in building_vector.geometry.boundary
        for line in get_lines(line_string.coords[:])
        for i, j in ortho.intersection(
            shapely.geometry.LineString(
                line
            )
        )
    ]


def edgeless(ortho, edges_pixels, resize=None):
    full_size = ortho.image
    full_size[list(zip(*edges_pixels))] = np.zeros(3)
    if resize is None:
        return full_size
    else:
        rrrr


def get_mask(shape, edges_pixels):
    mask = np.zeros(shape)
    mask[list(zip(*edges_pixels))] = np.full(3, 255)
    return mask

def mask_channel(ortho, edges_pixels, resize=None):
    full_size = np.dstack(
        (
            ortho.image,
            get_mask(ortho.image.shape,edges_pixels)
        )
    )
    if resize is None:
        return full_size
    else:
        rrrr


def pooler(pooling_functions):
    return lambda x : np.array(
        [
            utils.resolve(function)(
                x,
                axis=(3, 4)
            ).flatten()
            for function in pooling_functions
        ]
    ).flatten('F')


def scatter(
    vector_dir,
    building,
    ortho_infos,
    fusion,
    resize=None,
    pooling=None,
    clip=False,
    J=3
):
    vector_building = GeoBuilding.GeoBuilding.from_file(
        os.path.join(
            vector_dir,
            building
        )
    )
    ortho = find_building(
        vector_building,
        ortho_infos,
        clip=clip,
        margins=(2, 2)
    )
    if fusion == 'deletion':
        proxy = torch.unsqueeze(
            torch.from_numpy(
                np.moveaxis(
                    edgeless(
                        ortho,
                        edges_pixels(ortho, vector_building),
                        resize
                    ),
                    -1,
                    0
                )
            ),
            0
        ).float()
    elif fusion == 'channel':
        proxy = torch.unsqueeze(
            torch.from_numpy(
                np.moveaxis(
                    mask_channel(
                        ortho,
                        edges_pixels(ortho, vector_building),
                        resize
                    ),
                    -1,
                    0
                )
            ),
            0
        ).float()
    else:
        raise NotImplementedError('{} not implemented'.format(fusion))
    try:
        scattered = Scattering2D(J=J, shape=tuple(proxy.size()[2:])).cuda().forward(proxy.cuda()).cpu().numpy()
    except RuntimeError:
        scattered = Scattering2D(J=J, shape=tuple(proxy.size()[2:])).forward(proxy).cpu().numpy()
    extractor = pooler(pooling) if pooling is not None else lambda x: x
    return extractor(scattered)


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
    elif method == 'scattering':
        return lambda building: scatter(
            vector_dir,
            building,
            ortho_infos,
            **method_args
        )
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(method)
        )


def compute_features(buildings, cache_dir, cache_args, vector_dir, vector_ext, ortho_infos, method, **method_args):
    cache_args.update(
        {
            'vector_dir': vector_dir,
            'vector_ext': vector_ext,
            'method': method,
            'parameters': method_args
        }
    )
    cached_features = utils.fetch_features(
        buildings,
        'radiometric',
        cache_dir,
        **cache_args
    )

    features = {
        building:
        np.concatenate(
            get_method(
                vector_dir,
                ortho_infos,
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
            desc='Radiometric features using ' + method
        )
    }
    utils.cache_features(
        cache_dir,
        'radiometric',
        cache_args,
        features
    )
    cached_features.update(features)
    return cached_features


def radiometric_features(
    buildings,
    cache_dir,
    vector_dir,
    ortho_dir,
    ortho_ext='geotiff',
    vector_ext='shp',
    **parameters
):
    radio_logger.info(
        'Computing radiometric features for buildings %s in %s with extension'
        + ' %s wrt Orthoimages in %s with extension %s applying parameters %s',
        buildings,
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
    return {
        method['method']: 
        compute_features(
            buildings,
            cache_dir,
            {
                'ortho_dir': ortho_dir,
                'ortho_ext': ortho_ext
            },
            vector_dir,
            vector_ext,
            ortho_infos,
            method['method'],
            **method['parameters'] if 'parameters' in method.keys() else {},
        )
        for method in parameters['methods']
    }
