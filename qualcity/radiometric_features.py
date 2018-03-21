# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

import numpy as np

from qualcity import GeoBuilding, GeoRaster

radio_logger = logging.getLogger(__name__)


def process_building(building, ortho_dir, ext, func, clip=True):
    radio_logger.info(
        (
            'Cliping' if clip else 'Cropping'
            + ' %s corresponding DSM in %s with extention %s'
        ),
        building,
        ortho_dir,
        ext,
        clip
    )
    orthos = fnmatch.filter(
        os.listdir(ortho_dir),
        ext
    )
    radio_logger.debug('Orthoimage names in %s: %s', ortho_dir, orthos)
    masks = {}
    if clip:
        radio_logger.debug(
            'All rasterized masks corresponding to building: %s',
            masks
        )
        masks = {
            res: building.rasterize(
                res,
                dtype=np.uint8
            ).apply(
                lambda mask: np.stack((mask, ) * 3, -1),
                vectorize=False
            )
            for res in set(
                [
                    GeoRaster.resolution(
                        os.path.join(ortho_dir, ortho)
                    )
                    for ortho in orthos
                ]
            )
        }
    return func(
        *functools.reduce(
            lambda lhs, rhs: (
                lhs[0].union(rhs[0]),
                lhs[1].union(rhs[1]) if clip else None
            ),
            [
                (
                    (
                        GeoRaster.GeoRaster.from_file(
                            os.path.join(ortho_dir, ortho),
                            dtype=np.uint8
                        ).crop(building.bbox),
                        masks[
                            GeoRaster.resolution(
                                os.path.join(ortho_dir, ortho)
                            )
                        ] if clip else None
                    )
                )
                for ortho in orthos
                if GeoRaster.overlap(
                    building.bbox,
                    GeoRaster.bounding_box(
                        os.path.join(ortho_dir, ortho)
                    )
                )
            ]
        )
    )


def histogram(ortho, mask, **hist_parameters):
    if mask is None:
        return [
            np.histogram(
                ortho.image[..., channel].flatten(),
                **hist_parameters
            )[0]
            for channel in range(ortho.shape[-1])
        ]
    else:
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


def histogram_features(
    vector_dir,
    ortho_dir,
    ext='geotiff',
    vector_ext='shp',
    clip=True,
    **parameters
):
    radio_logger.info(
        'Computing histogram features for all buildings in %s with extension '
        + '%s wrt Orthos in %s with extension %s with parameters %s',
        vector_dir,
        vector_ext,
        ortho_dir,
        ext,
        parameters
    )

    return {
        os.path.splitext(building)[0]:
        np.concatenate(
            process_building(
                GeoBuilding.GeoBuilding.from_file(
                    os.path.join(
                        vector_dir,
                        building
                    )
                ),
                ortho_dir,
                '*.' + ext,
                clip=clip,
                func=lambda x, mask: histogram(x, mask, **parameters)
            ),
            axis=-1
        )
        for building in fnmatch.filter(
            os.listdir(vector_dir),
            '*.' + vector_ext
        )
    }
