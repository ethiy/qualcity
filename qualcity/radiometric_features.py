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


def find_building(building, ortho_dir, ext, clip=True):
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
                dtype=np.uint8,
                channels=3
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
    return functools.reduce(
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
