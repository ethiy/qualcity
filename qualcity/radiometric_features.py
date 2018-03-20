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


def find_building(building, ortho_dir, ext):
    radio_logger.info(
        'Getting %s corresponding DSM in %s with extention %s',
        building,
        ortho_dir,
        ext
    )
    orthos = fnmatch.filter(
        os.listdir(ortho_dir),
        ext
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
        lambda lhs, rhs: lhs.union(rhs),
        [
            GeoRaster.GeoRaster.from_file(
                os.path.join(ortho_dir, ortho),
                dtype=np.uint8
            ).crop(building.bbox) * masks[
                GeoRaster.resolution(
                    os.path.join(ortho_dir, ortho)
                )
            ]
            for ortho in orthos
            if GeoRaster.overlap(
                building.bbox,
                GeoRaster.bounding_box(
                    os.path.join(ortho_dir, ortho)
                )
            )
        ]
    )
