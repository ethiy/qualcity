#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import pathos.multiprocessing as mp

import shapefile
import shapely.geometry

import numpy as np

from qualcity import utils, GeoRaster

geo_vector_logger = logging.getLogger(__name__)


class GeoBuilding:
    """
        Geographic raster.
        Attribute `reference_point` stores the reference point.
        Attribute `pixel_sizes` stores the horizontal and vertical resolutions.
        Attribute `image` stores the matrix image.
    """

    def __init__(self, geometry):
        """
            Initiate GeoBuilding class.
            :param :
            :type geometry:
        """
        self.geometry = geometry
        self.bbox = tuple(utils.chunk(self.geometry.bounds, 2))

    @classmethod
    def from_file(cls, filename):
        """
            Create GeoBuilding `cls` from file in `filname`.
            :param filename: file path
            :type filename: string
            :return: cls
            :rtype: GeoBuilding
        """
        return cls(
            shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.shape(shp)
                    for shp in shapefile.Reader(filename).shapes()
                ]
            )
        )

    def __str__(self):
        return (
            'bounding_box: ' + str(self.bbox)
            + '\nGeometry: ' + str(self.geometry)
        )

    def __len__(self):
        return len(self.geometry.geoms)

    def rasterize(self, pixel_sizes, dtype=bool, jobs=8, channels=1):
        pool = mp.Pool(8)
        mask = np.array(
            [
                pool.map(
                    lambda w: self.geometry.contains(
                        shapely.geometry.Point(
                            self.bbox[0][0] + pixel_sizes[0] * (w + .5),
                            self.bbox[1][1] + pixel_sizes[1] * (h + .5),
                        )
                    ),
                    range(
                        int(
                            round(
                                (self.bbox[1][0] - self.bbox[0][0])
                                /
                                pixel_sizes[0]
                            )
                        )
                    )
                )
                for h in range(
                    int(
                        round(
                            (self.bbox[0][1] - self.bbox[1][1])
                            /
                            pixel_sizes[1]
                        )
                    )
                )
            ],
            dtype=dtype
        )
        return GeoRaster.GeoRaster(
            (self.bbox[0][0], self.bbox[1][1]),
            pixel_sizes,
            mask if channels == 1 else np.stack((mask,)*channels, -1)
        )
