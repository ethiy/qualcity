#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import shapefile

import shapely.geometry

import numpy as np
import matplotlib.pyplot as plt

from . import utils, GeoRaster

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
        self.area = self.geometry.area

    @classmethod
    def from_file(cls, filename):
        """
            Create GeoBuilding `cls` from file in `filname`.
            :param filename: file path
            :type filename: string
            :return: cls
            :rtype: GeoBuilding
        """
        sr = shapefile.Reader(filename)
        facets = sr.shapes()
        sr.shp.close() if sr.shp else None
        sr.shx.close() if sr.shx else None
        sr.dbf.close() if sr.dbf else None
        return cls(
            shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.shape(shp)
                    for shp in facets
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

    def plot(self, resolution, margins, georeference=False, **kwargs):
        for line_string in self.geometry.boundary:
            plt.plot(
                *[
                    [
                        (
                            value - reference * int(not georeference)
                        ) * res + margin
                        for value in axe
                    ]
                    for res, axe, reference, margin in zip(
                        resolution,
                        line_string.xy,
                        [self.bbox[0][0], self.bbox[1][1]],
                        margins
                    )
                ],
                **kwargs
            )

    def rasterize(self, pixel_sizes, dtype=bool):
        mask = np.array(
            [
                [
                    self.geometry.contains(
                        shapely.geometry.Point(
                            self.bbox[0][0] + pixel_sizes[0] * (w + .5),
                            self.bbox[1][1] + pixel_sizes[1] * (h + .5),
                        )
                    )
                    for w in range(
                        int(
                            round(
                                (self.bbox[1][0] - self.bbox[0][0])
                                /
                                pixel_sizes[0]
                            )
                        )
                    )
                ]
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
            mask
        )
