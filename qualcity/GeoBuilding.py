#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import shapefile
import shapely.geometry

import matplotlib.pyplot as plt

from qualcity import utils

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
