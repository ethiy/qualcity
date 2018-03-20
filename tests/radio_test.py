#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import unittest

import os
import fnmatch

import numpy as np

import matplotlib.pyplot as plt

from qualcity import radiometric_features as radiof
from qualcity import GeoBuilding


class GeoRadiometricTest(unittest.TestCase):
    """
    Test case for the 'GeoBuilding' 'qualcity' submodule.
    """

    def setUp(self):
        self.vector_dir = os.path.join(
            '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
            'export-3DS/_labels'
        )
        self.ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
        self.building = GeoBuilding.GeoBuilding.from_file(
            os.path.join(
                self.vector_dir,
                '20466.shp'
            )
        )

    def test_find_building(self):
        building_ortho = radiof.find_building(
            self.building,
            self.ortho_dir,
            '*.geotiff'
        )
        self.assertFalse(
            (
                building_ortho.image - (255 * plt.imread(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        '../ressources/tests/building_20466_clip.png'
                    )
                )[:, :, :3]).astype(np.uint8)
            ).all()
        )
        self.assertEqual(
            building_ortho.bbox,
            (
                (623485.300003052, 6851964.71999817),
                (623499.900003052, 6851975.91999817)
            )
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
