#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from city import GeoBuilding

import unittest

import os
import fnmatch

import numpy as np


class GeoBuildingTest(unittest.TestCase):
    """
    Test case for the 'GeoBuilding' 'city' submodule.
    """

    def setUp(self):
        self.vector_dir = os.path.join(
            '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
            'export-3DS/_labels'
        )
        self.building = GeoBuilding.GeoBuilding.from_file(
            os.path.join(
                self.vector_dir,
                '20466.shp'
            )
        )

    def test_read(self):
        self.assertEqual(
            self.building.bbox,
            (
                (623485.300003052, 6851964.64001465),
                (623499.86000061, 6851975.91999817)
            )
        )
        self.assertEqual(
            len(self.building),
            12
        )

    def test_rasterization(self):
        mask = self.building.rasterize(
            (0.06, -0.06),
        )
        self.assertFalse(
            (
                mask.image - np.genfromtxt(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        '../ressources/tests/building_20466_rasterized.out'
                    ),
                    dtype=int
                )
            ).all()
        )
        self.assertEqual(
            mask.bbox,
            (
                (623485.300003052, 6851964.63999817),
                (623499.880003052, 6851975.91999817)
            )
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
