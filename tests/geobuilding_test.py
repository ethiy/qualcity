#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from qualcity import GeoBuilding

import unittest

import os
import fnmatch


class GeoBuildingTest(unittest.TestCase):
    """
    Test case for the 'GeoBuilding' 'qualcity' submodule.
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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
