#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import unittest

import os
import fnmatch

import numpy as np

from matplotlib.image import imread, imsave

from qualcity import radiometric_features as radiof
from qualcity import GeoBuilding
from qualcity import GeoRaster


class GeoRadiometricTest(unittest.TestCase):
    """
    Test case for the 'radiometric_features' 'qualcity' submodule.
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
        self.ortho_infos = {
            os.path.join(self.ortho_dir, ortho_name): GeoRaster.geo_info(
                os.path.join(self.ortho_dir, ortho_name)
            )
            for ortho_name in fnmatch.filter(
                os.listdir(self.ortho_dir),
                '*.geotiff'
            )
        }

    def test_clip_building(self):
        ortho, mask = radiof.find_building(
            self.building,
            self.ortho_infos,
            clip=True,
            margins=(0, 0)
        )
        self.assertFalse(
            (
                (ortho * mask).image
                -
                (
                    255 * imread(
                        os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            '../ressources/tests/building_20466_clip.png'
                        )
                    )[:, :, :3]
                ).astype(np.uint8)
            ).all()
        )

    def test_crop_building(self):
        self.assertFalse(
            (
                radiof.find_building(
                    self.building,
                    self.ortho_infos,
                    clip=False,
                    margins=(0, 0)
                ).image
                -
                (
                    255 * imread(
                        os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            '../ressources/tests/building_20466_crop.png'
                        )
                    )[:, :, :3]
                ).astype(np.uint8)
            ).all()
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
