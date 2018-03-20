#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from qualcity import GeoRaster

import unittest

import os
import fnmatch

import numpy as np


class GeoRasterTest(unittest.TestCase):
    """
    Test case for the 'GeoRaster' 'qualcity' submodule.
    """

    def setUp(self):
        self.dsm_dir = '/home/ethiy/Data/Elancourt/DSM'
        self.ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
        self.raster_dir = os.path.join(
            '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
            'export-3DS/rasters'
        )
        dsms = fnmatch.filter(
            os.listdir(self.dsm_dir),
            '*.geotiff'
        )

        self.dsm = GeoRaster.GeoRaster.from_file(
            os.path.join(self.raster_dir, '20466.tiff'),
            dtype=np.float
        )
        self.ortho = GeoRaster.GeoRaster.from_file(
            os.path.join(
                self.ortho_dir,
                '78-2014-0621-6851-LA93-0M20-E100.geotiff'
            ),
            dtype=np.uint8
        )

    def test_overlap(self):
        self.assertFalse(GeoRaster.overlap(self.ortho.bbox, self.dsm.bbox))
        with self.assertRaisesRegex(TypeError, 'Cannot slice with'):
            GeoRaster.overlap(self.ortho, self.dsm)

    def test_bounding_box(self):
        self.assertEqual(
            GeoRaster.bounding_box(
                os.path.join(self.raster_dir, '20466.tiff')
            ),
            self.dsm.bbox
        )

    def test_dsm_reading(self):
        self.assertEqual(
            self.dsm.reference_point,
            (623485.3000030518, 6851975.919998169)
        )
        self.assertEqual(
            self.dsm.shape,
            (188, 243)
        )
        self.assertEqual(
            self.dsm.bbox,
            (
                (623485.3000030518, 6851964.639998169),
                (623499.8800030517, 6851975.919998169)
            )
        )

    def test_ortho_reading(self):
        self.assertEqual(
            self.ortho.reference_point,
            (621000.0, 6851000.0)
        )
        self.assertEqual(
            self.ortho.shape,
            (5000, 5000, 3)
        )
        self.assertEqual(
            self.ortho.bbox,
            ((621000.0, 6850000.0), (622000.0, 6851000.0))
        )

    def test_slicing(self):
        sliced = self.ortho[:500, 128:]
        self.assertEqual(
            sliced.reference_point,
            (621025.6, 6851000.0)
        )
        self.assertEqual(
            sliced.shape,
            (500, 4872, 3)
        )
        self.assertEqual(
            sliced.bbox,
            ((621025.6, 6850900.0), (622000.0, 6851000.0))
        )

    def test_applying(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            'Multiresolution raster union is not yet implemented!'
        ):
            self.dsm.apply(self.ortho, lambda x, y: max(x, y))

        self.assertEqual(
            self.dsm + self.dsm,
            2 * self.dsm
        )
        self.assertEqual(
            self.dsm[40:, 85:].union(self.dsm[:45]).union(self.dsm[35:, :91]),
            self.dsm
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
