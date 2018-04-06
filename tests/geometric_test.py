#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import unittest

import os
import fnmatch

import operator

import numpy as np

from qualcity import geometric_features as geo


class GeoGeometricTest(unittest.TestCase):
    """
    Test case for the 'geometric_features' 'qualcity' submodule.
    """

    def setUp(self):
        self.graph_dir = os.path.join(
            '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS',
            'dual_graphs'
        )

    def test_graph_reading(self):
        gt = {
            'area': 139.392,
            'degree': 4,
            'normal': np.array([0.904114, 0.427291, -0]),
            'centroid': np.array([43.3478, 367.755, -9.67999])
        }
        for tkey, tvalue in geo.read(
            os.path.join(self.graph_dir, '3078.txt')
        ).node[1].items():
            if isinstance(tvalue, np.ndarray):
                self.assertFalse(
                    (tvalue - gt[tkey]).all()
                )
            else:
                self.assertEqual(tvalue, gt[tkey])

    def test_graph_features(self):
        self.assertEqual(
            geo.features(
                os.path.join(self.graph_dir, '3078.txt'),
                ['degree', 'area', 'centroid'],
                ['min', 'max']
            ),
            [
                5,
                4,
                4,
                6.81227,
                182.258,
                4.1108000437870773,
                146.03189846019944
            ]
        )
        self.assertEqual(
            geo.features(
                os.path.join(self.graph_dir, '3078.txt'),
                ['degree', 'area'],
                'histogram',
                bins=10
            ),
            [5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2]
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
