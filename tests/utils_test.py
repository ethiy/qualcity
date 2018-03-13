#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from qualcity import utils

import unittest


class UtilsTest(unittest.TestCase):
    """
    Test case for the 'utils' 'qualcity' submodule.
    """

    def test_median(self):
        self.assertEqual(utils.median([-5, -5, -3, -4, 0, -1]), -3.5)
        self.assertEqual(utils.median([-5, -5, -4, 0, -1]), -4.0)

    def test_mean(self):
        self.assertEqual(utils.mean((-5, -5, -3, -4, 0, -1)), -3.0)
        self.assertEqual(utils.mean([-5, -5, -4, 0, -1]), -3.0)

    def test_dict_fuse(self):
        self.assertEqual(
            utils.fuse(
                {'4': 545, '10': 641, '8': 45},
                {'4': 15, '10': 8574, '7': 9473}
            ),
            {
                '4': (545, 15),
                '8': (45, None),
                '10': (641, 8574),
                '7': (None, 9473)
            }
        )

    def test_stat_func(self):
        with self.assertRaises(KeyError):
            utils.stat(max)

        with self.assertRaises(KeyError):
            utils.stats([12, 5, 0, 11.1], ['max', 'min', 'histogram'])

        assertEqual(
            utils.stats([12, 5, 0, 11.1], 'histogram', bins=range(0, 20)),
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        )

    def test_resolve(self):
        self.assertEqual(
            str(utils.resolve('sklearn.decomposition.PCA')),
            '<class \'sklearn.decomposition.pca.PCA\'>'
        )

        with self.assertRaisesRegex(ValueError, 'Could not resolve'):
            utils.resolve('sklearn.decomp.PCA')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
