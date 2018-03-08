#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import functools
import math

import numpy as np

import gdal
import gdalconst

import matplotlib.pyplot as plt

raster_utils_logger = logging.getLogger(__name__)


class GeoRaster:
    """
        Geographic raster.
        Attribute `reference_point` stores the reference point.
        Attribute `pixel_sizes` stores the horizontal and vertical resolutions.
        Attribute `image` stores the matrix image.
    """

    def __init__(self, reference_point, pixel_sizes, image):
        """
            Initiate GeoRaster class.
            :param reference_point: reference point
            :type reference_point: tuple
            :param pixel_sizes: pixel resolutions
            :type pixel_sizes: tuple
            :param image: 3d image matrix
            :type image: np.array
        """
        self.reference_point = reference_point
        self.pixel_sizes = pixel_sizes
        self.image = image

    @classmethod
    def from_file(cls, filename, dtype=np.uint8):
        """
            Create GeoRaster `cls` from file in `filname`.
            :param filename: file path
            :type filename: string
            :param dtype: depth type
            :type dtype: type
            :return: cls
            :rtype: GeoRaster
        """
        dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
        Ox, px, _, Oy, _, py = dataset.GetGeoTransform()
        return cls(
            (Ox, Oy),
            (px, py),
            np.dstack(
                [
                    dataset.GetRasterBand(band).ReadAsArray().astype(dtype)
                    for band in range(1, dataset.RasterCount + 1)
                ]
            )
        )

    def slice(self, row_slice, col_slice):
        """
            Create GeoRaster `cls` from slice `slice` in `georaster`.
            :param row_slice: file path
            :type row_slice: slice
            :param col_slice: file path
            :type col_slice: slice
            :return: the sliced georaster
            :rtype: GeoRaster
        """
        return GeoRaster(
            tuple(
                [
                    O + p * dim
                    for O, p, dim in zip(
                        self.reference_point,
                        self.pixel_sizes,
                        [
                            row_slice.indices(self.height())[0],
                            col_slice.indices(self.width())[0]
                        ]
                    )
                ]
            ),
            self.pixel_sizes,
            self.image[row_slice, col_slice]
        )

    def dtype(self):
        """
            Get image dtype.
        """
        return self.image.dtype

    def shape(self):
        """
            Get image shape.
        """
        return self.image.shape

    def height(self):
        """
            Get image height.
        """
        return self.image.shape[0]

    def width(self):
        """
            Get image width.
        """
        return self.image.shape[1]

    def plot(self, **kwargs):
        """
            Plot georaster image.
        """
        plt.imshow(self.image, **kwargs)

    @functools.lru_cache()
    def bbox(self):
        """
            Get crop points in coordinates in Georaster.
            :return: bounding box
            :rtype: list
        """
        return (
            self.reference_point,
            tuple(
                [
                    O + p * dim
                    for O, p, dim in zip(
                        self.reference_point,
                        self.pixel_sizes,
                        self.image.shape[:2]
                    )
                ]
            )
        )

    def crop_slice(self, georaster):
        """
            Get crop points in coordinates in image.
            :param georaster: geo raster
            :type georaster: GeoRaster
            :return: extremal points defining the crop region
            :rtype: list
        """
        bbox = georaster.bbox()
        return [
            (
                (y - self.reference_point[1])/self.pixel_sizes[1],
                (x - self.reference_point[0])/self.pixel_sizes[0]
            )
            for x, y in bbox
        ]

    def crop(self, georaster):
        """
            Crop the corresponding matrix to the bounding box and the defined
            margins.
            :param georaster: geo raster
            :type georaster: GeoRaster
            :return: croped GeoRaster
            :rtype: GeoRaster
        """
        (i_min, j_min), (i_max, j_max) = self.crop_slice(georaster)
        (pi_min, pi_max), (pj_min, pj_max) = [
            (
                max(math.floor(l_min), 0),
                min(math.ceil(l_max), self.image.shape[0])
            )
            for l_min, l_max in [(i_min, i_max), (j_min, j_max)]
        ]
        if pi_min >= pi_max or pj_min >= pj_max:
            return GeoRaster(
                georaster.bbox()[0],
                self.pixel_sizes,
                np.array(0, dtype=self.image.dtype, ndmin=len(self.shape()))
            )
        else:
            return GeoRaster(
                georaster.bbox()[0],
                self.pixel_sizes,
                self.image[pi_min: pi_max, pj_min: pj_max, :]
            )


def main():
    ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
    image_names = fnmatch.filter(
        os.listdir(ortho_dir),
        '*.geotiff'
    )
    sample = GeoRaster.from_file(os.path.join(ortho_dir, image_names[0]))
    print(sample.reference_point, sample.pixel_sizes, sample.image.shape)
    print(sample.bbox())
    plt.figure()
    cropped = sample.slice(
        slice(sample.height() * 2, None),
        slice(sample.width()//2, None)
    )
    print(cropped.reference_point, cropped.bbox(), cropped.shape())
    cropped.plot()

    plt.figure()
    sample.crop(
        sample.slice(
            slice(None, None),
            slice(sample.width()//2, None)
        )
    ).plot()

    plt.show()


if __name__ == '__main__':
    main()
