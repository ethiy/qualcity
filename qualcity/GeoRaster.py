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

    def __eq__(self, other):
        return (
            self.reference_point == other.reference_point
            and self.pixel_sizes == other.pixel_sizes
            and(self.image == other.image).all()
        )

    def __add__(self, other):
        if not self.size():
            return other
        elif not other.size():
            return self
        else:
            (x_min, x_max), (y_min, y_max) = [
                (min(collec), max(collec))
                for collec in [
                    [item for subtup in dim for item in subtup]
                    for dim in zip(zip(*self.bbox()), zip(*other.bbox()))
                ]
            ]
            if self.pixel_sizes != other.pixel_sizes:
                raise NotImplementedError(
                    'Multiresolution raster union is not yet implemented!'
                )
            elif self.dtype() != other.dtype():
                raise TypeError('Cannot merge two different dtype images')
            elif self.shape()[2:] != other.shape()[2:]:
                raise ValueError('Operands could not be broadcast together')
            else:
                result = GeoRaster(
                    (x_min, y_max),
                    self.pixel_sizes,
                    np.full(
                        tuple(
                            [
                                int((x_max - x_min)/self.pixel_sizes[0]),
                                int((y_min - y_max)/self.pixel_sizes[1])
                            ]
                            + list(self.shape()[2:])
                        ),
                        -np.inf,
                        dtype=self.dtype()
                    )
                )

                for raster in [self, other]:
                    (i_max, j_min), (i_min, j_max) = result.get_slice(
                        raster.bbox()
                    )
                    result.image[i_min: i_max, j_min: j_max] = raster.image

                return result

    def __getitem__(self, key):
        """
            Create GeoRaster `cls` from slice `slice` in `georaster`.
            :param key: key slice
            :type key: slice
            :return: the sliced georaster
            :rtype: GeoRaster
        """
        try:
            row_slice, col_slice = key
        except TypeError:
            row_slice = key
            col_slice = slice(None)
        if (
            not isinstance(row_slice, slice)
            or not isinstance(col_slice, slice)
        ):
            raise TypeError('%s is not a slice', key)
        return GeoRaster(
            (
                self.reference_point[0]
                + self.pixel_sizes[0] * col_slice.indices(self.width())[0],
                self.reference_point[1]
                + self.pixel_sizes[1] * row_slice.indices(self.height())[0]
            ),
            self.pixel_sizes,
            self.image[row_slice, col_slice]
        )

    def dtype(self):
        """
            Get image dtype.
            :return: image dtype.
            :rtype: type
        """
        return self.image.dtype

    def size(self):
        """
            Get image size.
            :return: image size.
            :rtype: int
        """
        return self.image.size

    def shape(self):
        """
            Get image shape.
            :return: image shape.
            :rtype: tuple
        """
        return self.image.shape

    def height(self):
        """
            Get image height.
            :return: image height.
            :rtype: int
        """
        return self.image.shape[0]

    def width(self):
        """
            Get image width.
            :return: image width.
            :rtype: int
        """
        return self.image.shape[1]

    def plot(self, **kwargs):
        """
            Plot georaster image.
        """
        plt.imshow(self.image, **kwargs)

    def bbox(self):
        """
            Get crop points in coordinates in Georaster.
            :return: bounding box
            :rtype: list
        """
        return (
            (
                self.reference_point[0],
                self.reference_point[1] + self.pixel_sizes[1] * self.height()
            ),
            (
                self.reference_point[0] + self.pixel_sizes[0] * self.width(),
                self.reference_point[1]
            )
        )

    def get_slice(self, bbox):
        """
            Get crop points in coordinates in image.
            :param bbox: bounding box
            :type bbox: list
            :return: extremal points defining the crop region
            :rtype: list
        """
        return [
            (
                int((y - self.reference_point[1])/self.pixel_sizes[1]),
                int((x - self.reference_point[0])/self.pixel_sizes[0])
            )
            for x, y in bbox
        ]

    def crop(self, bbox):
        """
            Crop the corresponding matrix to the bounding box and the defined
            margins.
            :param bbox: bounding box
            :type bbox: list
            :return: croped GeoRaster
            :rtype: GeoRaster
        """
        (i_max, j_min), (i_min, j_max) = self.get_slice(bbox)
        return self[i_min: i_max, j_min:j_max]


def main():
    ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
    image_names = fnmatch.filter(
        os.listdir(ortho_dir),
        '*.geotiff'
    )
    sample = GeoRaster.from_file(os.path.join(ortho_dir, image_names[0]))
    print(sample.bbox())
    print(sample.reference_point)
    plt.figure()
    cropped = sample[:500, 100:]
    print(cropped.reference_point)
    print(cropped.shape())
    print(cropped.bbox())
    cropped.plot()

    plt.figure()
    _cropped = sample.crop(cropped.bbox())
    print(_cropped == cropped)
    _cropped.plot()

    addition = sample[:, 2500:] + sample[2500:, :2500] + sample[:2500, :2500]
    print(addition.bbox())
    print(addition == sample)
    plt.figure()
    addition.plot()
    plt.show()


if __name__ == '__main__':
    main()
