#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import operator
import functools
import math

import numpy as np

import gdal
import gdalconst

import matplotlib.pyplot as plt

geo_raster_logger = logging.getLogger(__name__)


def bounding_box(dsm_name):
    geo_raster_logger.info('Getting %s bounding_box', dsm_name)
    dataset = gdal.Open(dsm_name, gdalconst.GA_ReadOnly)
    geo_raster_logger.info('Getting geo-transform from %s', dsm_name)
    Ox, px, _, Oy, _, py = dataset.GetGeoTransform()
    geo_raster_logger.debug(
        'Geo-transform -> origin: %s, pixel resolution: %s',
        (Ox, Oy),
        (px, py)
    )
    return (
        (
            Ox,
            Oy + py * dataset.RasterYSize
        ),
        (
            Ox + px * dataset.RasterXSize,
            Oy
        )
    )


def overlap(lbb, rbb):
    geo_raster_logger.info('Overlap between %s and %s', lbb, rbb)
    return (
        max(lbb[0][0], rbb[0][0]) < min(lbb[1][0], rbb[1][0])
        and max(lbb[0][1], rbb[0][1]) < min(lbb[1][1], rbb[1][1])
    )


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

        self.size = self.image.size
        self.shape = self.image.shape
        (self.height, self.width) = self.shape[:2]

        self.bbox = (
            (
                self.reference_point[0],
                self.reference_point[1] + self.pixel_sizes[1] * self.height
            ),
            (
                self.reference_point[0] + self.pixel_sizes[0] * self.width,
                self.reference_point[1]
            )
        )

        self.dtype = self.image.dtype

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
            ) if dataset.RasterCount > 1
            else dataset.GetRasterBand(1).ReadAsArray().astype(dtype)
        )

    def clone(self):
        """
            Return a copy
            :return: the GeoRaster copy
            :rtype: GeoRaster
        """
        return GeoRaster(
            self.reference_point,
            self.pixel_sizes,
            self.image
        )

    def empty(self):
        """
            Checks if the image is empty
            :return: the predicat truthness
            :rtype: bool
        """
        return not bool(self.size)

    def __str__(self):
        return (
            'Reference point: ' + str(self.reference_point)
            + '\nPixel resolution: ' + str(self.pixel_sizes)
            + '\nImage: ' + str(self.image)
        )

    def __eq__(self, other):
        return (
            self.reference_point == other.reference_point
            and self.pixel_sizes == other.pixel_sizes
            and(self.image == other.image).all()
        )

    def __mul__(self, other):
        return self.apply(other, operator.mul)

    def __rmul__(self, other):
        return self.apply(other, lambda x, y: y * x)

    def __add__(self, other):
        return self.apply(other, operator.add)

    def __radd__(self, other):
        return self.apply(other, operator.add)

    def __sub__(self, other):
        return self.apply(other, operator.sub)

    def __rsub__(self, other):
        return self.apply(other, lambda x, y: y - x)

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
            raise TypeError('Cannot slice with {}'.format(key))
        return GeoRaster(
            (
                self.reference_point[0]
                + self.pixel_sizes[0] * col_slice.indices(self.width)[0],
                self.reference_point[1]
                + self.pixel_sizes[1] * row_slice.indices(self.height)[0]
            ),
            self.pixel_sizes,
            self.image[row_slice, col_slice]
        )

    def plot(self, **kwargs):
        """
            Plot georaster image.
        """
        plt.imshow(self.image, **kwargs)

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
                int(round((y - self.reference_point[1])/self.pixel_sizes[1])),
                int(round((x - self.reference_point[0])/self.pixel_sizes[0]))
            )
            for x, y in bbox
        ]

    def crop(self, bbox, margins=(0, 0)):
        """
            Crop the corresponding matrix to the bounding box and the defined
            margins.
            :param bbox: bounding box
            :type bbox: list
            :param margins: crop margins
            :type margins: tuple
            :return: croped GeoRaster
            :rtype: GeoRaster
        """
        (i_max, j_min), (i_min, j_max) = self.get_slice(bbox)
        imar, jmar = margins
        return self[
            max(i_min - imar, 0): max(i_max + imar, 0),
            max(j_min - jmar, 0): max(j_max + jmar, 0)
        ]

    def union(self, other):
        return self.apply(other, lambda x, y: y)

    def apply(self, other, func, nan=0):
        if not isinstance(other, GeoRaster):
            return GeoRaster(
                self.reference_point,
                self.pixel_sizes,
                func(self.image, other)
            )
        else:
            if self.pixel_sizes != other.pixel_sizes:
                raise NotImplementedError(
                    'Multiresolution raster union is not yet implemented!'
                )
            if self.dtype != other.dtype:
                raise TypeError('Cannot merge two different dtype images')
            if self.shape[2:] != other.shape[2:]:
                raise ValueError('Operands could not be broadcast together')

            smin, smax = self.bbox
            omin, omax = other.bbox
            x_min, y_min = [min(s, o) for s, o in zip(smin, omin)]
            x_max, y_max = [max(s, o) for s, o in zip(smax, omax)]

            result = GeoRaster(
                (x_min, y_max),
                self.pixel_sizes,
                np.full(
                    tuple(
                        [
                            int(round((y_min - y_max)/self.pixel_sizes[1])),
                            int(round((x_max - x_min)/self.pixel_sizes[0]))
                        ]
                        + list(self.shape[2:])
                    ),
                    nan,
                    dtype=self.dtype
                )
            )

            (i_max, j_min), (i_min, j_max) = result.get_slice(
                self.bbox
            )
            result.image[i_min: i_max, j_min: j_max] = self.image[
                :i_max - i_min,
                :j_max - j_min
            ]

            (i_max, j_min), (i_min, j_max) = result.get_slice(
                other.bbox
            )
            result.image[i_min: i_max, j_min: j_max] = func(
                result.image[i_min: i_max, j_min: j_max],
                other.image[:i_max - i_min, :j_max - j_min]
            )
            return result


def main():
    ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
    dsm_dir = '/home/ethiy/Data/Elancourt/DSM'
    raster_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704',
        'export-3DS/rasters'
    )
    dsms = fnmatch.filter(
        os.listdir(dsm_dir),
        '*.geotiff'
    )
    rasters = fnmatch.filter(
        os.listdir(raster_dir),
        '*.tiff'
    )
    # sample = GeoRaster.from_file(os.path.join(ortho_dir, image_names[0]))
    # # sample.image = sample.image[:, :, 1]
    # print(sample)
    # print(sample.bbox)
    # print(sample.reference_point)
    # plt.figure()
    # cropped = sample[:500, 100:]
    # print(cropped.reference_point)
    # print(cropped.shape)
    # print(cropped.bbox)
    # cropped.plot()
    #
    # plt.figure()
    # _cropped = sample.crop(cropped.bbox)
    # print(_cropped == cropped)
    # _cropped.plot()
    #
    # addition = sample[:, 2500:] + sample[2500:, :2500] + sample[:2500, :2500]
    # print(addition.bbox)
    # print(addition == sample)
    # plt.figure()
    # addition.plot()
    # plt.show()

    # print(rasters[0])
    raster = GeoRaster.from_file(
        os.path.join(raster_dir, '20466.tiff'),
        dtype=np.float
    )
    print(raster.bbox)
    l = []
    d = []
    for dsm in dsms:
        if overlap(
            bounding_box(os.path.join(dsm_dir, dsm)),
            bounding_box(os.path.join(raster_dir, '20466.tiff'))
        ):
            l.append(
                GeoRaster.from_file(
                    os.path.join(dsm_dir, dsm),
                    dtype=np.float
                ).crop(
                    raster.bbox
                )
            )
        # # print(dsm)
        # # print(overlap(raster.bbox, _dsm.bbox))
        # crop = _dsm.crop(
        #     raster.bbox
        # )
        # print(crop.shape)
        # if crop.size:
        #     l.append(crop)
        #     d.append(crop.bbox)
    # print(d)
    im = l[0] + -1 * l[1]
    print(im.shape)
    print(l[0] + -1 * l[1] == l[0] - l[1])
    plt.figure()
    im.plot(cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
