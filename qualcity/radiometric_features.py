# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import math
import operator
import functools

import numpy as np

from .GeoRaster import overlap, bounding_box, GeoRaster

ortho_logger = logging.getLogger(__name__)


def ():
    pass


def main():
    ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
    figure = plt.figure()
    merge_orthos(ortho_dir, ext='*.geotiff').plot()
    plt.show()


if __name__ == '__main__':
    main()
