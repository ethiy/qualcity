#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

import os
import fnmatch

import logging

import functools
import operator

import numpy as np

import georasters as gr

import matplotlib.pyplot as plt

ortho_logger = logging.getLogger(__name__)


def merge_orthos(ortho_dir, ext='*.geotiff', jobs=4):
    images = [
        gr.from_file(os.path.join(ortho_dir, image_name))
        for image_name in fnmatch.filter(
            os.listdir(ortho_dir),
            ext
        )
    ]
    print(images[0].geot)
    return functools.reduce(
        lambda limage, rimage: limage.union(rimage),
        images
    )


def main():
    ortho_dir = '/home/ethiy/Data/Elancourt/OrthoImages'
    figure = plt.figure()
    merge_orthos(ortho_dir, ext='*.geotiff').plot()
    plt.show()


if __name__ == '__main__':
    main()
