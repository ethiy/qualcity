#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import fnmatch
import os

import shapefile

SHAPEFILE = "ESRI Shapefile"


def read(filename):
    return set([feature[-1] for feature in shapefile.Reader(filename).records()])


def main():
    directory = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/labels'
    print [read(filename) for filename in fnmatch.filter([os.path.join(directory, file) for file in os.listdir(directory)], '*.shp')]


if __name__ == '__main__':
    main()
