#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import os
import fnmatch

import shapefile

SHAPEFILE = "ESRI Shapefile"


def read(filename):
    return [(feature[-3], feature[-2], feature[-1]) for feature in shapefile.Reader(filename).records()]


def main():
    labels_dir = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/_labels'
    files = fnmatch.filter(os.listdir(labels_dir), '*.shp')
    print {os.path.splitext(f)[0]: read(os.path.join(labels_dir, f)) for f in files}


if __name__ == '__main__':
    main()
