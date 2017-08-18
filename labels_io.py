#! /usr/bin/env python
# -*- coding: <utf-8> -*-

import shapefile

SHAPEFILE = "ESRI Shapefile"


def read(filename):
    return [feature[-1] for feature in shapefile.Reader(filename).records()]
