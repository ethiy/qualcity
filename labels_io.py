#! /usr/bin/env python2
# -*- coding: <utf-8> -*-

from __future__ import print_function

import os
import fnmatch

import shapefile

SHAPEFILE = "ESRI Shapefile"

ERROR_DICTIONARY = {
    'over_segmentation': ['over_segmentation', 'over'],
    'under_segmentation': ['under_segmentation', 'under'],
    'None': ['None', ''],
    'half_building': ['half_building'],
    'changed': ['changed'],
    'occlusion': ['occlusion', 'vegetation']
}

UNQUALIFIED_ERROR_LIST = [
    'half_building',
    'changed',
    'occlusion'
]


def read(filename):
    records = shapefile.Reader(filename).records()
    return [(feature[-3], feature[-2], feature[-1]) for feature in records]


def errors_per_building(filename):
    return map(lambda t: list(set(t)), zip(*read(filename)))


def unqualified_errors(filename):
    return lint(errors_per_building(filename)[0])


def lint(errors):
    linted = filter(
        lambda error: error not in ERROR_DICTIONARY['None'],
        errors
    )
    return linted if linted != [] else 'None'


def unqualified_errors_statistics(error, labels_dir, files):
    unq_errors = filter(
        lambda _error: _error not in ERROR_DICTIONARY['None'],
        [unqualified_errors(os.path.join(labels_dir, f)) for f in files]
    )
    errors = filter(
        lambda _error: reduce(
            lambda x, y: x or y,
            [e in _error for e in ERROR_DICTIONARY[error]]
        ),
        unq_errors
    )

    print(
        'ratio of half buildings:\n   - among Unqualified errors: ',
        len(errors) / float(len(unq_errors)),
        '\n   - among all files: ',
        len(errors) / float(len(files))
    )


def main():
    labels_dir = '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/export-3DS/_labels'
    files = fnmatch.filter(os.listdir(labels_dir), '*.shp')

    map(
        lambda error: unqualified_errors_statistics(error, labels_dir, files),
        UNQUALIFIED_ERROR_LIST
    )


if __name__ == '__main__':
    main()
