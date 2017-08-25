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
    'mis_segmentation': ['mis_segmentation'],
    'slope': ['slope'],
    'footprint': ['footprint'],
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

BUILDING_ERROR_LIST = [
    'over_segmentation',
    'under_segmentation',
    'footprint'
]

FACET_ERROR_LIST = [
    'over_segmentation',
    'under_segmentation',
    'mis_segmentation',
    'slope'
]

ERROR_CATEGORY_DICTIONARY = {
    'Unqualified': UNQUALIFIED_ERROR_LIST,
    'Building': BUILDING_ERROR_LIST,
    'Facet': FACET_ERROR_LIST
}

ERROR_CATEGORY_INDEX = {
    'Unqualified': 0,
    'Building': 1,
    'Facet': 2
}


def read(filename):
    records = shapefile.Reader(filename).records()
    return [(feature[-3], feature[-2], feature[-1]) for feature in records]


def simplify_errors_per_building(filename):
    return map(lambda t: list(set(t)), zip(*read(filename)))


def errors_per_building(filename, error_category):
    return lint(
        simplify_errors_per_building(filename)[
            ERROR_CATEGORY_INDEX[error_category]
        ]
    )


def lint(errors):
    linted = filter(
        lambda error: error not in ERROR_DICTIONARY['None'],
        errors
    )
    return linted if linted != [] else 'None'


def errors_statistics(error, error_category, labels_dir, files):
    category_errors = filter(
        lambda _error: _error not in ERROR_DICTIONARY['None'],
        [
            errors_per_building(os.path.join(labels_dir, f), error_category)
            for f
            in files
        ]
    )
    errors = filter(
        lambda _error: reduce(
            lambda x, y: x or y,
            [e in _error for e in ERROR_DICTIONARY[error]]
        ),
        category_errors
    )

    return (
        len(errors) / float(len(category_errors)),
        len(errors) / float(len(files))
    )


def print_statistics(error, error_category, labels_dir, files):
    x, y = errors_statistics(error, error_category, labels_dir, files)
    print(
        'ratio of \'',
        error,
        '\' error :\n   - among \'',
        error_category,
        '\' errors: ',
        x,
        '\n   - among all files: ',
        y
    )


def summarize_statistics():
    pass


def main():
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )
    files = fnmatch.filter(os.listdir(labels_dir), '*.shp')

    print('Facet:', ERROR_CATEGORY_DICTIONARY['Facet'])
    print(
        'Facet:',
        map(
            lambda error: errors_statistics(
                error,
                'Facet',
                labels_dir,
                files
            ),
            ERROR_CATEGORY_DICTIONARY['Facet']
        )
    )


if __name__ == '__main__':
    main()
