#! /usr/bin/env python2
# -*- coding: <utf-8> -*-

from __future__ import print_function

import operator

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


def set_score(error):
    raw = error.split(': ')
    if len(raw) == 2:
        return (raw[0], int(raw[1]))
    elif len(raw) == 1:
        return (raw[0], 10)
    else:
        raise IOError


def list_errors(errors):
    return [set_score(error) for error in errors.split(', ')]


def get_errors(feature):
    return (
        list_errors(feature[-3]),
        list_errors(feature[-2]),
        list_errors(feature[-1])
    )


def read(filename):
    records = shapefile.Reader(filename).records()
    return reduce(
        lambda lhs, rhs: (
            lhs[0] + rhs[0],
            lhs[1] + rhs[1],
            lhs[2] + rhs[2]
        ),
        [get_errors(feature) for feature in records]
    )


def unify_errors(filename):
    unique_errors = map(lambda t: list(set(t)), read(filename))
    return map(
        lambda _unique_errors: {
            error: reduce(
                max,
                [
                    _couple[1]
                    for _couple
                    in filter(
                        lambda couple: couple[0] == error,
                        _unique_errors
                    )
                ]
            )
            for error in dict(_unique_errors).keys()
        },
        unique_errors
    )


# def errors_per_building(filename):
#     map(
#         ,
#         unique_errors
#     )
#     return


def errors_per_building(filename, error_category):
    return errors_per_building(filename)[
        ERROR_CATEGORY_INDEX[error_category]
    ]


def error_couple(filename):
    return (
        errors_per_building(filename, 'Building'),
        errors_per_building(filename, 'Facet')
    )


def lint(errors):
    linted = filter(
        lambda error: error not in ERROR_DICTIONARY['None'],
        errors
    )
    return linted if linted != [] else 'None'


def list_category_errors(error_category, labels_dir, files):
    return filter(
        lambda _error: _error not in ERROR_DICTIONARY['None'],
        [
            errors_per_building(os.path.join(labels_dir, f), error_category)
            for f
            in files
        ]
    )


def errors_statistics(error, error_category, labels_dir, files):
    category_errors = list_category_errors(error_category, labels_dir, files)
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


def print_statistics_summury(error_category, labels_dir, files):
    print(
        'ratio of \'',
        error_category,
        '\' errors among all errors:',
        len(
            list_category_errors(error_category, labels_dir, files)
        )
        /
        float(len(files))
    )
    map(
        lambda error: print_statistics(
            error,
            error_category,
            labels_dir,
            files
        ),
        ERROR_CATEGORY_DICTIONARY[error_category]
    )


def summarize_statistics(error_category, labels_dir, files):
    print(error_category, ':', ERROR_CATEGORY_DICTIONARY[error_category])
    print(
        error_category,
        ':',
        map(
            lambda error: errors_statistics(
                error,
                error_category,
                labels_dir,
                files
            ),
            ERROR_CATEGORY_DICTIONARY[error_category]
        )
    )


def main():
    labels_dir = os.path.join(
        '/home/ethiy/Data/Elancourt/Bati3D/EXPORT_1246-13704/',
        'export-3DS/_labels'
    )
    files = fnmatch.filter(os.listdir(labels_dir), '*.shp')

    # map(
    #     lambda error_category: print_statistics_summury(
    #         error_category, labels_dir, files
    #     ),
    #     ERROR_CATEGORY_INDEX.keys()
    # )

    print({
        _file: unify_errors(os.path.join(labels_dir, _file))
        for _file in files
    })


if __name__ == '__main__':
    main()
