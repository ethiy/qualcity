#! /usr/bin/env python2
# -*- coding: <utf-8> -*-

from __future__ import print_function

import operator

import os
import fnmatch

import shapefile

SHAPEFILE = "ESRI Shapefile"

ERROR_DICTIONARY = {
    'over_segmentation': [
        'over_segmentation',
        'over',
        'over_segmantation',
        'over_segermentation',
        'oversegmentation',
        'over_segmentationover_segmentation'
    ],
    'under_segmentation': ['under_segmentation', 'under', 'unde_segmentation'],
    'mis_segmentation': ['mis_segmentation', 'missegmentation', 'mis'],
    'slope': ['slope'],
    'footprint': ['footprint', 'footprint_error'],
    'too_low': ['too_low'],
    'None': ['None', ''],
    'half_building': ['half_building', 'half_bulding'],
    'changed': ['changed'],
    'Unknown': ['Unknown'],
    'occlusion': ['occlusion', 'vegetation']
}

UNQUALIFIED_ERROR_LIST = [
    'half_building',
    'changed',
    'occlusion',
    'Unknown'
]

BUILDING_ERROR_LIST = [
    'over_segmentation',
    'under_segmentation',
    'footprint',
    'too_low'
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

CLASSES = {
    0: 'None',
    1: 'Unqualified',
    2: 'Building',
    3: 'Facet'
}

INV_CLASSES = {v: k for k, v in CLASSES.iteritems()}


def labels_map(directory):
    return {
        os.path.splitext(shape)[0]: INV_CLASSES[
                error_classes(
                    os.path.join(directory, shape),
                    5
                )
            ]
        for shape in fnmatch.filter(
            os.listdir(directory),
            '*.shp'
        )
    }


def entry(error):
    for _error, synonyms in ERROR_DICTIONARY.iteritems():
        if error in synonyms:
            return _error
    raise LookupError


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


def lint(errors):
    linted = {
        error: score
        for error, score in errors.iteritems()
        if error not in ERROR_DICTIONARY['None']
    }
    return linted if linted != {} else 'None'


def unify_errors(filename):
    unique_errors = map(lambda t: list(set(t)), read(filename))
    max_score_errors = map(
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
    return map(
        lint,
        max_score_errors
    )


def class_error(errors):
    return 0 if errors == 'None' else max(errors.values())


def errors_per_building(filename):
    return map(
        lambda _unified_errors: {
            error: reduce(
                max,
                [
                    _unified_errors[synonym]
                    for synonym in ERROR_DICTIONARY[error]
                    if synonym in _unified_errors.keys()
                ]
            )
            for error in set(
                [entry(key) for key in _unified_errors.keys()]
            )
        }
        if _unified_errors != 'None' else 'None',
        unify_errors(filename)
    )


def error_class_score(filename):
    return [
        class_error(error)
        for error in errors_per_building(filename)
    ]


def error_classes(filename, threshold):
    unq, bul, fac = error_class_score(filename)
    if unq >= threshold:
        return 'Unqualified'
    elif bul >= threshold:
        return 'Building'
    elif fac >= threshold:
        return 'Facet'
    else:
        return 'None'


def exists(sub_error, error_type, errors, threshold):
    if errors[ERROR_CATEGORY_INDEX[error_type]] == 'None':
        return 0
    elif sub_error not in errors[ERROR_CATEGORY_INDEX[error_type]].keys():
        return 0
    else:
        return int(
            errors[ERROR_CATEGORY_INDEX[error_type]][sub_error] >= threshold
        )


def error_array(filename, threshold, error_type):
    errs = errors_per_building(filename)
    return map(
        lambda sub_error: exists(sub_error, error_type, errs, threshold),
        ERROR_CATEGORY_DICTIONARY[error_type]
    )


def error_arrays(filename, threshold):
    return [
        error_array(filename, threshold, error_type)
        for error_type in ['Unqualified', 'Building', 'Facet']
    ]


def get_category_errors(filename, error_category):
    return errors_per_building(filename)[
        ERROR_CATEGORY_INDEX[error_category]
    ]


def error_couple(filename, first, second):
    return (
        get_category_errors(filename, first),
        get_category_errors(filename, second)
    )


def similtaneous_errors_list(labels_dir, files, first, second):
    return filter(
        lambda couple: couple[0] not in ERROR_DICTIONARY['None']
        and couple[1] not in ERROR_DICTIONARY['None'],
        [
            error_couple(os.path.join(labels_dir, f), first, second)
            for f in files
        ]
    )


def similtaneous_errors_lists():
    pass


def similtaneous_errors_scores(labels_dir, files, first, second):
    return [
        (max(dic_couple[0].values()), max(dic_couple[1].values()))
        for dic_couple
        in similtaneous_errors_list(labels_dir, files, first, second)
    ]


def score_similtaneous_errors(labels_dir, files, first, second):
    return float(
        reduce(
            lambda x, y: x + y,
            [
                min(couple)
                for couple
                in similtaneous_errors_scores(labels_dir, files, first, second)
            ]
        )
    ) / 10.


def print_similtaneous_summary(labels_dir, files, first, second):
    joint = score_similtaneous_errors(labels_dir, files, first, second)
    print(
        second + ' errors and ' + first + ' errors joint probability: ',
        joint / float(len(files)),
        '\n' + second + ' errors probability knowing ' + first + ' errors: ',
        joint / score_category_errors(first, labels_dir, files),
        '\nBuilding errors  knowing no Facet errors',

    )


def list_category_errors(error_category, labels_dir, files):
    return filter(
        lambda _error: _error not in ERROR_DICTIONARY['None'],
        [
            get_category_errors(os.path.join(labels_dir, f), error_category)
            for f
            in files
        ]
    )


def score_category_errors(error_category, labels_dir, files):
    return float(
        reduce(
            operator.add,
            [
                max(dic.values())
                for dic in list_category_errors(
                    error_category,
                    labels_dir,
                    files
                )
            ]
        )
    ) / 10.


def errors_statistics(error, error_category, labels_dir, files):
    category_errors = list_category_errors(error_category, labels_dir, files)
    number_of_category_errors = score_category_errors(
        error_category,
        labels_dir,
        files
    )
    errors = filter(
        lambda _errors: reduce(
            lambda x, y: x or y,
            [error in _errors.keys()]
        ),
        category_errors
    )
    number_of_errors = float(
        reduce(
            operator.add,
            [max(dic.values()) for dic in errors]
        )
    ) / 10.

    return (
        number_of_errors / number_of_category_errors,
        number_of_errors / float(len(files))
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


def print_statistics_summary(error_category, labels_dir, files):
    print(
        'ratio of \'',
        error_category,
        '\' errors among all errors:',
        score_category_errors(error_category, labels_dir, files)
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

    print (errors_per_building(os.path.join(labels_dir, files[0])))
    print (error_arrays(os.path.join(labels_dir, files[0]), 5))


if __name__ == '__main__':
    main()
